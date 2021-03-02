import datetime
from PIL import Image
import io, cv2, os
import numpy as np
from collections import namedtuple

from datetime import date
import time
from tqdm import tqdm
from typing import Mapping, Any, List

import requests
from requests import models
from requests.adapters import HTTPAdapter
from requests.models import Response
from bs4 import BeautifulSoup

import torch
from src.model import CNN
from src.dataset import Data
import pandas as pd


class HTTPConfig:
    BASE_URL = "https://irs.thsrc.com.tw"
    BOOKING_PAGE_URL = "https://irs.thsrc.com.tw/IMINT/?locale=tw"
    SUBMIT_FORM_URL = "https://irs.thsrc.com.tw/IMINT/;jsessionid={}?wicket:interface=:{}:BookingS1Form::IFormSubmitListener"
    CONFIRM_TRAIN_URL = (
        "https://irs.thsrc.com.tw/IMINT/?wicket:interface=:{}:BookingS2Form::IFormSubmitListener"
    )
    CONFIRM_TICKET_URL = (
        "https://irs.thsrc.com.tw/IMINT/?wicket:interface=:{}:BookingS3Form::IFormSubmitListener"
    )

    class HTTPHeader:
        USER_AGENT = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:77.0) Gecko/20100101 Firefox/77.0"
        )
        ACCEPT_HTML = "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
        ACCEPT_IMG = "image/webp,*/*"
        ACCEPT_LANGUAGE = "zh-TW,zh;q=0.8,en-US;q=0.5,en;q=0.3"
        ACCEPT_ENCODING = "gzip, deflate, br"

        # Host URL
        BOOKING_PAGE_HOST = "irs.thsrc.com.tw"


class HTTPRequest:
    def __init__(self, max_retries: int = 3) -> None:
        self.sess = requests.Session()
        self.sess.mount("https://", HTTPAdapter(max_retries=max_retries))
        self.page_count = 0

        self.common_head_html: dict = {
            "Host": HTTPConfig.HTTPHeader.BOOKING_PAGE_HOST,
            "User-Agent": HTTPConfig.HTTPHeader.USER_AGENT,
            "Accept": HTTPConfig.HTTPHeader.ACCEPT_HTML,
            "Accept-Language": HTTPConfig.HTTPHeader.ACCEPT_LANGUAGE,
            "Accept-Encoding": HTTPConfig.HTTPHeader.ACCEPT_ENCODING,
        }

    def request_booking_page(self) -> Response:
        return self.sess.get(
            HTTPConfig.BOOKING_PAGE_URL, headers=self.common_head_html, allow_redirects=True
        )

    def request_security_code_img(self, book_page: bytes) -> Response:
        img_url = parse_security_img_url(book_page)
        return self.sess.get(img_url, headers=self.common_head_html)

    def submit_booking_form(self, params: Mapping[str, Any]) -> Response:
        url = HTTPConfig.SUBMIT_FORM_URL.format(self.sess.cookies["JSESSIONID"], self.page_count)
        return self.sess.post(
            url, headers=self.common_head_html, params=params, allow_redirects=True
        )


BOOKING_PAGE: Mapping[str, Any] = {
    "security_code_img": {"id": "BookingS1Form_homeCaptcha_passCode"}
}
ERROR_FEEDBACK: Mapping[str, Any] = {"name": "span", "attrs": {"class": "feedbackPanelERROR"}}


def parse_security_img_url(html: bytes) -> str:
    page = BeautifulSoup(html, features="html.parser")
    element = page.find(**BOOKING_PAGE["security_code_img"])
    return HTTPConfig.BASE_URL + element["src"]


class AbstractViewModel:
    def __init__(self) -> None:
        pass

    def parse(self, html: bytes) -> List[Any]:
        raise NotImplementedError

    def _parser(self, html: bytes) -> BeautifulSoup:
        return BeautifulSoup(html, features="html.parser")


Error = namedtuple("Error", ["msg"])


class ErrorFeedback(AbstractViewModel):
    def __init__(self) -> None:
        super(ErrorFeedback, self).__init__()
        self.errors: List[Error] = []

    def parse(self, html: bytes) -> List[Error]:
        page = self._parser(html)
        items = page.find_all(**ERROR_FEEDBACK)
        for it in items:
            self.errors.append(Error(it.text))

        return self.errors


if __name__ == "__main__":
    model = CNN()
    model.load("checkpoints/0228_ori/model.pth")
    model.cuda()
    model.eval()

    target_path = "dataset/crawled_data"
    if not os.path.exists(target_path):
        os.makedirs(target_path)
        os.makedirs(os.path.join(target_path, "ori"))

    crawled_data = []
    image_cnt = 1
    error_cnt = 1
    num_runs = 10000

    for _ in tqdm(range(num_runs)):
        client = HTTPRequest()
        error_feedback = ErrorFeedback()

        params = {
            "BookingS1Form:hf:0": "",
            "selectStartStation": 2,
            "selectDestinationStation": 12,
            "trainCon:trainRadioGroup": 0,
            "seatCon:seatRadioGroup": "radio17",
            "bookingMethod": 0,
            "toTimeInputField": "2021/02/28",
            "toTimeTable": "730A",
            "toTrainIDInputField": "",
            "backTimeInputField": "2021/02/28",
            "backTimeTable": "",
            "backTrainIDInputField": "",
            "ticketPanel:rows:0:ticketAmount": "1F",
            "ticketPanel:rows:1:ticketAmount": "0H",
            "ticketPanel:rows:2:ticketAmount": "0W",
            "ticketPanel:rows:3:ticketAmount": "0E",
            "ticketPanel:rows:4:ticketAmount": "0P",
            "homeCaptcha:securityCode": "",
        }

        params["toTimeInputField"] = (datetime.date.today() + datetime.timedelta(days=1)).strftime(
            "%Y/%m/%d"
        )
        params["backTimeInputField"] = (
            datetime.date.today() + datetime.timedelta(days=1)
        ).strftime("%Y/%m/%d")

        try:
            book_page = client.request_booking_page()
            img_resp = client.request_security_code_img(book_page.content)
            image_ori = np.array(Image.open(io.BytesIO(img_resp.content)))

            image = cv2.resize(image_ori, (128, 128))
            with torch.no_grad():
                image = (
                    cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)[np.newaxis, np.newaxis, :, :] / 255.0
                )
                image = torch.tensor(image, dtype=torch.float32).cuda()
                code = CNN.decode(model(image))[0]
                pred = Data.decode(code)

            params["homeCaptcha:securityCode"] = "".join(pred)

            ### show debug image
            # cv2.putText(image_ori, "".join(pred), (20, 20), 3, 1, (255, 0, 255))
            # cv2.imshow("img", image_ori)
            # cv2.waitKey(0)

            result = client.submit_booking_form(params)
            errors = error_feedback.parse(result.content)
            if len(errors) == 0:
                cv2.imwrite(
                    os.path.join(target_path, "ori", str(image_cnt).zfill(5) + ".jpg"), image_ori
                )
                crawled_data.append([str(image_cnt).zfill(5) + ".jpg", "".join(pred)])
                image_cnt += 1
            else:
                cv2.imwrite("dataset/error/" + str(error_cnt).zfill(5) + ".jpg", image_ori)
                print("Accuracy:", (image_cnt - 1) / (image_cnt + error_cnt - 2))
                error_cnt += 1

            time.sleep(2)
        except:
            pass

    output = pd.DataFrame(crawled_data)
    output.to_csv(
        os.path.join(target_path, "labels.csv"), header=["filename", "label"], index=False
    )

