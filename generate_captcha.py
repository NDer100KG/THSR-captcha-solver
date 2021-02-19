import os
import random
import time
from typing import List, Union

from tqdm import tqdm
import pandas as pd

import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from PIL import Image, ImageFont
from PIL.ImageDraw import Draw  # type: ignore
from sklearn.linear_model import Ridge  # type: ignore
from sklearn.preprocessing import PolynomialFeatures  # type: ignore

CHARS = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"


class GenerateCaptcha:
    def __init__(
            self,
            width: int = 145,
            height: int = 55,
            font_size: int = 50
        ) -> None:
        self._width = width
        self._height = height
        self._font_size = font_size
        self._mode = "L"  # 8-bit pixel
        #self._font = ImageFont.truetype("tahoma.ttf", size=font_size-10)
        self._font = ImageFont.truetype("font/Calibri_Regular.ttf", size=font_size)

    def generate(self) -> Union[Image.Image, List]:
        image = Image.new(self._mode, (self._width, self._height), color=255)
        chars = [s for s in CHARS]
        c_list = np.random.choice(chars, size=4, replace=False)
        image = self.draw_characters(image, c_list)
        image = self.add_arc(image)
        image = self.add_noise(image)
        image = self.add_sp_noise(image)
        return image, c_list

    def add_noise(self, img: Image.Image, color_bound: int = 80) -> Image.Image:
        arr = np.array(img)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                cur_c = arr[i, j]
                c = random.randint(0, color_bound)
                arr[i, j] = cur_c-c if cur_c>color_bound else cur_c+c
        return Image.fromarray(arr)

    def add_sp_noise(self, img: Image.Image, prob: float = 0.03) -> Image.Image:
        arr = np.array(img)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                p = random.random()
                if p < prob:
                    arr[i, j] = 0 if arr[i, j] > 128 else 255
        return Image.fromarray(arr)
    
    def add_arc(self, img: Image.Image) -> Image.Image:
        arr = np.array(img)
        start = random.randint(20, 25)
        diff = random.randint(15, 18)
        y = [start, start-diff//2, start-diff]
        rx = np.array([0, random.randint(32, 38), arr.shape[1]])
        x = PolynomialFeatures(degree=2).fit_transform(rx[:, np.newaxis])
        model = Ridge().fit(x, y)
        xx = np.arange(arr.shape[1])
        x = PolynomialFeatures(degree=2).fit_transform(xx[:, np.newaxis])
        yy = np.round(model.predict(x)).astype('int')
        for i in range(len(xx)):
            ry = range(yy[i]-2, yy[i]+2)
            val = np.where(arr[ry, xx[i]]<128, 255, 0)
            arr[ry, xx[i]] = val
        return Image.fromarray(arr)

    def _draw_character(self, img: Image.Image, c: str) -> Image.Image:
        w, h = self._font_size-6, self._font_size-6 #Draw(img).textsize(c, font=self._font)

        dx = random.randint(0, 6)
        dy = random.randint(0, 6)
        im = Image.new(self._mode, (w + dx, h + dy), color=255)
        Draw(im).text((dx, dy), c, font=self._font, fill=0)

        # rotate
        im = im.crop(im.getbbox())
        im = im.rotate(random.uniform(-10, 5), Image.BILINEAR, expand=1, fillcolor=255)

        # warp
        ddx = w * random.uniform(0.1, 0.2)
        ddy = h * random.uniform(0.1, 0.2)
        x1 = int(random.uniform(-ddx, ddx))
        y1 = int(random.uniform(-ddy, ddy))
        x2 = int(random.uniform(-ddx, ddx))
        y2 = int(random.uniform(-ddy, ddy))
        w2 = w + abs(x1) + abs(x2)
        h2 = h + abs(y1) + abs(y2)
        data = (
            x1, y1,
            -x1, h2 - y2,
            w2 + x2, h2 + y2,
            w2 - x2, -y1,
        )
        im = im.resize((w2, h2))
        im = im.transform((w, h), Image.QUAD, data, fill=255, fillcolor=255)
        return im

    def draw_characters(self, img: Image.Image, chars: List[str]) -> Image.Image:
        images = []
        for c in chars:
            images.append(self._draw_character(img, c))

        text_width = sum([im.size[0] for im in images])

        width = max(text_width, self._width)
        #img = img.resize((width, self._height))

        average = int(text_width / len(chars))
        rand = int(0.1 * average)
        offset = int(average * 0.1)

        table = [150 for i in range(256)]
        for idx, im in enumerate(images):
            w, h = Draw(im).textsize(chars[idx], font=self._font)
            mask = im.point(table)
            img.paste(im, (offset, (self._height - h) // 2), mask)
            offset = offset + w + random.randint(-rand, 0)

        h_offset = 4
        arr = np.array(img)[h_offset:-h_offset, :offset+w//3]
        arr = np.where(arr<255, 0, 255)
        return Image.fromarray(arr.astype(np.int32))

if __name__ == "__main__":
    num_caps = 50000
    save_path = 'captcha'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    debug = False

    captcha = GenerateCaptcha()

    data = []
    for i in tqdm(range(num_caps)):
        img, c_list = captcha.generate()
        
        if debug:
            plt.imshow(np.array(img))
            plt.show(block = True)

        img.convert("RGB").save(os.path.join(save_path, str(i).zfill(5) + ".png"))

        data.append([str(i).zfill(5) + ".png", "".join(c_list)])
    
    data = pd.DataFrame(data)
    data.to_csv(os.path.join(save_path, "label.csv"), header = None, index = False)
    print("Done!")
