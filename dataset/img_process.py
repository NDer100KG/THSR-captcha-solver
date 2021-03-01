from PIL import Image
import cv2, os

from skimage import transform, data
import matplotlib.pyplot as plt

import numpy as np
from sklearn.preprocessing import binarize
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

dic = [[0] * 2 for i in range(100)]
for i in range(100):
    dic[i][0] = 25
    dic[i][1] = 25
dic[50][0] = 26
dic[50][1] = 24
dic[48][0] = 23
dic[48][1] = 30
dic[46][0] = 27
dic[46][1] = 25
dic[45][0] = 21
dic[45][1] = 30


def process(dir):
    try:
        img = cv2.imread(dir)  # (45, 133, 3)
        if dir.endswith(".png"):
            dir = dir.replace("png", "jpg")
        dir = dir.replace("ori", "img_processed")
        dir = "test.jpg"
        dst = cv2.fastNlMeansDenoisingColored(img, None, 31, 31, 7, 21)
        height1, width1, channels1 = img.shape  # get img height and width

        plt.figure(figsize=(width1, height1), dpi=100)
        plt.axis("off")
        plt.imshow(dst)
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)  # 去白邊

        plt.savefig(dir, dpi=10)
        img2 = cv2.imread(dir)  # (450, 1330, 3)

        ret, thresh = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY_INV)  # 黑白化
        plt.imshow(thresh)
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)  # 去白邊
        # get img height and width
        height, width, channels = thresh.shape

        imgarr = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)

        imgarr[:, 100 : width - 40] = 0
        imagedata = np.where(imgarr == 255)  # find where are white

        X = np.array([imagedata[1]])
        Y = height - imagedata[0]

        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression

        poly_reg = PolynomialFeatures(degree=2)
        X_ = poly_reg.fit_transform(X.T)
        regr = LinearRegression()
        regr.fit(X_, Y)

        X2 = np.array([[i for i in range(0, width)]])
        X2_ = poly_reg.fit_transform(X2.T)

        for ele in np.column_stack([regr.predict(X2_).round(0), X2[0],]):
            pos = height - int(ele[0])
            thresh[pos - int(dic[height1][0]) : pos + int(dic[height1][1]), int(ele[1])] = (
                255 - thresh[pos - int(dic[height1][0]) : pos + int(dic[height1][1]), int(ele[1])]
            )  # 這裡可以更改回歸線條上下範圍

        plt.imshow(thresh)
        newdst = transform.resize(thresh, (48, 140))  # resize (h,w)
        plt.close()  # close figure of origine size
        plt.figure(figsize=(140, 48), dpi=100)  # 存成固定大小 (w, h)
        plt.axis("off")
        plt.imshow(newdst)

        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)  # 去白邊
        plt.savefig(dir, dpi=1)

        plt.close()
    except:
        print("err")


def process2(dir):
    img = cv2.imread(dir)
    if dir.endswith(".png"):
        dir = dir.replace("png", "jpg")
    dir = dir.replace("ori", "img_processed")
    dst = cv2.fastNlMeansDenoisingColored(img, None, 31, 31, 7, 21)
    dst = cv2.resize(dst, 10 * (dst.shape[0], dst.shape[1]))
    height1, width1, channels1 = img.shape  # get img height and width

    ret, thresh = cv2.threshold(dst, 127, 255, cv2.THRESH_BINARY_INV)  # 黑白化
    height, width, channels = thresh.shape

    imgarr = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)

    imgarr[:, 100 : width - 40] = 0
    imagedata = np.where(imgarr == 255)  # find where are white

    X = np.array([imagedata[1]])
    Y = height - imagedata[0]

    poly_reg = PolynomialFeatures(degree=2)
    X_ = poly_reg.fit_transform(X.T)
    regr = LinearRegression()
    regr.fit(X_, Y)

    X2 = np.array([[i for i in range(0, width)]])
    X2_ = poly_reg.fit_transform(X2.T)

    for ele in np.column_stack([regr.predict(X2_).round(0), X2[0],]):
        pos = height - int(ele[0])
        thresh[pos - int(dic[height1][0]) : pos + int(dic[height1][1]), int(ele[1])] = (
            255 - thresh[pos - int(dic[height1][0]) : pos + int(dic[height1][1]), int(ele[1])]
        ) 

    cv2.resize(thresh, (48, 140))
    cv2.imwrite("test.jpg", thresh)


if __name__ == "__main__":
    from glob import glob
    from tqdm import tqdm

    folder_path = "dataset/test/ori"

    all_images = sorted(glob(folder_path + "/*.png"))
    # print(all_images)

    pbar = tqdm(all_images)
    for image in pbar:
        pbar.set_description(image)
        process(image)
