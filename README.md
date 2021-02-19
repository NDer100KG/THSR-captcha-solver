# THSR-captcha-solver

**僅供學術研究用途，請勿使用於其他用途。**

The original training codes come from [y252328/THSR-captcha-solver](https://github.com/y252328/THSR-captcha-solver), and codes for generating captcha comes from [BreezeWhite/THSR-Ticket](https://github.com/BreezeWhite/THSR-Ticket). Thanks for their good work!

- [THSR-captcha-solver](#thsr-captcha-solver)
  - [Introduction](#introduction)
  - [Installation](#installation)
  - [Preperation](#preperation)
  - [Training](#training)
  - [Testing](#testing)
  - [Reference](#reference)

## Introduction

This project uses the convolutional neural network(CNN) to solve the captcha in Taiwan High Speed Rail booking website, and uses pytorch to implement it.

## Installation

### Conda
```
conda env create -f environment.yml
conda activate THSR
```

### pip 
```
pip install -r requirements.txt
```


## Preperation
1. Down Font from the internet and place in folder `font`
Captcha generation: captcha and labels will be generated in `captcha/`
```
python generate_captcha.py
```


## Training
``` 
python src/train.py
```
預設每個epoch會存一個checkpoint

## Testing
可以使用main.py裡的test function做單張圖的inference。


## Reference
[1] [simple-railway-captcha-solver](https://github.com/JasonLiTW/simple-railway-captcha-solver)\
[2] [[爬蟲實戰] 如何破解高鐵驗證碼 (1) - 去除圖片噪音點?](https://youtu.be/6HGbKdB4kVY)\
[3] [[爬蟲實戰] 如何破解高鐵驗證碼 (2) - 使用迴歸方法去除多餘弧線?](https://youtu.be/4DHcOPSfC4c)\
[4] [pytorch-book
](https://github.com/chenyuntc/pytorch-book)
