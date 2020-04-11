#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import io
import json
import os
import os.path
import re
import csv
import subprocess
import cv2  # OpenCVのインポート
import pandas as pd
import datetime

from PIL import Image
from math import sqrt, exp, log
from matplotlib import cm
from matplotlib import pyplot as plt

import numpy as np

import time
import math

class ImageSegmentation:

    def __init__(self, is_debug=False):
        self.is_debug = is_debug
        self.temp_val = 25
        self.csv_filename = ""
        self.img_pixel_list = []

        # ゼロ埋めするか？（するならTrue）
        self.zfill_flag = True
        # 任意の温度以上を指定
        self.temp_val = 30

    pass

    def get_img_pixel_list(self):
        """
        Return img_pixel_list
        :return:
        """
        return self.img_pixel_list

    # 縦方向分割数split_heght = 4
    # 横方向分割数 split_width = 4
    def img_split(self, im,w,h, split_heght, split_width):
        # 読み込んだ画像の高さと幅を指定分割数で割る
        height = h / split_heght
        width = w / split_width
    
        # 縦の分割枚数
        for h1 in range(split_heght):
            # 横の分割枚数
            for w1 in range(split_width):
                w2 = w1 * width
                h2 = h1 * height
                # 分割して検証する座標を取得
                self.img_pixel_list.append([w2, h2, width + w2, height + h2])
                yield im.crop((w2, h2, width + w2, height + h2))


    # 分割元ファイルパス file_name
    # 出力フォルダ名 out_path
    # 縦方向分割数split_heght = 4
    # 横方向分割数 split_width = 4
    def split_image(self, out_path, file_name, split_heght, split_width):
        # 画像の読み込み
        im = Image.open(file_name)
        w = im.size[0]
        h = im.size[1]
        length = math.log10(split_heght * split_width) + 1
        os.makedirs(out_path, exist_ok=True)
        for number, ig in enumerate(self.img_split(im,w,h, split_heght, split_width), 1):
            # 出力
            if self.zfill_flag:
                ig.save(out_path + "/" + str(number).zfill(int(length)) + ".PNG", "PNG")
            else:
                ig.save(out_path + "/" + str(number) +".PNG", "PNG")
        
        # figure()でグラフを表示する領域をつくり，figオブジェクトにする．
        fig, axes_list = plt.subplots(split_heght, split_width, figsize=(split_heght,split_width), dpi=200)

        img_name = 1
        for ax in axes_list.ravel():
            # 描画の表示設定
            ax.axis('off')
            flir_img_file_name = "./" + out_path + "/" + str(img_name).zfill(2) + ".PNG"
            bin_img = Image.open(flir_img_file_name)

            ax.imshow(bin_img, cmap=plt.cm.gray)
            img_name = img_name + 1

        # 描画
        plt.subplots_adjust(wspace=0.01, hspace=-0.45)
        plt.show()


    # CSVファイルパス csv_file_name
    # 分割ピクセルリスト img_pixel_list
    def average_temperature(self, csv_file_name, img_pixel_list):

        csv_file = open(csv_file_name, "r", encoding="ms932", errors="", newline="" )
        #リスト形式
        f1 = csv.reader(csv_file, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
        header = next(f1) 

        tmp_avr_list = []
        cn =  1
        csv_list = list(f1)
        for img_pixel in img_pixel_list:
            tmp_sum =  0.0
            tmp_sum_count = 0
            for row in csv_list:
                if img_pixel[0] <= int(row[0]) < img_pixel[2]:
                    if img_pixel[1] <= int(row[1]) < img_pixel[3]:
                        tmp_sum = tmp_sum + float(row[2])
                        tmp_sum_count+=1
            cn+=1
            tmp_avr_list.append(round(tmp_sum/tmp_sum_count,3))

        return tmp_avr_list


    # 温度比較 temp_list
    def diff_temperature(self, temp_list):

        # CSVから温度を16分割したそれぞれの温度を算出
        temp_base = temp_list[0]
        temp_target = temp_list[1]

        temp_diff = []
        temp_rate = []
        
        # 16分割固定で確認
        item_count = 0
        while item_count < 16:
            temp_diff.append(round((temp_base[item_count] - temp_target[item_count]),2))
            temp_rate.append(round((temp_base[item_count]/temp_target[item_count])/temp_base[item_count],2))
            item_count+=1
        
        temp_list.append(temp_diff)
        temp_list.append(temp_rate)

        return temp_list


    # -- -- -- --
    # -- メイン --
    # -- -- -- --
    # 縦方向分割数 split_heght = 4
    # 横方向分割数 split_width = 4
    def excute_analysis(self, f_nm_ext, csv_file_name, split_heght, split_width):
        f_nm,ext = os.path.splitext(os.path.basename(f_nm_ext))

        self.img_pixel_list = []
        # 画像分割
        self.split_image(f_nm, f_nm_ext, split_heght, split_width)
        # 平均算出
        return self.average_temperature(csv_file_name, self.get_img_pixel_list())
