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

from PIL import Image
from math import sqrt, exp, log
from matplotlib import cm
from matplotlib import pyplot as plt

import numpy as np


class FlirImageExtractor:

    def __init__(self, exiftool_path="exiftool", is_debug=False):
        self.exiftool_path = exiftool_path
        self.is_debug = is_debug
        self.temp_val = 25
        self.flir_img_filename = ""
        self.thermal_filename = ""
        self.csv_filename = ""
        self.image_suffix = "_rgb_image.jpg"
        self.thumbnail_suffix = "_rgb_thumb.jpg"
        self.thermal_suffix = "_thermal.png"
        self.default_distance = 1.0

        # valid for PNG thermal images
        self.use_thumbnail = False
        self.fix_endian = True

        self.rgb_image_np = None
        self.thermal_image_np = None

    pass

    def process_image(self, flir_img_filename, csv_filename, temp_val):
        """
        Given a valid image path, process the file: extract real thermal values
        and a thumbnail for comparison (generally thumbnail is on the visible spectre)
        :param flir_img_filename:
        :return:
        """
        if self.is_debug:
            print("INFO Flir image filepath:{}".format(flir_img_filename))

        if not os.path.isfile(flir_img_filename):
            raise ValueError("Input file does not exist or this user don't have permission on this file")

        self.flir_img_filename = flir_img_filename
        self.csv_filename = csv_filename        
        self.temp_val = temp_val

        if self.get_image_type().upper().strip() == "TIFF":
            # valid for tiff images from Zenmuse XTR
            self.use_thumbnail = True
            self.fix_endian = False

        self.rgb_image_np = self.extract_embedded_image()
        self.thermal_image_np = self.extract_thermal_image()

    def get_image_type(self):
        """
        Get the embedded thermal image type, generally can be TIFF or PNG
        :return:
        """
        meta_json = subprocess.check_output(
            [self.exiftool_path, '-RawThermalImageType', '-j', self.flir_img_filename])
        meta = json.loads(meta_json.decode())[0]

        return meta['RawThermalImageType']

    def get_temp_val(self):
        """
        Return temp_val
        :return:
        """
        return self.temp_val

    def get_flir_img_filename(self):
        """
        Return flir_img_filename
        :return:
        """
        return self.flir_img_filename

    def get_thermal_filename(self):
        """
        Return thermal_filename
        :return:
        """
        return self.thermal_filename

    def get_csv_filename(self):
        """
        Return csv_filename
        :return:
        """
        return self.csv_filename

    def get_rgb_np(self):
        """
        Return the last extracted rgb image
        :return:
        """
        return self.rgb_image_np

    def get_thermal_np(self):
        """
        Return the last extracted thermal image
        :return:
        """
        return self.thermal_image_np

    def extract_embedded_image(self):
        """
        extracts the visual image as 2D numpy array of RGB values
        """
        image_tag = "-EmbeddedImage"
        if self.use_thumbnail:
            image_tag = "-ThumbnailImage"

        visual_img_bytes = subprocess.check_output([self.exiftool_path, image_tag, "-b", self.flir_img_filename])
        visual_img_stream = io.BytesIO(visual_img_bytes)

        visual_img = Image.open(visual_img_stream)
        visual_np = np.array(visual_img)

        return visual_np

    def extract_thermal_image(self):
        """
        extracts the thermal image as 2D numpy array with temperatures in oC
        """

        # read image metadata needed for conversion of the raw sensor values
        # E=1,SD=1,RTemp=20,ATemp=RTemp,IRWTemp=RTemp,IRT=1,RH=50,PR1=21106.77,PB=1501,PF=1,PO=-7340,PR2=0.012545258
        meta_json = subprocess.check_output(
            [self.exiftool_path, self.flir_img_filename, '-Emissivity', '-SubjectDistance', '-AtmosphericTemperature',
             '-ReflectedApparentTemperature', '-IRWindowTemperature', '-IRWindowTransmission', '-RelativeHumidity',
             '-PlanckR1', '-PlanckB', '-PlanckF', '-PlanckO', '-PlanckR2', '-j'])
        meta = json.loads(meta_json.decode())[0]

        # exifread can't extract the embedded thermal image, use exiftool instead
        thermal_img_bytes = subprocess.check_output([self.exiftool_path, "-RawThermalImage", "-b", self.flir_img_filename])
        thermal_img_stream = io.BytesIO(thermal_img_bytes)

        thermal_img = Image.open(thermal_img_stream)
        thermal_np = np.array(thermal_img)

        # raw values -> temperature
        subject_distance = self.default_distance
        if 'SubjectDistance' in meta:
            subject_distance = FlirImageExtractor.extract_float(meta['SubjectDistance'])

        if self.fix_endian:
            # fix endianness, the bytes in the embedded png are in the wrong order
            thermal_np = np.vectorize(lambda x: (x >> 8) + ((x & 0x00ff) << 8))(thermal_np)

        raw2tempfunc = np.vectorize(lambda x: FlirImageExtractor.raw2temp(x, E=meta['Emissivity'], OD=subject_distance,
                                                                          RTemp=FlirImageExtractor.extract_float(
                                                                              meta['ReflectedApparentTemperature']),
                                                                          ATemp=FlirImageExtractor.extract_float(
                                                                              meta['AtmosphericTemperature']),
                                                                          IRWTemp=FlirImageExtractor.extract_float(
                                                                              meta['IRWindowTemperature']),
                                                                          IRT=meta['IRWindowTransmission'],
                                                                          RH=FlirImageExtractor.extract_float(
                                                                              meta['RelativeHumidity']),
                                                                          PR1=meta['PlanckR1'], PB=meta['PlanckB'],
                                                                          PF=meta['PlanckF'],
                                                                          PO=meta['PlanckO'], PR2=meta['PlanckR2']))
        thermal_np = raw2tempfunc(thermal_np)
        return thermal_np

    @staticmethod
    def raw2temp(raw, E=1, OD=1, RTemp=20, ATemp=20, IRWTemp=20, IRT=1, RH=50, PR1=21106.77, PB=1501, PF=1, PO=-7340,
                 PR2=0.012545258):
        """
        convert raw values from the flir sensor to temperatures in C
        # this calculation has been ported to python from
        # https://github.com/gtatters/Thermimage/blob/master/R/raw2temp.R
        # a detailed explanation of what is going on here can be found there
        """

        # constants
        ATA1 = 0.006569
        ATA2 = 0.01262
        ATB1 = -0.002276
        ATB2 = -0.00667
        ATX = 1.9

        # transmission through window (calibrated)
        emiss_wind = 1 - IRT
        refl_wind = 0

        # transmission through the air
        h2o = (RH / 100) * exp(1.5587 + 0.06939 * (ATemp) - 0.00027816 * (ATemp) ** 2 + 0.00000068455 * (ATemp) ** 3)
        tau1 = ATX * exp(-sqrt(OD / 2) * (ATA1 + ATB1 * sqrt(h2o))) + (1 - ATX) * exp(
            -sqrt(OD / 2) * (ATA2 + ATB2 * sqrt(h2o)))
        tau2 = ATX * exp(-sqrt(OD / 2) * (ATA1 + ATB1 * sqrt(h2o))) + (1 - ATX) * exp(
            -sqrt(OD / 2) * (ATA2 + ATB2 * sqrt(h2o)))

        # radiance from the environment
        raw_refl1 = PR1 / (PR2 * (exp(PB / (RTemp + 273.15)) - PF)) - PO
        raw_refl1_attn = (1 - E) / E * raw_refl1
        raw_atm1 = PR1 / (PR2 * (exp(PB / (ATemp + 273.15)) - PF)) - PO
        raw_atm1_attn = (1 - tau1) / E / tau1 * raw_atm1
        raw_wind = PR1 / (PR2 * (exp(PB / (IRWTemp + 273.15)) - PF)) - PO
        raw_wind_attn = emiss_wind / E / tau1 / IRT * raw_wind
        raw_refl2 = PR1 / (PR2 * (exp(PB / (RTemp + 273.15)) - PF)) - PO
        raw_refl2_attn = refl_wind / E / tau1 / IRT * raw_refl2
        raw_atm2 = PR1 / (PR2 * (exp(PB / (ATemp + 273.15)) - PF)) - PO
        raw_atm2_attn = (1 - tau2) / E / tau1 / IRT / tau2 * raw_atm2
        raw_obj = (raw / E / tau1 / IRT / tau2 - raw_atm1_attn -
                   raw_atm2_attn - raw_wind_attn - raw_refl1_attn - raw_refl2_attn)

        # temperature from radiance
        temp_celcius = PB / log(PR1 / (PR2 * (raw_obj + PO)) + PF) - 273.15
        return temp_celcius

    @staticmethod
    def extract_float(dirtystr):
        """
        Extract the float value of a string, helpful for parsing the exiftool data
        :return:
        """
        digits = re.findall(r"[-+]?\d*\.\d+|\d+", dirtystr)
        return float(digits[0])


    # def plot_over(self):
    #     print('指定温度は' + str(self.get_temp_val()) + ' 度以上を表示')
    #     self.plot(False)


    # def plot_less(self):
    #     print('指定温度は' + str(self.get_temp_val()) + ' 度以下を表示')
    #     self.plot(True)

    # def plot(self, is_less):
    def plot(self):
        """
        Plot the rgb + thermal image (easy to see the pixel values)
        :return:
        """
        rgb_np = self.get_rgb_np()
        thermal_np = self.get_thermal_np()
        flir_img_filename = self.get_flir_img_filename()
        self.save_images()
        thermal_filename = self.get_thermal_filename()
        temp_val = self.get_temp_val()
        csv_filename = self.get_csv_filename()

        # 元画像の表示 ----- 
        # 画像の読み込み
        ori_img = np.array( Image.open(flir_img_filename) )

        # 温度計算後の画像プロット ----- 
        # plt.imshow(thermal_np, cmap='hot')

        # 指定温度のみのプロット -----
        thermal_np_tmp_over = np.where((thermal_np >= temp_val),thermal_np,255)
        thermal_np_tmp_less = np.where((thermal_np <= temp_val),thermal_np,255)

        # 指定温度の色変えプロット ----- 
        img_cv = cv2.imread(thermal_filename)
        flir_img_filename_3 = "/content/edit.jpg"
        if img_cv is not None:
            # 禁じ手のループ
            for (i,j), value in np.ndenumerate(self.thermal_image_np):
                if value >= temp_val:
                    b, g, r = img_cv[i, j]
                    if (b, g, r) == (255, 255, 255):
                        continue
                    img_cv[i, j] = b, 0, 0

            cv2.imwrite(flir_img_filename_3, img_cv) 
            # 画像の読み込み
            tmp_img = np.array( Image.open(flir_img_filename_3) )

        # タイトルと画像データをリスト化
        bin_imgs = {'original': ori_img, 'target temperature change color': tmp_img, 'target temperature over': thermal_np_tmp_over, 'target temperature less': thermal_np_tmp_less}

        #figure()でグラフを表示する領域をつくり，figオブジェクトにする．
        fig, axes_list = plt.subplots(2, 2, figsize=(3,3), dpi=250)

        color_conunt_text = ""
        for ax, (label, bin_img) in zip(axes_list.ravel(), bin_imgs.items()):
            self.set_style_plt(ax, label)
            ax.set_title(label,loc='left',fontsize='6')
            ax.imshow(bin_img, cmap=plt.cm.gray)

        plt.show()
        
        self.export_thermal_to_csv(csv_filename)

    def set_style_plt(self, ax, label):
        ax.set_title(label,loc='left',fontsize='4')
        ax.tick_params(labelbottom=False,
                    labelleft=False,
                    labelright=False,
                    labeltop=False)
        ax.tick_params(bottom=False,
                    left=False,
                    right=False,
                    top=False)


    def save_images(self):
        """
        Save the extracted images
        :return:
        """
        rgb_np = self.get_rgb_np()
        thermal_np = self.extract_thermal_image()

        img_visual = Image.fromarray(rgb_np)
        thermal_normalized = (thermal_np - np.amin(thermal_np)) / (np.amax(thermal_np) - np.amin(thermal_np))
        img_thermal = Image.fromarray(np.uint8(cm.inferno(thermal_normalized) * 255))

        fn_prefix, _ = os.path.splitext(self.flir_img_filename)
        thermal_filename = fn_prefix + self.thermal_suffix
        self.thermal_filename = thermal_filename
        image_filename = fn_prefix + self.image_suffix
        if self.use_thumbnail:
            image_filename = fn_prefix + self.thumbnail_suffix

        if self.is_debug:
            print("DEBUG Saving RGB image to:{}".format(image_filename))
            print("DEBUG Saving Thermal image to:{}".format(thermal_filename))

        img_visual.save(image_filename)
        img_thermal.save(thermal_filename)

    def export_thermal_to_csv(self, csv_filename):
        """
        Convert thermal data in numpy to json
        :return:
        """

        with open(csv_filename, 'w') as fh:
            writer = csv.writer(fh, delimiter=',')
            writer.writerow(['x', 'y', 'temp (c)'])

            pixel_values = []
            for e in np.ndenumerate(self.thermal_image_np):
                x, y = e[0]
                c = e[1]
                pixel_values.append([x, y, c])

            writer.writerows(pixel_values)


    def check_thermal_at_csv(self, terget_rang, csv_filename, temp_val_more_than, temp_val_range_more, temp_val_range_less, temp_val_less_than):
        # 画像の温度情報を確認
        data= pd.read_csv(csv_filename)
        total_count = len(data["temp (c)"].dropna(how='any'))
        temp_more = len(data[data["temp (c)"] >= temp_val_more_than].dropna(how='any'))/total_count*100
        temp_less = len(data[data["temp (c)"] <= temp_val_less_than].dropna(how='any'))/total_count*100
        temp_range = len(data[(data["temp (c)"] >= temp_val_range_more) & (data["temp (c)"] <= temp_val_range_less)].dropna(how='any'))/total_count*100

        print("------ ------ ------ ------")
        print("指定温度 [ " + str(temp_val_more_than) + " ℃ ] 以上の割合： "  +  str(round(temp_more, 2)) + " %")
        print("指定温度 [ " + str(temp_val_range_more) + " ℃ ] 以上 [ " + str(temp_val_range_less) + " ℃ ] 以下の割合： "  +  str(round(temp_range, 2)) + " %")
        print("指定温度 [ " + str(temp_val_less_than) + " ℃ ] 以下の割合： "  +  str(round(temp_less, 2)) + " %")
        print("------ ------ ------ ------")
        print("最大温度： " +  str(round(data["temp (c)"].max(), 2)) + " ℃")
        print("最小温度： " +  str(round(data["temp (c)"].min(), 2)) + " ℃")
        print("------ ------ ------ ------")

        print("\n")
        print("指定温度範囲の割合チェック")
        if round(temp_more, 2) < terget_rang:
            self.show_ok()
        else:
            self.show_ng()

        print("\n")
        print("指定温度範囲の割合チェック")
        if round(temp_range, 2) < terget_rang:
            self.show_ok()
        else:
            self.show_ng()

        print("\n")
        print("指定温度以下の割合チェック")
        if round(temp_less, 2) < terget_rang:
            self.show_ok()
        else:
            self.show_ng()

    def show_ok(self):
        self.show_image("/content/Thermimage_demo/img/m_ok.png")
        print("問題ありません。")

    def show_ng(self):
        self.show_image("/content/Thermimage_demo/img/m_ng.png")
        print("異常な数値が検出されました。")


    def show_image(self, image_path):
        # 画像の読み込み
        im = Image.open(image_path)
        # 画像をarrayに変換
        im_list = np.asarray(im)
        # サイズを調整
        plt.figure(figsize=(2, 2))
        # 罫線を非表示
        plt.axis('off')
        # 貼り付け
        plt.imshow(im_list)
        # 表示
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract and visualize Flir Image data')
    parser.add_argument('-i', '--input', type=str, help='Input image. Ex. img.jpg', required=True)
    parser.add_argument('-p', '--plot', help='Generate a plot using matplotlib', required=False, action='store_true')
    parser.add_argument('-exif', '--exiftool', type=str, help='Custom path to exiftool', required=False,
                        default='exiftool')
    parser.add_argument('-csv', '--extractcsv', help='Export the thermal data per pixel encoded as csv file',
                        required=False)
    parser.add_argument('-d', '--debug', help='Set the debug flag', required=False,
                        action='store_true')
    parser.add_argument('-tc', '--check_thermal_at_csv', help='Input image. Ex. img.jpg', required=False)
    args = parser.parse_args()

    fie = FlirImageExtractor(exiftool_path=args.exiftool, is_debug=args.debug)
    fie.process_image(args.input)

    if args.check_thermal_at_csv:
        fie.check_thermal_at_csv(args.input)

    if args.plot:
        fie.plot()

    if args.extractcsv:
        fie.export_thermal_to_csv(args.extractcsv)

    fie.save_images()
