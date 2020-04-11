#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function

from google.colab import files
import datetime
import os

class UploadFiles:

    def upload_files():

        uploaded = files.upload()
        file_name = list(uploaded.keys())[0]
        NOW_TIME = datetime.datetime.now()+ datetime.timedelta(hours=9)
        img_filename_list = []
        csv_filename_list = []
        for f_nm_ext in list(uploaded.keys()):
            f_nm,ext = os.path.splitext(os.path.basename(f_nm_ext))
            img_filename_list.append("./" + f_nm_ext)

            csv_filename = f_nm + "_" + str(NOW_TIME.strftime("%Y%m%d%H%M%S")) +  ".csv"
            csv_filename_list.append("./" + csv_filename)

        print(f_nm_ext)