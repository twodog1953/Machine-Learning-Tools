# the tools used for data preprocessing and transformation, derived from class and work projects.
import os
import glob
import pandas as pd
from datetime import datetime, timedelta
import openpyxl
from openpyxl.workbook import Workbook
from openpyxl.styles import Font, Alignment
from openpyxl.styles.borders import Border, Side
from openpyxl.utils.dataframe import dataframe_to_rows as dtr
import calendar
import numpy as np


# all functions
def read_from_txt(file_path):
    # input: the path of the .txt file
    # output: list containing contents in txt file line by line
    f = open(file_path, "r", encoding='utf-8')
    data = f.read()
    txt_linebyline = data.split("\n")
    f.close()
    return txt_linebyline


def hour_cal(in_col, out_col):
    # input start time/end time col (series)
    # output: (calculated) time col, with hour amount rounded to 2nd digits
    diff = (out_col.apply(lambda x: datetime.strptime(str(x), "%H:%M")) -
           in_col.apply(lambda x: datetime.strptime(str(x), "%H:%M")))
    diff = diff.apply(lambda x: round(timedelta.total_seconds(x) / 3600, 2))
    return diff