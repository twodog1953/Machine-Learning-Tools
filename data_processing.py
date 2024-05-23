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


def hour_cal(in_col, out_col, if_simp):
    # input start time/end time col (series)
    # output: (calculated) time col, with hour amount rounded to 2nd digits
    diff = (out_col.apply(lambda x: datetime.strptime(str(x), "%H:%M")) -
            in_col.apply(lambda x: datetime.strptime(str(x), "%H:%M")))
    diff = diff.apply(lambda x: round(timedelta.total_seconds(x) / 3600, 2))
    if if_simp == False:
        diff.loc[diff < 0] = diff.loc[diff < 0] + 24
    return diff


def rounding(sst, pit, set, pot):
    # input: scheduled start time, clock in time, scheduled end time, clock out time (in series/pd col format)
    st_diff = hour_cal(sst, pit, True)
    et_diff = hour_cal(set, pot, True)
    # temporarily assign clock time to real time
    real_st = pit.copy()
    real_et = pot.copy()
    # find index of start times that's in 10min range of scheduled range
    # then adjust real time with same index to scheduled times for rounding
    # note: no need to add exception, as it won't change time data on original df!
    real_st_i = st_diff[(st_diff > -0.17) & (st_diff < 0.17)].index
    real_st[real_st_i] = sst[real_st_i]
    real_et_i = st_diff[(et_diff > -0.17) & (et_diff < 0.17)].index
    real_et[real_et_i] = set[real_et_i]
    # lastly calculate hourly amount with real time
    out = hour_cal(real_st, real_et, True)
    return out, real_st, real_et


def date_range_cal(yy, mm):
    # calculate first day and last day of a month in a year
    # input: year, month
    # output: first day, last day (format: string, 'month/day/year')
    yy = int(yy)
    mm = int(mm)
    # Get the first day of the month
    first_day = datetime(yy, mm, 1)
    # Get the last day of the month
    last_day = datetime(yy, mm, calendar.monthrange(yy, mm)[1])
    first_day = first_day.strftime('%m/%d/%y')
    last_day = last_day.strftime('%m/%d/%y')
    return first_day, last_day


def hol(hol_day, day_col, in_col, out_col, holot_col):
    # reassign hours calculated from hour_cal to reg_hour and hol_hour
    # input: holiday date (datetime obj, in dic{})
    # output: hol/ot hour col (prior to OT)
    hol_day = datetime.strptime(hol_day, "%m/%d/%Y")
    hol_st = hol_day
    hol_et = hol_day + timedelta(days=1)

    # hol_st = datetime.strftime(hol_st, "%m/%d/%Y")
    # hol_et = datetime.strftime(hol_et, "%m/%d/%Y")

    day_col = day_col.apply(lambda x: datetime.strftime(x, "%m/%d/%Y"))

    now_st = pd.Series(map(lambda x, y: datetime.strptime(x + " " + y, "%m/%d/%Y %H:%M"), day_col, in_col))
    now_et = pd.Series(map(lambda x, y: datetime.strptime(x + " " + y, "%m/%d/%Y %H:%M"), day_col, out_col))
    now_et[now_et < now_st] += timedelta(days=1)

    overlap_st = now_st.apply(lambda x: max(x, hol_st))
    overlap_et = now_et.apply(lambda x: min(x, hol_et))
    overlap_hrs = pd.Series(map(lambda x, y: round((y - x).total_seconds() / 3600, 2), overlap_st, overlap_et))
    overlap_hrs = overlap_hrs.apply(lambda x: max(0, x))
    valid_overlap_st = overlap_st[overlap_hrs != 0]
    valid_overlap_et = overlap_et[overlap_hrs != 0]
    valid_overlap_i = valid_overlap_st.index
    hol_df = pd.DataFrame({"hop_in": valid_overlap_st,
                           "hop_out": valid_overlap_et,
                           "i": valid_overlap_i})

    # deal with situation where OT needs to merge with Hol Hours
    otin_col = pd.Series(map(lambda x, y: x - timedelta(hours=float(y)), now_et, holot_col))
    otout_col = now_et[otin_col.index].copy()
    ot_df = pd.DataFrame({"ot_in": otin_col,
                          "ot_out": otout_col,
                          "i": otin_col.index})
    ot_hop_match = pd.merge(hol_df, ot_df, how="left", on="i")
    ot_hop_match = ot_hop_match.dropna(subset=["ot_in", "ot_out"])
    # print(ot_hop_match)
    if ot_hop_match.empty == False:
        overlap_otst = pd.Series(map(lambda x, y: max(x, y), ot_hop_match["hop_in"], ot_hop_match["ot_in"]))
        overlap_otet = pd.Series(map(lambda x, y: min(x, y), ot_hop_match["hop_out"], ot_hop_match["ot_out"]))
        deduct_ind = ot_hop_match["i"].tolist()
        overlap_othrs = pd.Series(map(lambda x, y: round((y - x).total_seconds() / 3600, 2), overlap_otst, overlap_otet))
        overlap_othrs = overlap_othrs.apply(lambda x: max(0, x))
        overlap_othrs.index = deduct_ind

        overlap_hrs[deduct_ind] = overlap_hrs[deduct_ind] - overlap_othrs
        # print(overlap_hrs[deduct_ind])
    return overlap_hrs


def if_weekday(d_col):
    # input: df col of date
    # output: bool of date col, determine if its weekday
    wd = d_col.apply(lambda x: pd.to_datetime(x).weekday())
    out = wd < 5
    return out


def add_rate(pos_col, s_col, rate_table):
    # input: position col, site col, rate table
    # rate table format: {site1: {position1: [$$, $$$], position2: [$$, $$$]},
    #                     site2: {position1: [$$, $$$], position2: [$$, $$$]}}
    # output:
    col_len = int(len(pos_col))
    reg_col = pd.Series(np.zeros(col_len))
    othol_col = pd.Series(np.zeros(col_len))

    for s in rate_table:
        for pos in rate_table[s]:
            reg_col.loc[(pos_col.str.contains(pos, case=False)) &
                        (s_col.str.contains(s))] = rate_table[s][pos][0]
            othol_col.loc[(pos_col.str.contains(pos, case=False)) &
                          (s_col.str.contains(s))] = rate_table[s][pos][1]
    return reg_col, othol_col