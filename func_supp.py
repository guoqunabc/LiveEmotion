import os
import math
import jieba
import scipy
import wordcloud
import stylecloud
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

from tqdm import tqdm
from scipy import signal
from scipy.misc import derivative
from scipy.interpolate import interp1d
from xml.dom import minidom
from snownlp import SnowNLP
from path_config import *


def barrage_parse(path_file):
    doc = minidom.parse(path_file)
    data_barrage = doc.getElementsByTagName("d")
    dict_data = {'sentence': [_.childNodes[0].data for _ in data_barrage],
                 'time': [float(_.getAttribute('p').split(',')[0]) for _ in data_barrage],
                 # 'type': [_.getAttribute('p').split(',')[1] for _ in data_barrage],
                 'user': [_.getAttribute('user') for _ in data_barrage], }
    _df = pd.DataFrame(dict_data)
    return _df


def sec_trans_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)


def ciyun(text_input, file_output):
    stylecloud.gen_stylecloud(
        text=text_input,  # 上面分词的结果作为文本传给text参数
        size=512,
        max_words=150,
        font_path='msyh.ttc',  # 字体设置
        palette='cartocolors.qualitative.Pastel_7',  # 调色方案选取，从palettable里选择
        gradient='horizontal',  # 渐变色方向选了垂直方向
        # icon_name='fab fa-weixin',  # 蒙版选取，从Font Awesome里选
        output_name=file_output  # 输出词云图
    )
    print('词云图已生成')
    return


def pvt_time(data, index, values, aggfunc, name_col, time_type, window=None, signals=None):
    df_pvt = pd.pivot_table(data, index=index, values=values, aggfunc=aggfunc)
    df_pvt.columns = [f"{name_col}/{time_type}"]
    df_pvt[time_type] = df_pvt.index
    df_pvt['时钟时间'] = [sec_trans_time(_) for _ in df_pvt[time_type]]
    # df_pvt['视频时间'] = df_pvt[time_type].apply(
    #     lambda x: sec_trans_time(x) if x < time_delta else sec_trans_time(x - time_delta))
    df_pvt = df_pvt.reset_index(drop=True)
    y_output = [f"{name_col}/{time_type}"]
    if None is not signals:
        df_pvt[f'{name_col}平滑'] = signal.savgol_filter(
            df_pvt[f"{name_col}/{time_type}"], window_length=window, polyorder=3)
        df_pvt[f'{name_col}平滑'] = signal.savgol_filter(df_pvt[f'{name_col}平滑'], window_length=window, polyorder=3)
        df_pvt[f'{name_col}平滑'] = signal.savgol_filter(df_pvt[f'{name_col}平滑'], window_length=window, polyorder=3)
        peaks, _ = signal.find_peaks(df_pvt[f'{name_col}平滑'], distance=window * 1.5)
        df_pvt['波峰'] = 0
        df_pvt.loc[peaks, '波峰'] = df_pvt.loc[peaks, f'{name_col}平滑']
        anti_peaks, _ = signal.find_peaks(-df_pvt[f'{name_col}平滑'], distance=window * 1.5)
        df_pvt['波谷'] = 0
        df_pvt.loc[anti_peaks, '波谷'] = df_pvt.loc[anti_peaks, f'{name_col}平滑']
        y_output += [f'{name_col}平滑']
        y_output += signals
        path_file = os.path.join(dir_res, f'{name_col}-{time_type}-{"".join(signals)}.xlsx')
        df_pvt['情感均值'] = None
        slc_peaks = ((df_pvt['波峰'] > 0) | (df_pvt['波谷'] > 0))
        for _ in df_pvt[slc_peaks].index:
            _win = 30
            _emo = data[((_ - _win) <= data['time']) & (data['time'] <= (_ + 30))]['情感'].mean()
            df_pvt.loc[_, '情感均值'] = _emo
        df_pvt[slc_peaks].to_excel(path_file)
        print(f'{path_file} 导出成功')

    fig = px.line(df_pvt, x=time_type, y=y_output)
    path_fig = os.path.join(dir_img, f"{name_col}-{time_type}")
    fig.write_html(f'{path_fig}.html')
    fig.write_image(f'{path_fig}.png')
    print(f'{path_fig}图表导出成功')

    return df_pvt, fig
