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


path_barrage_1 = os.path.join(dir_raw, '1.xml')
path_barrage_2 = os.path.join(dir_raw, '2.xml')
df_1 = barrage_parse(path_barrage_1)
df_2 = barrage_parse(path_barrage_2)
time_delta = df_1.iloc[-1]['time']
df_2['time'] += time_delta
df = pd.concat([df_1, df_2], axis=0).reset_index(drop=True)
print(df.shape)

list_kouling = ['一代神U', 'LCD永不为奴', 'Redmi联名阿童木', '512G版本来了', '全体起立']
df = df[~df['sentence'].str.strip().isin(list_kouling)].reset_index(drop=True)
print(df.shape)


tqdm.pandas()
# 情感
df['情感'] = df.progress_apply(lambda row: SnowNLP(row['sentence']).sentiments, axis=1)
# df['分词'] = df.progress_apply(lambda row: SnowNLP(row['sentence']).words, axis=1)
data = [go.Histogram(x=df['情感'], nbinsx=20)]
fig = go.Figure(data)
path_fig = os.path.join(dir_img, '情感直方图')
fig.write_html(f'{path_fig}.html')
df['情感'].mean()

# 分词
file_fix = os.path.join(dir_root, '自定义词.txt')
jieba.load_userdict(file_fix)
df['分词'] = df.progress_apply(lambda row: jieba.lcut_for_search(row['sentence']), axis=1)

df['minute'] = df['time'].apply(lambda x: math.floor(x / 60))
df['second'] = df['time'].apply(lambda x: math.floor(x))

df_pvt_minute = pd.pivot_table(df, index='minute', values=['sentence'], aggfunc=['count'])
df_pvt_minute.columns = ['弹幕条数/分']
df_pvt_minute['分钟'] = df_pvt_minute.index

fig = px.line(df_pvt_minute, x='分钟', y='弹幕条数/分')
path_fig = os.path.join(dir_img, '弹幕条数-分')
fig.write_html(f'{path_fig}.html')
fig.write_image(f'{path_fig}.png')

df_pvt_second = pd.pivot_table(df, index='second', values=['sentence'], aggfunc=['count'])
df_pvt_second.columns = ['弹幕条数/秒']
df_pvt_second['秒钟'] = df_pvt_second.index
df_pvt_second['时间'] = [sec_trans_time(_) for _ in df_pvt_second['秒钟']]
df_pvt_second['视频时间'] = df_pvt_second['秒钟'].apply(lambda x:
                                                  sec_trans_time(x) if x < time_delta
                                                  else sec_trans_time(x - time_delta))

df_pvt_second = df_pvt_second.reset_index(drop=True)
df_pvt_second['弹幕条数平滑'] = signal.savgol_filter(df_pvt_second['弹幕条数/秒'], window_length=121, polyorder=3)
df_pvt_second['弹幕条数平滑'] = signal.savgol_filter(df_pvt_second['弹幕条数平滑'], window_length=121, polyorder=3)
df_pvt_second['弹幕条数平滑'] = signal.savgol_filter(df_pvt_second['弹幕条数平滑'], window_length=121, polyorder=3)
peaks, _ = signal.find_peaks(df_pvt_second['弹幕条数平滑'], distance=180)
df_pvt_second['波峰'] = 0
df_pvt_second.loc[peaks, '波峰'] = df_pvt_second.loc[peaks, '弹幕条数平滑']
anti_peaks, _ = signal.find_peaks(-df_pvt_second['弹幕条数平滑'], distance=180)
df_pvt_second['波谷'] = 0
df_pvt_second.loc[anti_peaks, '波谷'] = df_pvt_second.loc[anti_peaks, '弹幕条数平滑']

fig = px.line(df_pvt_second, x='秒钟', y=['弹幕条数/秒', '弹幕条数平滑', '波峰'])
path_fig = os.path.join(dir_img, '弹幕条数-秒')
fig.write_html(f'{path_fig}.html')
fig.write_image(f'{path_fig}.png')

path_file = os.path.join(dir_res, '弹幕数量-波峰波谷.xlsx')
# df_pvt_second[((df_pvt_second['波峰'] > 0) | (df_pvt_second['波谷'] > 0))].to_excel(path_file)
df_pvt_second['情感均值'] = None
for _ in df_pvt_second[df_pvt_second['波峰'] > 0].index:
    _win = 30
    _emo = df[((_ - _win) <= df['time']) & (df['time'] <= (_ + 30))]['情感'].mean()
    df_pvt_second.loc[_, '情感均值'] = _emo
df_pvt_second[df_pvt_second['波峰'] > 0].to_excel(path_file)

df_pvt_minute = pd.pivot_table(df, index='minute', values=['情感'], aggfunc=['mean'])
df_pvt_minute.columns = ['弹幕情感/分']
# df_pvt_minute['弹幕情感平滑'] = signal.savgol_filter(df_pvt_minute['弹幕情感/分'], window_length=31, polyorder=5)
df_pvt_minute['分钟'] = df_pvt_minute.index
peaks, _ = signal.find_peaks(df_pvt_minute['弹幕情感/分'], distance=5)
df_pvt_minute['波峰'] = 0
df_pvt_minute.loc[peaks, '波峰'] = df_pvt_minute.loc[peaks, '弹幕情感/分']
anti_peaks, _ = signal.find_peaks(-df_pvt_minute['弹幕情感/分'], distance=180)
df_pvt_minute['波谷'] = 0
df_pvt_minute.loc[anti_peaks, '波谷'] = df_pvt_minute.loc[anti_peaks, '弹幕情感/分']

# fig = px.line(df_pvt_minute, x='分钟', y=['弹幕情感/分', '波峰', '波谷'])
# fig = px.line(df_pvt_minute, x='分钟', y=['弹幕情感/分', '弹幕情感平滑'])
fig = px.line(df_pvt_minute, x='分钟', y=['弹幕情感/分'])
path_fig = os.path.join(dir_img, '弹幕情感-分')
fig.write_html(f'{path_fig}.html')
fig.write_image(f'{path_fig}.png')

path_file = os.path.join(dir_res, '弹幕情感-波峰波谷.xlsx')
df_pvt_minute[((df_pvt_minute['波峰'] > 0) | (df_pvt_minute['波谷'] > 0))].to_excel(path_file)

df_pvt_second = pd.pivot_table(df, index='second', values=['情感'], aggfunc=['mean'])
df_pvt_second.columns = ['弹幕情感/秒']
df_pvt_second['弹幕情感平滑'] = signal.savgol_filter(df_pvt_second['弹幕情感/秒'], window_length=121, polyorder=3)
df_pvt_second['弹幕情感平滑'] = signal.savgol_filter(df_pvt_second['弹幕情感平滑'], window_length=121, polyorder=3)
df_pvt_second['秒钟'] = df_pvt_second.index

fig = px.line(df_pvt_second, x='秒钟', y=['弹幕情感/秒', '弹幕情感平滑'])
path_fig = os.path.join(dir_img, '弹幕情感-秒')
fig.write_html(f'{path_fig}.html')
fig.write_image(f'{path_fig}.png')

## 词云测试
jieba.load_userdict(file_fix)
df['分词'] = df.progress_apply(lambda row: jieba.lcut_for_search(row['sentence']), axis=1)
contents = []
[contents.extend(_) for _ in df['分词']]
words_remove = ['一代', '中奖中奖', '中', '中中', '啊', '哈哈', '版本', '永不', '不为', '家伙', '没有']
path_stop_word = os.path.join(dir_root, 'cn_stopwords.txt')
list_stop_word = pd.read_csv(path_stop_word, header=None)[0].tolist()
words_remove += list_kouling
words_remove += list_stop_word
contents = [_ for _ in contents if _ not in words_remove]
contents = [_.strip().upper() for _ in contents]
contents = [_ for _ in contents if len(_) > 1]
contents = ' '.join(contents)
path_res = os.path.join(dir_res, 'test.png')
ciyun(contents, path_res)
