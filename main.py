import datetime
import math
import os.path

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

from wordcloud import WordCloud
from path_config import *
from func_supp import *

# 数据载入
path_barrage = os.path.join(dir_raw, '12S.xml')
df = barrage_parse(path_barrage)
df['sentence'] = [_.upper() for _ in df['sentence']]
print(f"原始弹幕数量:{df.shape[0]}")

df['sentence'].value_counts()[:20]

path_meaningless = os.path.join(dir_root, '常见无意义弹幕.txt')
list_meaningless = pd.read_csv(path_meaningless, header=None)[0].tolist()
list_kouling = ['第一代骁龙8+移动平台', '小米影像战略升级']

list_kouling = [_.upper() for _ in list_kouling]
slc_kouling = df['sentence'].str.contains('|'.join(list_kouling))

df = df[~slc_kouling].reset_index(drop=True)
print(f"去除口令后弹幕数量:{df.shape[0]}")

tqdm.pandas()
# 分词与情绪识别
file_fix = os.path.join(dir_root, '自定义词.txt')
jieba.load_userdict(file_fix)
df['分词'] = df.progress_apply(lambda row: jieba.lcut_for_search(row['sentence'].upper()), axis=1)
df['情感'] = df.progress_apply(lambda row: SnowNLP(row['sentence']).sentiments, axis=1)

# 情绪识别统计图
data = [go.Histogram(x=df['情感'], nbinsx=20)]
fig = go.Figure(data)
path_fig = os.path.join(dir_img, '情感直方图')
fig.write_html(f'{path_fig}.html')
print(f"情绪得分均值：{df['情感'].mean()}")

# 数据格式梳理
df['minute'] = df['time'].apply(lambda x: math.floor(x / 60))
df['second'] = df['time'].apply(lambda x: math.floor(x))

# 分时统计与导出
pvt_time_cnt_min, _fig = pvt_time(
    data=df, index='minute', values=['sentence'], aggfunc=['count'],
    name_col='弹幕条数', time_type='分')

pvt_time_cnt_sec, fig_time_cnt_sec = pvt_time(
    data=df, index='second', values=['sentence'], aggfunc=['count'],
    name_col='弹幕条数', time_type='秒', window=121, signals=['波峰'])

pvt_time_cnt_sec_2 = pd.pivot_table(df[df['情感'] >= 0.55], index='second', values=['sentence'], aggfunc=['count'])
pvt_time_cnt_sec_2.columns = ['正面弹幕']
pvt_time_cnt_sec_2[f'平滑'] = signal.savgol_filter(pvt_time_cnt_sec_2.iloc[:, 0], window_length=121, polyorder=3)
pvt_time_cnt_sec_2[f'平滑'] = signal.savgol_filter(pvt_time_cnt_sec_2[f'平滑'], window_length=121, polyorder=3)
pvt_time_cnt_sec_1 = pd.pivot_table(
    df[(df['情感'] > 0.5) & (df['情感'] < 0.55)], index='second', values=['sentence'], aggfunc=['count'])
pvt_time_cnt_sec_1.columns = ['中性弹幕']
pvt_time_cnt_sec_1[f'平滑'] = signal.savgol_filter(pvt_time_cnt_sec_1.iloc[:, 0], window_length=121, polyorder=3)
pvt_time_cnt_sec_1[f'平滑'] = signal.savgol_filter(pvt_time_cnt_sec_1[f'平滑'], window_length=121, polyorder=3)
pvt_time_cnt_sec_0 = pd.pivot_table(df[df['情感'] < 0.5], index='second', values=['sentence'], aggfunc=['count'])
pvt_time_cnt_sec_0.columns = ['负面弹幕']
pvt_time_cnt_sec_0[f'平滑'] = signal.savgol_filter(pvt_time_cnt_sec_0.iloc[:, 0], window_length=121, polyorder=3)
pvt_time_cnt_sec_0[f'平滑'] = signal.savgol_filter(pvt_time_cnt_sec_0[f'平滑'], window_length=121, polyorder=3)

df_temp = pd.concat([pvt_time_cnt_sec_0, pvt_time_cnt_sec_1, pvt_time_cnt_sec_2], axis=1).fillna(0)
fig = px.area(df_temp, x=df_temp.index, y=['负面弹幕', '中性弹幕', '正面弹幕'], title='弹幕数量堆积图')
fig = px.line(df_temp, x=df_temp.index, y=['负面弹幕', '中性弹幕', '正面弹幕'], title='弹幕数量曲线图')
# fig.show()
fig_time_cnt_sec.add_trace(go.Scatter(x=pvt_time_cnt_sec_2.index, y=pvt_time_cnt_sec_2[f'平滑'], name='正面弹幕'))
fig_time_cnt_sec.add_trace(go.Scatter(x=pvt_time_cnt_sec_1.index, y=pvt_time_cnt_sec_1[f'平滑'], name='中性弹幕'))
fig_time_cnt_sec.add_trace(go.Scatter(x=pvt_time_cnt_sec_0.index, y=pvt_time_cnt_sec_0[f'平滑'], name='负面弹幕'))
# fig_time_cnt_sec.show()

path_save = os.path.join(dir_img, '三情感分析.html')
fig_time_cnt_sec.write_html(path_save)
pvt_time_emo_min, _fig = pvt_time(
    data=df, index='minute', values=['情感'], aggfunc=['mean'],
    name_col='弹幕情感', time_type='分')

pvt_time_emo_sec, _fig = pvt_time(
    data=df, index='second', values=['情感'], aggfunc=['mean'],
    name_col='弹幕情感', time_type='秒', window=181, signals=['波峰', '波谷'])

# 按时间段输出
path_time_period = os.path.join(dir_raw, '时间段.xlsx')
df_time_period = pd.read_excel(path_time_period)
df_time_period['sec_beg'] = df_time_period.progress_apply(
    lambda row: row['开始时间'].hour * 3600 + row['开始时间'].minute * 60 + row['开始时间'].second, axis=1)
df_time_period['sec_end'] = df_time_period.progress_apply(
    lambda row: row['结束时间'].hour * 3600 + row['结束时间'].minute * 60 + row['结束时间'].second, axis=1)
# df_time_period['sec_persist'] = df_time_period['sec_end'] - df_time_period['sec_beg']
df_time_period['弹幕数量'] = df_time_period.progress_apply(
    lambda row: df.query(f"time<={row['sec_end']} and time>={row['sec_beg']}").shape[0], axis=1)
df_time_period['弹幕数量/分钟'] = df_time_period.progress_apply(
    lambda row: row['弹幕数量'] / (row['sec_end'] - row['sec_beg']), axis=1)
df_time_period['情感均值'] = df_time_period.progress_apply(
    lambda row: df.query(f"time<={row['sec_end']} and time>={row['sec_beg']}")['情感'].mean(), axis=1)
df_time_period['积极占比'] = df_time_period.progress_apply(
    lambda row: df.query(f"time<={row['sec_end']} and time>={row['sec_beg']} and 情感>=0.55").shape[0] / row['弹幕数量'],
    axis=1)
df_time_period['中性占比'] = df_time_period.progress_apply(
    lambda row: df.query(f"time<={row['sec_end']} and time>={row['sec_beg']} and 情感<0.55 and 情感>=0.5").shape[0] / row[
        '弹幕数量'], axis=1)
df_time_period['消极占比'] = df_time_period.progress_apply(
    lambda row: df.query(f"time<={row['sec_end']} and time>={row['sec_beg']} and 情感<0.5").shape[0] / row['弹幕数量'],
    axis=1)
# df_time_period['关键词'] = df_time_period.progress_apply(
#     lambda row: df.query(f"time<={row['sec_end']} and time>={row['sec_beg']}")['分词'], axis=1)
path_time_period = os.path.join(dir_res, '分产品.xlsx')
df_time_period.to_excel(path_time_period)

# 按关键词输出
path_kw_emo = os.path.join(dir_raw, '分产品.xlsx')
df_kw_emo = pd.read_excel(path_kw_emo)
df_kw_emo['关键词'] = df_kw_emo.apply(
    lambda row: '|'.join(str(row['关键词']).split()), axis=1)
df_kw_emo['涉及弹幕数量'] = df_kw_emo.apply(
    lambda row: df['sentence'].str.contains(row['关键词']).sum(), axis=1)
df_kw_emo['情感均值'] = df_kw_emo.apply(
    lambda row: df[df['sentence'].str.contains(row['关键词'])]['情感'].mean(), axis=1)
df_kw_emo['积极占比'] = df_kw_emo.progress_apply(
    lambda row: ((df['sentence'].str.contains(row['关键词'])) & (df['情感'] >= 0.55)).sum() / row['涉及弹幕数量'],
    axis=1)
df_kw_emo['中性占比'] = df_kw_emo.progress_apply(
    lambda row: ((df['sentence'].str.contains(row['关键词'])) & (df['情感'] >= 0.5) & (df['情感'] < 0.55)).sum() / row[
        '涉及弹幕数量'],
    axis=1)
df_kw_emo['消极占比'] = df_kw_emo.progress_apply(
    lambda row: ((df['sentence'].str.contains(row['关键词'])) & (df['情感'] < 0.5)).sum() / row['涉及弹幕数量'],
    axis=1)

path_res = os.path.join(dir_res, 'kw_emo.xlsx')
df_kw_emo.to_excel(path_res)

## 词云测试
jieba.load_userdict(file_fix)
df['分词'] = df.progress_apply(lambda row: jieba.lcut_for_search(row['sentence']), axis=1)
contents = []
[contents.extend(_) for _ in df['分词']]
words_remove = ['一代', '中奖中奖', '中', '中中', '啊', '哈哈', '版本', '永不', '不为', 'LCD 永不为奴', '家伙', '没有']
path_stop_word = os.path.join(dir_root, 'cn_stopwords.txt')
list_stop_word = pd.read_csv(path_stop_word, header=None)[0].tolist()
words_remove += list_kouling
words_remove += list_stop_word
contents = [_ for _ in contents if _ not in words_remove]
contents = [_.strip().upper() for _ in contents]
contents = [_ for _ in contents if len(_) > 1]
contents = ' '.join(contents)

font_path = r'C:\Windows\Fonts\simhei.ttf'
wc = WordCloud(collocations=False, font_path=font_path, width=1400, height=1400, margin=2,
               max_words=120, max_font_size=300, mode='RGBA', background_color='white', repeat=False).generate(contents)
path_res = os.path.join(dir_res, '全体词云.png')
wc.to_file(path_res)

# 不同情感关键词云
word_pos_irrelevant = pd.read_csv(os.path.join(dir_root, 'irrelevant_positive.txt'), header=None)[0].tolist()
all_words = [x for j in df.query(f"情感>=0.55")['分词'] for x in j]
all_words = [_ for _ in all_words if str(_).strip() not in list_stop_word + word_pos_irrelevant]
path_emo_kw = os.path.join(dir_res, 'emo_positive_kw.xlsx')
pd.Series(all_words).value_counts().to_excel(path_emo_kw)
contents = ' '.join([_ for _ in all_words if len(_) > 1])
wc = WordCloud(collocations=False, font_path=font_path, width=1400, height=1400, margin=2,
               max_words=60, max_font_size=300, mode='RGBA', background_color='white', repeat=False).generate(contents)
path_res = os.path.join(dir_res, '正面词云.png')
wc.to_file(path_res)

word_neg_irrelevant = pd.read_csv(os.path.join(dir_root, 'irrelevant_negative.txt'), header=None)[0].tolist()
all_words = [x for j in df.query(f"情感<=0.5")['分词'] for x in j]
all_words = [_ for _ in all_words if str(_).strip() not in list_stop_word + word_neg_irrelevant]
path_emo_kw = os.path.join(dir_res, 'emo_negative_kw.xlsx')
pd.Series(all_words).value_counts().to_excel(path_emo_kw)
contents = ' '.join([_ for _ in all_words if len(_) > 1])
wc = WordCloud(collocations=False, font_path=font_path, width=1400, height=1400, margin=2,
               max_words=60, max_font_size=300, mode='RGBA', background_color='white', repeat=False).generate(contents)
path_res = os.path.join(dir_res, '负面词云.png')
wc.to_file(path_res)
