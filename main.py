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
from func_supp import *

# 数据载入
path_barrage_1 = os.path.join(dir_raw, '1.xml')
path_barrage_2 = os.path.join(dir_raw, '2.xml')
df_1 = barrage_parse(path_barrage_1)
df_2 = barrage_parse(path_barrage_2)
time_delta = df_1.iloc[-1]['time']
df_2['time'] += time_delta
df = pd.concat([df_1, df_2], axis=0).reset_index(drop=True)
print(f"原始弹幕数量:{df.shape[0]}")

list_kouling = ['一代神U', 'LCD永不为奴', 'Redmi联名阿童木', '512G版本来了', '全体起立']
slc_kouling = df['sentence'].str.contains('|'.join(list_kouling))
df = df[~slc_kouling].reset_index(drop=True)
print(f"去除口令后弹幕数量:{df.shape[0]}")

tqdm.pandas()
# 分词与情绪识别
file_fix = os.path.join(dir_root, '自定义词.txt')
jieba.load_userdict(file_fix)
df['分词'] = df.progress_apply(lambda row: jieba.lcut_for_search(row['sentence']), axis=1)
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
from func_supp import *
pvt_time(data=df, index='minute', values=['sentence'], aggfunc=['count'],
         name_col='弹幕条数', time_type='分')

pvt_time(data=df, index='second', values=['sentence'], aggfunc=['count'],
         name_col='弹幕条数', time_type='秒', window=121, signals=['波峰'])

pvt_time(data=df, index='second', values=['sentence'], aggfunc=['count'],
         name_col='弹幕条数', time_type='秒', window=121, signals=['波峰'])

pvt_time(data=df, index='minute', values=['情感'], aggfunc=['mean'],
         name_col='弹幕情感', time_type='分')

pvt_time(data=df, index='second', values=['情感'], aggfunc=['mean'],
         name_col='弹幕情感', time_type='秒', window=181, signals=['波峰', '波谷'])


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
