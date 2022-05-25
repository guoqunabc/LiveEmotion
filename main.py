import math
import jieba
import pandas as pd
import plotly.express as px

from tqdm import tqdm
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


path_barrage_1 = os.path.join(dir_raw, '1.xml')
path_barrage_2 = os.path.join(dir_raw, '2.xml')
df_1 = barrage_parse(path_barrage_1)
df_2 = barrage_parse(path_barrage_2)
time_delta = df_1.iloc[-1]['time']
df_2['time'] += time_delta
df = pd.concat([df_1, df_2], axis=0).reset_index(drop=True)
print(df.shape)

list_kouling = ['一代神U', 'LCD永不为奴', 'Redmi联名阿童木', '512G版本来了', '全体起立']
df = df[~df['sentence'].isin(list_kouling)].reset_index(drop=True)

tqdm.pandas()
# df['snownlp'] = df.progress_apply(lambda row: SnowNLP(row['sentence']), axis=1)
# df['分词'] = df['snownlp'].apply(lambda x: x.words)
# df['情感'] = df['snownlp'].apply(lambda x: x.sentiments)
df['情感'] = df.progress_apply(lambda row: SnowNLP(row['sentence']).sentiments, axis=1)
# df['分词'] = df['sentence'].apply(lambda x: SnowNLP(x).sentiments)
# df['情感'] = df['sentence'].apply(lambda x: SnowNLP(x).sentiments)
df['情感'].mean()
df['情感'].describe()

df['minute'] = df['time'].apply(lambda x: math.floor(x / 60))
df['second'] = df['time'].apply(lambda x: math.floor(x))

df_pvt_minute = pd.pivot_table(df, index='minute', values=['sentence'], aggfunc=['count'])
df_pvt_minute.columns = ['弹幕条数/分']
df_pvt_minute['分钟'] = df_pvt_minute.index

fig = px.line(df_pvt_minute, x='分钟', y='弹幕条数/分')
path_html = os.path.join(dir_img, '弹幕条数-分.html')
fig.write_html(path_html)

df_pvt_second = pd.pivot_table(df, index='second', values=['sentence'], aggfunc=['count'])
df_pvt_second.columns = ['弹幕条数/秒']
df_pvt_second['秒钟'] = df_pvt_second.index

fig = px.line(df_pvt_second, x='秒钟', y='弹幕条数/秒')
path_html = os.path.join(dir_img, '弹幕条数-秒.html')
fig.write_html(path_html)

df_pvt_minute = pd.pivot_table(df, index='minute', values=['情感'], aggfunc=['mean'])
df_pvt_minute.columns = ['弹幕情感/分']
df_pvt_minute['分钟'] = df_pvt_minute.index

fig = px.line(df_pvt_minute, x='分钟', y='弹幕情感/分')
path_html = os.path.join(dir_img, '弹幕情感-分.html')
fig.write_html(path_html)
