import math
import jieba
import wordcloud
import stylecloud
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


def ciyun(text_input, file_output):
    stylecloud.gen_stylecloud(
        text=text_input,  # 上面分词的结果作为文本传给text参数
        size=512,
        max_words=300,
        font_path='msyh.ttc',  # 字体设置
        palette='cartocolors.qualitative.Pastel_7',  # 调色方案选取，从palettable里选择
        gradient='horizontal',  # 渐变色方向选了垂直方向
        # icon_name='fab fa-weixin',  # 蒙版选取，从Font Awesome里选
        output_name=file_output  # 输出词云图
    )
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
df = df[~df['sentence'].isin(list_kouling)].reset_index(drop=True)
print(df.shape)

tqdm.pandas()
df['情感'] = df.progress_apply(lambda row: SnowNLP(row['sentence']).sentiments, axis=1)
df['分词'] = df.progress_apply(lambda row: SnowNLP(row['sentence']).words, axis=1)

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

fig = px.line(df_pvt_second, x='秒钟', y='弹幕条数/秒')
path_fig = os.path.join(dir_img, '弹幕条数-秒')
fig.write_html(f'{path_fig}.html')
fig.write_image(f'{path_fig}.png')

df_pvt_minute = pd.pivot_table(df, index='minute', values=['情感'], aggfunc=['mean'])
df_pvt_minute.columns = ['弹幕情感/分']
df_pvt_minute['分钟'] = df_pvt_minute.index

fig = px.line(df_pvt_minute, x='分钟', y='弹幕情感/分')
path_fig = os.path.join(dir_img, '弹幕情感-分')
fig.write_html(f'{path_fig}.html')
fig.write_image(f'{path_fig}.png')

df_pvt_second = pd.pivot_table(df, index='second', values=['情感'], aggfunc=['mean'])
df_pvt_second.columns = ['弹幕情感/秒']
df_pvt_second['秒钟'] = df_pvt_second.index

fig = px.line(df_pvt_second, x='秒钟', y='弹幕情感/秒')
path_fig = os.path.join(dir_img, '弹幕情感-秒')
fig.write_html(f'{path_fig}.html')
fig.write_image(f'{path_fig}.png')

## 词云测试
contents = []
[contents.extend(_) for _ in df['分词']]
# words_remove = ['中', '中中', '啊', '哈哈']
# contents = [_ for _ in contents if _ not in words_remove]
contents = ' '.join(contents)
path_res = os.path.join(dir_res, 'test.png')
ciyun(contents, path_res)