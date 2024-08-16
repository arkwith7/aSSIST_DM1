#!/usr/bin/env python
# coding: utf-8

# 각 채널별로 channel_video_text_data의 '대표영상텍스트' 컬럼의 TF-IDF 기반 키워드를 추출하고, 이를 각 채널별 댓글 키워드와 비교하여 유사도를 계산하고 시각화

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
from konlpy.tag import Mecab
from collections import Counter
import re


# In[2]:


import matplotlib.pyplot as plt
import matplotlib as mpl

plt.rcParams['font.family'] = 'NanumGothic'
mpl.rcParams['axes.unicode_minus'] = False

plt.plot([-4, -3, -2, -1, 0, 1, 2, 3, 4], [12, 32, -4, 0, 5, 2, 19, 9, 3])
plt.xlabel('x축')
plt.ylabel('y축')
plt.title('제목')
plt.show()


# In[3]:


# 영상 통계 및 댓글 데이터 로드
file_path = 'youtube_channel_comments_data_20240606_104600.csv'
data = pd.read_csv(file_path)
# 영상 텍스트 데이터 로드
video_text_data = pd.read_csv('youtube_channel_video_text_data.csv')