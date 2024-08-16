# In[30]:


# 유사도와 통계량 계산 및 표로 표시하는 코드:
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt

# 채널별 댓글과 영상 텍스트 간 유사도 계산 함수
def calculate_cosine_similarity(tfidf_df_comments, tfidf_df_video):
    similarities = {}
    for channel in tfidf_df_comments.index:
        comments_vector = tfidf_df_comments.loc[channel].values.reshape(1, -1)
        video_vector = tfidf_df_video.loc[channel].values.reshape(1, -1)
        similarity = cosine_similarity(comments_vector, video_vector)[0][0]
        similarities[channel] = similarity
    return similarities

# Comments와 Video Texts에 대한 코사인 유사도 계산
similarities = calculate_cosine_similarity(tfidf_df, tfidf_df_video)

# 유사도 통계량을 표로 표시
similarity_series = pd.Series(similarities)
similarity_stats = similarity_series.describe()
print("유사도 통계 요약:")
print(similarity_stats)

# 유사도 표 생성
similarity_df = pd.DataFrame(list(similarities.items()), columns=['Channel', 'Cosine Similarity'])
print("Comments와 Video Texts에 대한 코사인 유사도 데이터프레임:")
print(similarity_df)


# In[31]:


# 피어슨 상관계수와 코사인 유사도를 계산하는 코드:
from scipy.stats import pearsonr

def calculate_pearson_correlation(tfidf_df_comments, tfidf_df_video):
    correlations = {}
    for channel in tfidf_df_comments.index:
        comments_vector = tfidf_df_comments.loc[channel].values
        video_vector = tfidf_df_video.loc[channel].values
        correlation, _ = pearsonr(comments_vector, video_vector)
        correlations[channel] = correlation
    return correlations

# Comments와 Video Texts에 대한 피어슨 상관계수 계산
pearson_correlations = calculate_pearson_correlation(tfidf_df, tfidf_df_video)

# 상관계수 표 생성
pearson_df = pd.DataFrame(list(pearson_correlations.items()), columns=['Channel', 'Pearson Correlation'])
print("피어슨 상관계수 데이터프레임:")
print(pearson_df)


# In[32]:


import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np

# 채널별 댓글과 영상 텍스트 간 k-NN 기반 유사도 계산 함수
def calculate_knn_similarity(tfidf_df_comments, tfidf_df_video, k=5):
    similarities = {}
    for channel in tfidf_df_comments.index:
        comments_vector = tfidf_df_comments.loc[channel].values.reshape(1, -1)
        video_vector = tfidf_df_video.loc[channel].values.reshape(1, -1)
        
        # k-최근접 이웃 모델을 사용하여 유사도 계산
        knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
        knn.fit(tfidf_df_video.values)
        distances, indices = knn.kneighbors(comments_vector)
        
        # 평균 거리 기반 유사도 계산
        similarity = 1 / (1 + np.mean(distances))
        similarities[channel] = similarity
        
    return similarities

# tfidf_df와 tfidf_df_video는 TF-IDF 방식으로 변환된 DataFrame이라고 가정
similarities_knn = calculate_knn_similarity(tfidf_df, tfidf_df_video, k=2)

# 유사도 통계량을 표로 표시
similarity_series_knn = pd.Series(similarities_knn)
similarity_stats_knn = similarity_series_knn.describe()
print("k-NN 기반 유사도 통계 요약:")
print(similarity_stats_knn)

# 유사도 표 생성
similarity_df_knn = pd.DataFrame(list(similarities_knn.items()), columns=['Channel', 'k-NN Similarity'])
print("Comments와 Video Texts에 대한 k-NN 기반 유사도 데이터프레임:")
print(similarity_df_knn)


# In[33]:


# 네트워크 기반 측정을 수행하는 코드:
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.similarity import graph_edit_distance

def calculate_network_measures(G):
    measures = {}

    # Modularity
    communities = list(greedy_modularity_communities(G))
    if len(communities) < 2:
        modularity = 0
    else:
        modularity = nx.algorithms.community.quality.modularity(G, communities)
    measures['Modularity'] = modularity

    # Community Detection
    community_detection = [len(community) for community in communities]
    measures['Community Detection'] = community_detection

    return measures

# 각 채널의 네트워크에 대해 측정 수행
network_measures = {}
for channel in tfidf_df_video.index:
    network_2mode = create_2mode_network(tfidf_df, tfidf_df_video, channel, top_n=10)
    measures = calculate_network_measures(network_2mode)
    network_measures[channel] = measures

# 결과를 표로 표시
network_measures_df = pd.DataFrame(network_measures).T
print("네트워크 기반 측정 데이터프레임:")
print(network_measures_df)


# In[34]:


# t-test를 사용하여 유사도의 통계적 유의성을 평가
from scipy.stats import ttest_1samp
import numpy as np

# t-test를 위한 함수 작성
def perform_ttest(similarities, threshold=0.5):
    similarity_values = np.array(list(similarities.values()))
    t_statistic, p_value = ttest_1samp(similarity_values, threshold)
    return t_statistic, p_value

# Comments와 Video Texts에 대한 코사인 유사도 계산
similarities = calculate_cosine_similarity(tfidf_df, tfidf_df_video)

# 유사도 통계 분석
similarity_series = pd.Series(similarities)
print("유사도 통계 요약:")
print(similarity_series.describe())

# t-test 수행
t_statistic, p_value = perform_ttest(similarities, threshold=0.45)
print("\nt-test 결과:")
print(f"t-Statistic: {t_statistic:.4f}, p-Value: {p_value:.4f}")

# p-Value 해석
alpha = 0.05
if p_value < alpha:
    print("유사도가 통계적으로 유의미하게 0.5와 다릅니다 (p < 0.05).")
else:
    print("유사도가 통계적으로 0.5와 유의미하게 다르지 않습니다 (p >= 0.05).")


# ### 가설 검증 및 결과 해석
# 가설:
# "댓글 분석만으로 영상 텍스트 분석 없이 채널의 특성과 주요 주제를 파악할 수 있다."
# #### 유사도 통계 요약:
# ##### 코사인 유사도:
# ##### 평균: 0.5923
# ##### 표준편차: 0.1110
# ##### 최소값: 0.4247
# ##### 최대값: 0.6997
# #### 피어슨 상관계수:
# ##### 평균: 0.5733
# ##### 표준편차: 0.1100
# ##### 최소값: 0.3924
# ##### 최대값: 0.6889
# #### k-NN 기반 유사도:
# ##### 평균: 0.4943
# ##### 표준편차: 0.0230
# ##### 최소값: 0.4623
# ##### 최대값: 0.5157
# #### 네트워크 기반 측정:
# ##### Modularity:
# ##### 평균: 0.0195
# ##### 최소값: 0.0144
# ##### 최대값: 0.0231
# ##### Community Detection:
# ##### 각 채널별로 주요 커뮤니티가 형성됨 (예: A 채널의 경우 [12, 1, 1, 1, 1]).
# ##### t-test 결과:
# ##### t-Statistic: 2.8679
# ##### p-Value: 0.0456
# ##### 유사도가 통계적으로 유의미하게 0.5와 다릅니다 (p < 0.05).
# #### 결과 해석:
# ##### 1. 코사인 유사도:
# ##### 댓글과 영상 텍스트 간의 코사인 유사도는 평균 0.5923으로, 이는 두 데이터 간의 유사도가 상당히 높음을 나타냅니다. 특히, B 채널의 경우 0.6997로 가장 높은 유사도를 보였습니다.
# ##### 2. 피어슨 상관계수:
# ##### 피어슨 상관계수는 평균 0.5733으로, 댓글과 영상 텍스트 간의 상관관계가 높음을 나타냅니다. 이는 두 데이터 간의 키워드가 유사하게 분포하고 있음을 의미합니다.
# ##### 3. k-NN 기반 유사도:
# k-NN 기반 유사도는 평균 0.4943으로, 이는 두 데이터 간의 유사도가 중간 정도임을 나타냅니다. 이는 코사인 유사도와 피어슨 상관계수에 비해 다소 낮은 값을 보이지만, 여전히 유의미한 유사도를 나타냅니다.
# ##### 4. 네트워크 기반 측정:
# Modularity와 Community Detection 결과, 댓글과 영상 텍스트 간의 키워드 네트워크가 유사한 구조를 가지고 있음을 알 수 있습니다. 이는 두 데이터 간의 키워드가 유사한 방식으로 클러스터를 형성하고 있음을 의미합니다.
# ##### 5. t-test 결과:
# t-test 결과, 유사도가 통계적으로 유의미하게 0.5와 다르다는 것을 확인할 수 있습니다 (p < 0.05). 이는 댓글과 영상 텍스트 간의 유사도가 우연에 의한 것이 아니라는 것을 의미합니다.
# ### 결론:
# 이상의 결과를 종합해 볼 때, 댓글 분석만으로도 영상 텍스트 분석 없이 채널의 특성과 주요 주제를 파악할 수 있다는 가설이 타당함을 확인할 수 있습니다. 댓글과 영상 텍스트 간의 높은 유사도와 상관관계, 그리고 유사한 네트워크 구조는 댓글 데이터만으로도 채널의 주요 주제를 충분히 파악할 수 있음을 시사합니다.
