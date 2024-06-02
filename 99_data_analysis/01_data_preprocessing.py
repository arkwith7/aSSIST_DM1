import openai
import re
import pandas as pd

# OpenAI API 키 설정
openai.api_key = 'your-api-key'

# 특수 문자 제거 함수
def clean_text(text):
    text = re.sub(r'[^ㄱ-ㅎ가-힣a-zA-Z0-9\s]', '', text)
    return text

# 언어 감지 함수
def detect_language(text):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"다음 텍스트의 주요 언어를 감지해 주세요: {text}",
        max_tokens=5
    )
    return response.choices[0].text.strip()

# 텍스트 임베딩 생성 함수
def get_chatgpt_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']

# 데이터 전처리
data = pd.read_csv('/mnt/data/youtube_channel_comments_data_20240602_033255.csv')
data['cleaned_commentText'] = data['commentText'].apply(clean_text)
data['language'] = data['cleaned_commentText'].apply(detect_language)

# 텍스트 임베딩 생성
data['embedding'] = data['cleaned_commentText'].apply(get_chatgpt_embedding)

# 결과 저장
data.to_csv('/mnt/data/processed_youtube_comments.csv', index=False)

# 데이터 확인
print(data.head())
