from google.oauth2 import service_account
from googleapiclient.discovery import build
from datetime import datetime, timedelta
import pandas as pd

# API 정보 설정
API_NAME = 'youtubeAnalytics'
API_VERSION = 'v2'
SERVICE_ACCOUNT_FILE = 'service_account.json'  # 다운로드한 서비스 계정 파일 경로

# 스코프 설정
SCOPES = ['https://www.googleapis.com/auth/youtube.readonly']
credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES)

# YouTube Analytics API 빌드
youtubeAnalytics = build(API_NAME, API_VERSION, credentials=credentials)

# 날짜 계산
months = 12
end_date = datetime.now().replace(day=1).strftime('%Y-%m-%d')
start_date = (datetime.now().replace(day=1) - timedelta(days=30 * months)).replace(day=1).strftime('%Y-%m-%d')
print("start_date: ", start_date)
print("end_date: ", end_date)

# 채널 ID 목록, 채널소유자로 부터 동의를 얻어야 수집이 가능한 정보임
channel_ids = ['UCMFk5S7g5DY-CZNVh_Kyz_A', 'UCY-mXLM6DsS9cmSwlh0tqSA', 'UC3iSLVH0MxHfwO69oHKpvog', 'UC6ggXTuBVchhwHeQ12Ita1w', 'UCCMFTDGarjgZLc1DlIbbvRg']

# 데이터 수집
all_data = []
for channel_id in channel_ids:
    response = youtubeAnalytics.reports().query(
        ids=f'channel=={channel_id}',
        startDate=start_date,
        endDate=end_date,
        metrics='subscribersGained',
        dimensions='month',
        sort='month'
    ).execute()

    for row in response.get('rows', []):
        data = {
            'channelId': channel_id,
            'month': row[0],
            'subscribersGained': row[1]
        }
        all_data.append(data)

# DataFrame 생성 및 CSV 저장
df = pd.DataFrame(all_data)
filename = f'youtube_subscribers_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
df.to_csv(filename, index=False)
print(f"Data saved to {filename}")