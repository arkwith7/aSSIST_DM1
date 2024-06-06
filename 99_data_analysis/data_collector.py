import os
import pandas as pd
import asyncio
import aiohttp
import json
from datetime import datetime, timedelta
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# .env 파일에서 YOUTUBE_API_KEY 가져오기
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('YOUTUBE_API_KEY')

# 캐시 파일 경로
cache_path = 'youtube_cache.json'

# 캐시 로드
def load_cache():
    try:
        with open(cache_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return {}

# 캐시 저장
def save_cache(cache):
    with open(cache_path, 'w') as file:
        json.dump(cache, file)

# 비동기 API 호출
async def async_api_call(url, params):
    print(f"Making API call to URL: {url} with params: {params}")
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, params=params) as response:
                if response.status == 403:
                    error_message = await response.json()
                    print(f"Quota exceeded or forbidden access: {error_message}")
                    return None
                elif response.status != 200:
                    error_details = await response.text()
                    print(f"API call failed with status {response.status}: {error_details}")
                    return None
                return await response.json()
        except Exception as e:
            print(f"An error occurred during API call: {e}")
            return None

# 채널 통계 가져오기
async def get_channel_stats(channel_id):
    cache = load_cache()
    if channel_id in cache:
        return cache[channel_id]

    params = {
        'part': 'statistics,snippet',
        'id': channel_id,
        'key': api_key
    }
    url = "https://www.googleapis.com/youtube/v3/channels"
    response = await async_api_call(url, params)
    print(f"Response for channel stats: {response}")  # 응답 데이터 출력
    if not response or 'items' not in response or not response['items']:
        print(f"No data found for channel {channel_id}")
        return {}

    item = response['items'][0]
    stats = item['statistics']
    snippet = item['snippet']
    channel_info = {
        'channelId': channel_id,
        'title': snippet['title'],
        'publishedAt': snippet['publishedAt'],
        'viewCount': stats.get('viewCount', 0),
        'commentCount': stats.get('commentCount', 0),
        'subscriberCount': stats.get('subscriberCount', 0),
        'hiddenSubscriberCount': stats.get('hiddenSubscriberCount', False),
        'videoCount': stats.get('videoCount', 0)
    }
    cache[channel_id] = channel_info
    save_cache(cache)
    return channel_info

# 비디오 상세 정보 가져오기 (기간 필터링 추가)
async def get_video_details(channel_id, start_date, end_date):
    cache = load_cache()
    cache_key = f"{channel_id}_{start_date}_{end_date}_videos"
    if cache_key in cache:
        return cache[cache_key]

    video_details = []
    params = {
        'channelId': channel_id,
        'part': 'id',
        'order': 'date',
        'type': 'video',
        'maxResults': 50,
        'publishedAfter': start_date,
        'publishedBefore': end_date,
        'key': api_key
    }
    url = "https://www.googleapis.com/youtube/v3/search"

    while True:
        response = await async_api_call(url, params)
        print("채널의 비디오 정보 가져오는 함수 실행")
        print(f"Response for video details: {response}")  # 응답 데이터 출력
        if response is None:
            print(f"Failed to fetch video details for channel {channel_id}. Check API key and network.")
            break

        if 'items' not in response or not response['items']:
            print(f"No video items found in response for channel {channel_id}. Check date range and channel ID.")
            break

        for item in response['items']:
            if 'id' in item and 'videoId' in item['id']:
                video_id = item['id']['videoId']
                video_url = "https://www.googleapis.com/youtube/v3/videos"
                video_params = {
                    'part': 'snippet,statistics,contentDetails',  # 'contentDetails' 추가
                    'id': video_id,
                    'key': api_key
                }
                video_response = await async_api_call(video_url, video_params)
                if video_response and 'items' in video_response and video_response['items']:
                    video_item = video_response['items'][0]
                    video_details.append({
                        'videoId': video_id,
                        'title': video_item['snippet']['title'],
                        'videoAuthorId': video_item['snippet']['channelId'],  # 비디오 작성자의 ID 추가
                        'channelTitle': video_item['snippet']['channelTitle'],  # 비디오 작성자(채널 이름) 추가         
                        'publishedAt': video_item['snippet']['publishedAt'],
                        'viewCount': video_item['statistics'].get('viewCount', 0),
                        'likeCount': video_item['statistics'].get('likeCount', 0),
                        'dislikeCount': video_item['statistics'].get('dislikeCount', 0),
                        'duration': video_item['contentDetails']['duration']  # 재생 시간 추가
                    })
                else:
                    print(f"No details found for video ID {video_id}.")
            else:
                print(f"No videoId found in item: {item}")

        if 'nextPageToken' in response:
            params['pageToken'] = response['nextPageToken']
        else:
            break

    cache[cache_key] = video_details  # 캐시 업데이트
    save_cache(cache)
    return video_details

# 비디오 댓글 가져오기
async def get_video_comments(video_id):
    cache = load_cache()  # 캐시 로드
    cache_key = f"comments_{video_id}"  # 캐시 키에 'comments_' 접두사 추가
    if cache_key in cache:
        return cache[cache_key]  # 캐시된 댓글 반환

    comments = []
    params = {
        'part': 'snippet,replies',  # 답글 정보도 포함
        'videoId': video_id,
        'maxResults': 100,
        'textFormat': 'plainText',
        'key': api_key
    }
    url = "https://www.googleapis.com/youtube/v3/commentThreads"

    while True:
        try:
            response = await async_api_call(url, params)
            if 'items' not in response:
                print(f"No items found in response for video {video_id}.")
                break

            for item in response['items']:
                top_comment = item['snippet']['topLevelComment']['snippet']
                top_comment_id = item['snippet']['topLevelComment']['id']
                comment_data = {
                    'videoId': video_id,
                    'commentId': top_comment_id,
                    'authorDisplayName': top_comment['authorDisplayName'],
                    'authorId': top_comment['authorChannelId']['value'],  # 작성자 ID로 변경
                    'textOriginal': top_comment['textOriginal'],
                    'likeCount': top_comment['likeCount'],
                    'publishedAt': top_comment['publishedAt'],
                    'parentCommentId': None  # 최상위 댓글은 부모 댓글 ID가 없음
                }
                comments.append(comment_data)
                # 답글 정보 추가
                if 'replies' in item:
                    for reply in item['replies']['comments']:
                        reply_snippet = reply['snippet']
                        comments.append({
                            'videoId': video_id,
                            'commentId': reply['id'],
                            'authorDisplayName': reply_snippet['authorDisplayName'],
                            'authorId': reply_snippet['authorChannelId']['value'],  # 답글 작성자 ID로 변경
                            'textOriginal': reply_snippet['textOriginal'],
                            'likeCount': reply_snippet['likeCount'],
                            'publishedAt': reply_snippet['publishedAt'],
                            'parentCommentId': top_comment_id  # 답글의 경우, 최상위 댓글 ID를 부모 ID로 설정
                        })

            if 'nextPageToken' in response:
                params['pageToken'] = response['nextPageToken']
            else:
                break
        except Exception as e:
            if 'commentsDisabled' in str(e):
                print(f"댓글이 비활성화된 비디오입니다: {video_id}")
                break  # 댓글 비활성화 오류 처리
            else:
                print(f"API 호출 중 예외 발생: {e}")
                break  # 다른 예외 발생 시 처리

    cache[cache_key] = comments  # 캐시 업데이트
    save_cache(cache)  # 캐시 저장
    return comments

# 메인 함수
async def main(channel_ids, months):
    if not channel_ids:
        print("No channel IDs provided.")
        return

    end_date = datetime.now()
    start_date = end_date - timedelta(days=30 * months)
    start_date_str = start_date.strftime('%Y-%m-%dT%H:%M:%SZ')
    end_date_str = end_date.strftime('%Y-%m-%dT%H:%M:%SZ')
    print(f"Start date: {start_date_str}, End date: {end_date_str}")

    all_data = []

    for channel_id in channel_ids:
        print(f"Fetching data for channel: {channel_id}")
        channel_stats = await get_channel_stats(channel_id)
        print(f"Channel stats for channel {channel_id}: {channel_stats}")

        if not channel_stats or 'channelId' not in channel_stats:
            print(f"No channel stats found for channel: {channel_id}")
            continue  # 채널 통계 정보가 없거나 channelId가 없으면 다음 채널로 넘어갑니다.

        video_stats = await get_video_details(channel_stats['channelId'], start_date_str, end_date_str)
        print("Video details fetching completed")
        print(f"Video stats for channel {channel_id}: {video_stats}")
        if not video_stats:
            print(f"No video stats found for channel: {channel_id}")
            continue  # 비디오 통계 정보가 없으면 다음 채널로 넘어갑니다.

        for video_stat in video_stats:
            if 'videoId' not in video_stat:
                print(f"No videoId found in video stats: {video_stat}")
                continue  # 비디오 ID가 없으면 다음 비디오로 넘어갑니다.

            comments = await get_video_comments(video_stat['videoId'])
            print(f"Comments fetched for video {video_stat['videoId']}")
            if not comments:
                print(f"No comments found for video: {video_stat['videoId']}")
                continue  # 댓글 정보가 없으면 다음 비디오로 넘어갑니다.

            for comment in comments:
                if 'commentId' not in comment:
                    print(f"No commentId found in comment: {comment}")
                    continue  # 댓글 ID가 없으면 다음 댓글로 넘어갑니다.

                all_data.append({
                    'channelId': channel_id,
                    'channelTitle': channel_stats.get('title', 'N/A'),
                    'channelPublishedAt': channel_stats.get('publishedAt', 'N/A'),
                    'subscriberCount': channel_stats.get('subscriberCount', 'N/A'),
                    'videoId': video_stat['videoId'],
                    'videoTitle': video_stat['title'],
                    'videoAuthorId': video_stat['videoAuthorId'],  # 비디오 작성자 ID 추가
                    'videoPublishedAt': video_stat['publishedAt'],
                    'duration': video_stat['duration'],  # 재생 시간 추가 
                    'viewCount': video_stat['viewCount'],
                    'likeCount': video_stat['likeCount'],
                    'dislikeCount': video_stat['dislikeCount'],
                    'commentId': comment['commentId'],
                    'commentAuthor': comment['authorDisplayName'],
                    'authorId': comment['authorId'],  # 댓글 작성자 ID 추가
                    'commentText': comment['textOriginal'],
                    'commentLikeCount': comment['likeCount'],
                    'commentPublishedAt': comment['publishedAt'],
                    'parentCommentId': comment.get('parentCommentId')  # 부모 댓글 ID 추가
                })

    if not all_data:
        print("No data collected.")
        return

    # 결과를 DataFrame으로 변환하고 CSV로 저장
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'youtube_channel_comments_data_{current_time}.csv'
    df = pd.DataFrame(all_data)
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

# 비동기 메인 함수 실행
if __name__ == "__main__":
    channel_ids = ['UCMFk5S7g5DY-CZNVh_Kyz_A', 'UCY-mXLM6DsS9cmSwlh0tqSA', 'UC3iSLVH0MxHfwO69oHKpvog', 'UC6ggXTuBVchhwHeQ12Ita1w', 'UCCMFTDGarjgZLc1DlIbbvRg']  # 채널 ID 예시
    months = 6  # 2개월치 데이터를 가져옵니다.
    asyncio.run(main(channel_ids, months))
