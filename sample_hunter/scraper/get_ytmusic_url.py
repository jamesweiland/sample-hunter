import pandas as pd
from ytmusicapi import YTMusic
import time

def main():
    df = pd.read_csv('query_list.csv')
    ytmusic = YTMusic()
    
    video_ids = []
    yt_urls = []
    for idx, row in df.iterrows():
        query = f"{row['artist']} {row['title']}"
        print(f"Searching: {query}")
        try:
            results = ytmusic.search(query, filter="songs")
            if results:
                video_id = results[0]['videoId']
                yt_url = f"https://www.youtube.com/watch?v={video_id}"
            else:
                video_id = ""
                yt_url = ""
        except Exception as e:
            print(f"Error searching {query}: {e}")
            video_id = ""
            yt_url = ""
        video_ids.append(video_id)
        yt_urls.append(yt_url)
        time.sleep(0.25)  # To be gentle with the API

    df['videoId'] = video_ids
    df['yt_url'] = yt_urls
    df.to_csv('query_list_with_urls.csv', index=False)
    print("Saved with YouTube URLs!")

if __name__ == "__main__":
    main()
