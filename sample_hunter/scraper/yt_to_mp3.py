import time
import requestium
from selenium.webdriver import Chrome, ChromeOptions
from urllib.parse import urlparse, parse_qs
import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import sys
# === Config ===
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
MP3_DIR = DATA_DIR / "mp3"
MP3_DIR.mkdir(parents=True, exist_ok=True)
INPUT_CSV = DATA_DIR / "query_list_with_urls.csv"
METADATA_CSV = DATA_DIR / "mp3_metadata.csv"

CNVMP3_DOMAIN = "https://cnvmp3.com/"
RATE_LIMIT_TIMEOUT: float = 60.0

class CnvMP3Exception(Exception):
    pass

class RateLimitException(CnvMP3Exception):
    pass

class CnvMP3Client:
    def __init__(self, headless: bool = True, timeout: float = RATE_LIMIT_TIMEOUT):
        self.headless = headless
        self.timeout = timeout

    def __enter__(self):
        options = ChromeOptions()
        if self.headless:
            options.add_argument("headless")
        self._driver = Chrome(options=options)
        self.session = requestium.Session(driver=self._driver)
        self.session.driver.get(CNVMP3_DOMAIN)
        self.session.transfer_driver_cookies_to_session()
        return self

    def __exit__(self, *exc):
        if self._driver:
            self._driver.quit()
        if self.session:
            self.session.close()

    def yt_to_mp3(self, yt_url: str) -> bytes | None:
        try:
            yt_id = parse_qs(urlparse(yt_url).query)["v"][0]
            # Step 1: Check database
            check_db_url = "https://cnvmp3.com/check_database.php"
            r = self.session.post(check_db_url, json={
                "youtube_id": yt_id, "quality": 4, "formatValue": 1
            })
            if r.status_code == 200 and r.json().get("success"):
                server_path = r.json()["data"]["server_path"]
                download = self.session.get(server_path)
                if download.status_code == 200:
                    return download.content
            # Step 2: Get video data
            video_data_url = "https://cnvmp3.com/get_video_data.php"
            r = self.session.post(video_data_url, json={"token": "1234", "url": yt_url})
            if r.status_code == 200 and r.json().get("success"):
                video_data = r.json()
                title = video_data.get("title", "yt_track")
                # Step 3: Download video ucep
                download_ucep_url = "https://cnvmp3.com/download_video_ucep.php"
                r = self.session.post(download_ucep_url, json={
                    "formatValue": 1, "quality": 4, "title": title, "url": yt_url
                })
                if r.status_code == 200 and r.json().get("success"):
                    download_link = r.json()["download_link"]
                    download = self.session.get(download_link)
                    if download.status_code == 200:
                        # Step 4: Insert to database (optional)
                        self.session.post("https://cnvmp3.com/insert_to_database.php", json={
                            "formatValue": 1, "quality": 4, "server_path": download_link,
                            "title": title, "youtube_id": yt_id
                        })
                        return download.content
        except Exception as e:
            print(f"Error in yt_to_mp3 for {yt_url}: {e}")
        return None


def write_to_disk(bytes_: bytes, path: Path) -> None:
    with open(path, "wb") as file:
        file.write(bytes_)

def main():
    df = pd.read_csv(INPUT_CSV)
    records = []
    with CnvMP3Client(headless= False) as client:
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            yt_url = row.get("yt_url", "")
            if not yt_url or pd.isna(yt_url):
                print(f"Skipping row {idx}: no yt_url")
                continue
            id_str = f"{idx+1:05d}"
            out_file = MP3_DIR / f"{id_str}.mp3"
            print(f"Downloading {yt_url} as {out_file.name}")
            try:
                mp3 = client.yt_to_mp3(yt_url)
            except Exception as e:
                print(f"Error downloading {yt_url}: {e}")
                continue
            if mp3:
                write_to_disk(mp3, out_file)
                print(f"Saved: {out_file}")
                records.append({
                    "id": id_str,
                    "artist": row.get("artist", ""),
                    "title": row.get("title", ""),
                    "album": row.get("album", ""),
                    "year": row.get("year", ""),
                    "videoId": row.get("videoId", ""),
                    "yt_url": yt_url,
                    "filename": out_file.name
                })
            else:
                print(f"Failed to download {yt_url}")
            time.sleep(1)
    meta_df = pd.DataFrame(records)
    meta_df.to_csv(METADATA_CSV, index=False)
    print(f"Metadata saved to {METADATA_CSV}")

if __name__ == "__main__":
    main()
