import time
import requestium
from requestium.requestium import RequestiumResponse
from selenium.webdriver import Chrome, ChromeOptions
from urllib.parse import urlparse, parse_qs
import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from sample_hunter._util import DATA_SAVE_DIR, DEFAULT_RETRIES, DEFAULT_RETRY_DELAY
import sys
import signal
from retry import retry

CNVMP3_DOMAIN = "https://cnvmp3.com/"
RATE_LIMIT_TIMEOUT: float = 60.0


class CnvMP3Exception(Exception):
    def __init__(self, r: RequestiumResponse, *args):
        self.response = r
        super().__init__(*args)


class VideoLimitException(CnvMP3Exception):
    pass


class RateLimitException(CnvMP3Exception):
    pass


class CnvMP3Client:
    """Class and context manager to interact with cnvmp3.com"""

    def __init__(self, headless: bool, timeout: float | None = None):
        """Store settings for requestium session"""
        self.headless = headless
        self.timeout = timeout or RATE_LIMIT_TIMEOUT

    def __enter__(self):
        options = ChromeOptions()
        if self.headless:
            options.add_argument("headless")
        self._driver = Chrome(options=options)
        self.session = requestium.Session(driver=self._driver)
        self.session.driver.get("https://www.cnvmp3.com/")
        self.session.transfer_driver_cookies_to_session()

        return self

    def __exit__(self, *exc):
        if self._driver:
            self._driver.quit()
        if self.session:
            self.session.close()

    @retry(tries=DEFAULT_RETRIES, delay=DEFAULT_RETRY_DELAY)
    def yt_to_mp3(self, yt_url: str) -> bytes | None:
        """The main function for the client. Convert a YouTube link to mp3 format and download it to `download_path`

        :return: the binary data representing the mp3 file
        """
        try:
            yt_id = parse_qs(urlparse(yt_url).query)["v"][0]
            r = self.check_database(yt_id)
            print(f"check database {r.status_code}")
            if r.json()["success"]:
                return self.request_download(r.json()["data"]["server_path"]).content
            else:
                print("Check db was not successful")
                video_data_response = self.get_video_data(yt_url)
                print(f"get video data {video_data_response.status_code}")
                video_data = video_data_response.json()
                if video_data["success"]:
                    # get title so we can insert to db later
                    title = video_data["title"]
                    download_video_ucep_response = self.download_video_ucep(
                        title=video_data["title"], yt_url=yt_url
                    )
                    print(f"download video ucep {r.status_code}")

                    r = download_video_ucep_response.json()
                    if r["success"]:
                        print("Download video ucep was successful")
                        download = self.request_download(r["download_link"]).content
                        self.insert_to_database(yt_id, title, r["download_link"])
                        return download
                    elif r["errorType"] == 3:
                        # youtube errors
                        raise CnvMP3Exception(
                            download_video_ucep_response,
                            f"YouTube had an error downloading {yt_url}",
                        )
        except RateLimitException:
            print(f"Rate limited. Sleeping {self.timeout} seconds then retrying...")
            time.sleep(self.timeout)
            raise
        except CnvMP3Exception as e:
            print(f"Error occurred for {yt_url}:")
            print(e.__class__, str(e))
            print(f"Response code: {e.response.status_code}")
            print(f"Response text: {e.response.text}")
            print(f"Request: {e.response.request}")
            return None

        raise RuntimeError("Something bad happened with requesting CnvMP3")

    def get_video_data(self, yt_url: str) -> RequestiumResponse:
        endpoint = "https://cnvmp3.com/get_video_data.php"
        body = {"token": "1234", "url": yt_url}
        r = self.session.post(endpoint, json=body)
        if r.status_code == 200:
            return r
        elif r.status_code == 429:
            raise RateLimitException(r)
        raise CnvMP3Exception(r, f"Getting video data failed for {yt_url}")

    def request_download(self, url: str) -> RequestiumResponse:
        headers = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "cross-site",
            "Sec-Fetch-User": "?1",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
            # "Host": "apio8dlp.cnvmp3.online",
            "Pragma": "no-cache",
            "Referer": "https://cnvmp3.com/",
            "Upgrade-Insecure-Requests": "1",
            "sec-ch-ua": '"Chromium";v="136", "Google Chrome";v="136", "Not.A/Brand";v="99"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Linux"',
            "Accept-Encoding": "gzip, deflate, br, zstd",
        }
        r = self.session.get(url, headers=headers)
        if r.status_code == 200:
            return r
        elif r.status_code == 429:
            raise RateLimitException(r)
        else:
            raise CnvMP3Exception(r, f"Requesting download failed for {url}")

    def download_video_ucep(self, title: str, yt_url: str) -> RequestiumResponse:
        url = "https://cnvmp3.com/download_video_ucep.php"
        body = {"formatValue": 1, "quality": 4, "title": title, "url": yt_url}
        headers = {
            "Content-Type": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
            "Referer": "https://cnvmp3.com/v25",
            "sec-ch-ua": '"Chromium";v="136", "Google Chrome";v="136", "Not.A/Brand";v="99"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Linux"',
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
        }
        r = self.session.post(url, json=body, headers=headers)
        if r.status_code != 200:
            if r.status_code == 429:
                raise RateLimitException(r)
            else:
                raise CnvMP3Exception(r, f"Failed to download video ucep for {yt_url}")
        if not r.json()["success"] and r.json()["errorType"] == 4:
            raise VideoLimitException(r, f"{yt_url} exceeds video limit")
        return r

    def check_database(self, yt_id: str) -> RequestiumResponse:
        url = "https://cnvmp3.com/check_database.php"
        body = {"youtube_id": yt_id, "quality": 4, "formatValue": 1}
        r = self.session.post(url, json=body)
        if r.status_code == 200:
            return r
        elif r.status_code == 429:
            raise RateLimitException(r)
        else:
            raise CnvMP3Exception(r, f"Checking database failed for {yt_id}")

    def insert_to_database(self, yt_id: str, title: str, server_path: str):
        url = "https://cnvmp3.com/insert_to_database.php"
        body = {
            "formatValue": 1,
            "quality": 4,
            "server_path": server_path,
            "title": title,
            "youtube_id": yt_id,
        }
        r = self.session.post(url, json=body)
        if r.status_code != 200:
            if r.status_code == 429:
                raise RateLimitException(r)
            raise CnvMP3Exception(r, f"Failed to insert db for {yt_id}")


def write_to_disk(bytes: bytes, path: Path) -> None:
    with open(path, "wb") as file:
        file.write(bytes)


def sigint_handler(sig, frame):
    """Save data on KeyboardInterrupt since this is a long script"""
    curr = frame
    df = None
    args = None
    while curr:
        for name, val in curr.f_locals.items():
            if name == "sampling_songs":
                df = val
            elif name == "args":
                args = val
        curr = curr.f_back
    if df is not None and args is not None:
        print(f"Saving data to {args.in_} before exiting...")
        if args.in_ == ".csv":
            df.to_csv(args.in_)
        else:
            df.to_json(args.in_, orient="index", indent=4)
    sys.exit(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--in",
        help="The path to the JSON or CSV file to get mp3 files for",
        dest="in_",
        type=Path,
        default=Path(DATA_SAVE_DIR / "sampling_songs.csv"),
    )

    parser.add_argument(
        "--headless",
        help="Open the selenium driver as a headless instance",
        action="store_true",
    )

    parser.add_argument(
        "--single",
        help="Option to pass a single URL to process, for debugging",
        type=str,
    )

    parser.add_argument(
        "--append",
        help="Option to not process rows that already have a path",
        action="store_true",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    signal.signal(signal.SIGINT, sigint_handler)

    if args.single:
        single_path = Path("/home/james/Downloads/test.mp3")
        with CnvMP3Client(headless=args.headless) as client:
            mp3 = client.yt_to_mp3(args.single)
            if mp3:
                write_to_disk(mp3, single_path)
        sys.exit(0)

    assert args.in_.suffix == ".json" or args.in_.suffix == ".csv"
    if args.in_.suffix == ".csv":
        sampling_songs = pd.read_csv(args.in_)
    else:
        sampling_songs = pd.read_json(args.in_, orient="index")

    if not args.append:
        sampling_songs["sampling_path"] = None
        to_process = sampling_songs
    else:
        to_process = sampling_songs[sampling_songs["sampling_path"].isnull()]

    with CnvMP3Client(headless=args.headless) as client:
        # do smth
        for idx, row in tqdm(list(to_process.iterrows())):
            row = row.astype("str")
            yt_url = row["yt_url"]
            if yt_url:
                print(f"Processing {yt_url}...")
                download_path = Path(row["path"]).parent / (row["title"] + ".mp3")
                try:
                    mp3 = client.yt_to_mp3(yt_url)
                except Exception as e:
                    print(
                        "An unknown error occurred. Saving progress before exiting..."
                    )
                    if args.in_.suffix == ".csv":
                        sampling_songs.to_csv(args.in_)
                    else:
                        sampling_songs.to_json(args.in_, orient="index", indent=4)
                    print(
                        f"Progress saved to {args.in_}. Progress can be resumed"
                        " without overwriting with the --append argument."
                    )

                    raise e

                if mp3:
                    write_to_disk(mp3, download_path)
                    sampling_songs.at[idx, "sampling_path"] = download_path
                    print(f"Downloaded {yt_url} to {download_path}")

    print(f"Saving table to {args.in_}...")
    if args.in_.suffix == ".csv":
        sampling_songs.to_csv(args.in_)
    else:
        sampling_songs.to_json(args.in_, orient="index", indent=4)

    print("All good!")
