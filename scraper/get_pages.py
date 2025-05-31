import os
import socket
import time
from scraper._util import PARENT_SITEMAP_URL, DriverContext, DATA_SAVE_DIR, THREADS
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from pathlib import Path
import requests
from bs4 import BeautifulSoup, ResultSet
from typing import List
from pydantic import BaseModel, ConfigDict, field_serializer
from bs4 import Tag
import argparse
import shutil
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, get_ident, local
from tqdm import tqdm


class DownloadNotFoundException(Exception):
    pass


class AtomicCounter:
    """Atomic counter for assigning ports to threads"""

    value: int
    end: int | None
    lock: Lock

    def __init__(self, start: int, end: int | None = None):
        self.value = start
        self.end = end
        self.lock = Lock()

    def fetch_and_increment(self, inc: int = 1) -> int:
        with self.lock:
            tmp = self.value
            self.value += 1
            if self.end is not None and self.value > self.end:
                raise ValueError("Can't increment past max value")
            return tmp


class SampleSet(BaseModel):
    name: str
    mega_url: str
    download_path: Path

    @classmethod
    def from_page(
        cls,
        url: str,
        headless: bool,
        data_save_path: Path,
        thread_local: local | None = None,
    ) -> "SampleSet":
        soup = BeautifulSoup(requests.get(url).text, "html.parser")
        mega_url = cls.get_mega_url(soup)
        assert mega_url != ""

        name = cls.get_name(soup)
        download_path = cls.download(
            mega_url,
            headless,
            data_save_path,
            thread_local.socks_port if thread_local else None,
            thread_local.control_port if thread_local else None,
        )

        return cls(name=name, mega_url=mega_url, download_path=download_path)

    @staticmethod
    def download(
        mega_url: str,
        headless: bool,
        data_save_path: Path,
        socks_port: int | None = None,
        control_port: int | None = None,
    ) -> Path:
        """Download a zip file from a mega URL using a Selenium driver"""

        selector = (
            "button.mega-button.positive.js-default-download.js-standard-download"
        )

        previous_downloads = set(os.listdir(data_save_path))

        def download_finished(driver) -> bool | Path:
            """Wait condition for clicking the button. If the download is finished, returns the path to the new file, otherwise, returns False"""
            filename = driver.find_element(By.CSS_SELECTOR, "span.filename").text
            if filename is None:
                return False

            new_downloads = set(os.listdir(data_save_path)).difference(
                previous_downloads
            )
            if len(new_downloads) > 0:
                for new_download in new_downloads:
                    path = Path(data_save_path / new_download)
                    # check for temp downloads by checking suffix
                    if path.stem == filename:
                        # if we find it, wait 5 seconds so the browser doesn't yell at us
                        time.sleep(5)
                        return path
            return False

        with DriverContext(
            headless=headless,
            data_save_path=data_save_path,
            socks_port=socks_port,
            control_port=control_port,
        ) as driver:
            try:
                if is_paywalled(driver):
                    driver.renew_ip()
                path = driver.click_with_retries(
                    mega_url, selector, wait_condition=download_finished
                )
                assert path is not None
                return path
            except TimeoutException:
                print(f"timed out trying to download {mega_url}")
                return Path()

    @staticmethod
    def get_mega_url(soup: BeautifulSoup, max_retries=5) -> str:
        # first, check if the download link is on the page
        a_tags: ResultSet[Tag] = soup.select("div.post-body a[href]")
        href_queue: List[str] = []
        for a in a_tags:
            if a.get("href", None) is not None:
                href_queue.append(str(a["href"]))

            retries = 0
            while href_queue and retries < max_retries:
                href = href_queue.pop(0)

                # first, check if the href is the download link
                if href is not None and href.startswith("https://mega"):  # type: ignore
                    return href  # type: ignore

                # then, check if you have to click the cover
                if href is not None and (
                    href.startswith("http://www.hiphopisread.com")
                    or href.startswith("http://samplesets.blogspot.com")
                ):
                    redirected_soup = BeautifulSoup(
                        requests.get(href).text, "html.parser"
                    )
                    redirected_a_tags: ResultSet[Tag] = redirected_soup.select(
                        "div.post-body a[href]"
                    )
                    for redirected_a in redirected_a_tags:
                        if redirected_a.get("href", None) is not None:
                            href_queue.append(str(redirected_a["href"]))
                            retries += 1

        raise DownloadNotFoundException("No mega.nz link found in entry-content div")

    @staticmethod
    def get_name(soup: BeautifulSoup) -> str:
        """Get the name of the sample set"""
        name_tag = soup.select_one("h3.post-title.entry-title a")
        if name_tag:
            return name_tag.get_text(strip=True)
        raise ValueError("Title not found")

    @field_serializer("download_path")
    def serialize_path(self, p: Path, _info):
        return p.__str__()


def get_sitemap_urls(url: str) -> List[str]:
    """Make a get request to `url` and parse out the sitemaps."""
    response = requests.get(url).text
    xml = BeautifulSoup(response, "xml")
    urls = [loc.text for loc in xml.find_all("loc")]
    return urls


def get_sample_set_urls(sitemap_url: str) -> List[str]:
    """Navigate to the sitemap and get all sample set links on that sitemap"""
    response = requests.get(sitemap_url).text
    xml = BeautifulSoup(response, "xml")
    sampleset_urls = []
    for loc in xml.find_all("loc"):
        if "sample-set" in loc.text:
            sampleset_urls.append(loc.text)
    return sampleset_urls


def is_paywalled(driver: DriverContext) -> bool:
    """Returns True if the mega.nz download is behind a paywall"""
    # check for both hard and soft paywall
    limited_quota = driver.select_element("div.dialog.header-before-icon.limited")
    exceeded_quota = driver.select_element("div.dialog.header-before-icon.exceeded")

    return (limited_quota is not None and limited_quota.is_displayed()) or (
        exceeded_quota is not None and exceeded_quota.is_displayed()
    )


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def parse_args() -> argparse.Namespace:
    """Set up the command line parser"""
    parser = argparse.ArgumentParser()

    parser.add_argument("--single", type=str, help="Option to only process one URL")
    parser.add_argument(
        "--data-save-path",
        type=Path,
        help="the directory to save downloaded data to",
        default=DATA_SAVE_DIR,
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Option to run the script using a headless browser",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_save_path = args.data_save_path.absolute()

    # clean data save path and make dirs if it doesnt exist
    shutil.rmtree(data_save_path)
    data_save_path.mkdir()

    prefs = {
        "download.default_directory": data_save_path.__str__(),
        "download.prompt_for_download": False,
        "directory_upgrade": True,
    }

    if args.single is not None:
        sampleset = SampleSet.from_page(args.single, args.headless, data_save_path)
        save_path = Path(args.data_save_path / "summary.json")
        with open(save_path, "w") as f:
            json.dump(sampleset.model_dump(), f, indent=4)
        exit(0)

    sitemaps = get_sitemap_urls(PARENT_SITEMAP_URL)

    sampleset_urls = []
    for sitemap in sitemaps:
        sampleset_urls.extend(get_sample_set_urls(sitemap))

    # set up multithreading
    port_counter = AtomicCounter(9_050, 10_000)
    results = []
    thread_local = local()

    def worker(url, headless, data_save_path):
        """Worker function for multithreading"""
        if not hasattr(thread_local, "socks_port"):
            port = port_counter.fetch_and_increment()
            while is_port_in_use(port):
                port = port_counter.fetch_and_increment()
            thread_local.socks_port = port
            print(f"Thread {get_ident()} was assigned SOCKS port {port}")

        if not hasattr(thread_local, "control_port"):
            port = port_counter.fetch_and_increment()
            while is_port_in_use(port):
                port = port_counter.fetch_and_increment()
            thread_local.control_port = port
            print(f"Thread {get_ident()} was assigned control port {port}")

        print(f"Processing {url}")
        sampleset = SampleSet.from_page(
            url, headless, data_save_path, thread_local=thread_local
        )
        return sampleset

    with tqdm(total=len(sampleset_urls), desc="Processing sample sets") as pbar:
        with ThreadPoolExecutor(max_workers=THREADS) as executor:
            future_to_url = {
                executor.submit(worker, url, args.headless, data_save_path): url
                for url in sampleset_urls
            }
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    results.append(future.result())
                    pbar.update(1)
                except DownloadNotFoundException:
                    print(f"{url} has a dead link")
                except Exception:
                    raise

    print(f"parsed {len(results)} sample sets")

    save_path = Path(data_save_path / "summary.json")
    with open(save_path, "w") as f:
        json.dump([s.model_dump() for s in results], f, indent=4)

    """
    for each sitemap:
        
        if loc contains 'sample-set':
            go to that page and get the meganz download link
            download the zip file to data_/{name}
            append samplesets with the relevant info
    """
