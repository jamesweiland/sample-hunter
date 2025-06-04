from pathlib import Path
import os
from threading import Lock


TOR_BROWSER_DIR: Path = Path("/home/james/tor-browser-linux-x86_64-14.5.3/tor-browser/")
TEMP_TOR_DATA_DIR: Path = Path("/home/james/code/sample-hunter/temp_tor_data/")
TOR_PASSWORD: str = os.environ["TOR_PASSWORD"]


PARENT_SITEMAP_URL: str = "https://www.hiphopisread.com/sitemap.xml"
SITEMAP_SAVE_PATH: Path = Path("_data/sitemaps/")
DATA_SAVE_DIR: Path = Path("_data/")
ZIP_ARCHIVE_DIR: Path = Path("_data/archive/")

DEFAULT_REQUEST_TIMEOUT: float = 15.0
DEFAULT_DOWNLOAD_TIME: float = 2700.0
DEFAULT_RETRIES: int = 5
DEFAULT_RETRY_DELAY: float = 5.0
THREADS: int = 1


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
