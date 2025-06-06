from pathlib import Path
import os
import sys
from threading import Lock
import multiprocessing

import pandas as pd


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
PROCS: int = multiprocessing.cpu_count()


class SigintHandler:
    """Handler to save a dataframe before exiting a long-running script that needs to be interrupted"""

    def __init__(self, df_name: str, path: Path):
        """Store information about the df.

        Args:
            df_name (str): the name of the object that the df is assigned to in the script
            path (Path): the path to save the df to
        """

        self.df_name = df_name
        self.path = path
        self.lock = multiprocessing.Lock()  # to make it safe for parallelization

    def __call__(self, sig, frame):
        """On sigint, save the df"""

        curr = frame
        df = None
        found = False
        while curr:
            for name, val in curr.f_locals.items():
                if name == self.df_name:
                    df = val
                    found = True
                    break
            if found:
                break
            curr = curr.f_back
        if df is not None and self.path is not None:
            print(f"Saving data to {self.path} before exiting...")
            if self.path == ".csv":
                with self.lock:
                    old_df = pd.read_csv(self.path)
                    df.to_csv(self.path)
            else:
                df.to_json(self.path, orient="index", indent=4)
            sys.exit(0)
        print("df not found and no data was saved")
        sys.exit(0)


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
