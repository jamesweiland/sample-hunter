import os
from pathlib import Path
import shutil
import uuid
from selenium.webdriver.firefox.options import Options
from tbselenium.tbdriver import TorBrowserDriver
from tbselenium.utils import launch_tbb_tor_with_stem
import tbselenium.common as cm
from typing import Any, Callable
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.remote.webelement import WebElement
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time
from stem import Signal
from stem.control import Controller

from sample_hunter._util import (
    DATA_SAVE_DIR,
    DEFAULT_DOWNLOAD_TIME,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_RETRIES,
    TEMP_TOR_DATA_DIR,
    TOR_BROWSER_DIR,
    TOR_PASSWORD,
)


class DriverContext:
    """Context manager for a driver session."""

    def __init__(
        self,
        headless: bool = False,
        request_timeout: float | None = None,
        download_time: float | None = None,
        print_: bool = True,
        data_save_path: Path | None = None,
        socks_port: int | None = None,
        control_port: int | None = None,
    ) -> None:
        """Store settings for the context."""
        self.headless = headless
        self.request_timeout = request_timeout or DEFAULT_REQUEST_TIMEOUT
        self.download_time = download_time or DEFAULT_DOWNLOAD_TIME
        self.data_save_path = data_save_path or DATA_SAVE_DIR
        self.print_ = print_
        self.socks_port = socks_port
        self.control_port = control_port

    def __enter__(self):
        """Create a driver session."""
        options = Options()
        if self.headless:
            options.add_argument("-headless")

        if self.data_save_path:
            options.set_preference("browser.download.folderList", 2)
            options.set_preference(
                "browser.download.dir", self.data_save_path.__str__()
            )
            options.set_preference("browser.download.useDownloadDir", True)
            options.set_preference("browser.download.showWhenStarting", False)
            options.set_preference(
                "browser.helperApps.neverAsk.saveToDisk", "application/zip"
            )

        # we assume adding an argument for a socks port means a full custom configuration
        if self.socks_port:
            # create unique data directories and ports for each context
            # if contexts share ports then when one is killed they all die
            self.tor_data_dir = Path(TEMP_TOR_DATA_DIR / f"tor_data{uuid.uuid4().hex}")
            self.tor_data_dir.mkdir(exist_ok=True)

            if self.print_:
                print(f"Creating Tor data dir at {str(self.tor_data_dir)}")

            self.launch_tb_with_custom_stem_(options)
        else:
            self.driver = TorBrowserDriver(TOR_BROWSER_DIR.__str__(), options=options)

        self.driver.implicitly_wait(self.request_timeout)
        self.driver.set_window_size(900, 600)
        return self

    def __exit__(self, *exc):
        """Make sure we always quit at end."""
        self.driver.quit()

        if self.print_:
            print(f"Killing process at socks port {self.socks_port}")
        self.tor_process.kill()

        # clean up unique data dir
        if self.tor_data_dir:
            shutil.rmtree(self.tor_data_dir, ignore_errors=True)

    def get_with_retries(
        self, url: str, selector: str, retries: int | None = None
    ) -> str:
        f"""Get a URL and return the source html.

        Wraps with many reties and requires the user to provide a wait css selector.
        Retries the GET request every 5 failed CSS checks, and waits {DEFAULT_REQUEST_TIMEOUT}
        seconds between checks.
        """
        retries = retries or DEFAULT_RETRIES
        if self.print_:
            print(f"Getting {url}...")

        for i in range(retries):
            try:
                if i % 5 == 0:
                    self.driver.get(url)
                    time.sleep(DEFAULT_REQUEST_TIMEOUT)
                self.driver.find_element(By.CSS_SELECTOR, selector)
                break
            except Exception as e:
                if i == (retries - 1):
                    raise
                else:
                    if self.print_:
                        print(f"Excepted on {url}: {str(e)}, retrying...")
                    time.sleep(DEFAULT_REQUEST_TIMEOUT)

        return self.driver.page_source

    def get_current_page(self) -> str:
        """Return the source html of the page the driver is currently on"""
        return self.driver.page_source

    def select_element(self, selector: str) -> WebElement | None:
        """Return the selected WebElement from the webdriver"""
        try:
            return self.driver.find_element(By.CSS_SELECTOR, selector)
        except NoSuchElementException:
            return None

    def renew_ip(self, password: str = TOR_PASSWORD):
        with Controller.from_port(port=str(self.control_port)) as controller:
            controller.authenticate(password=password)
            print(Signal)
            controller.signal(Signal.NEWNYM)  # type: ignore
            time.sleep(10)

    def click_with_retries(
        self,
        url: str,
        selector: str,
        retries: int | None = None,
        wait_condition: Callable | None = None,
    ) -> Any:
        f"""Get a URL and click a button identified by the CSS selector provided by the user.
        
        Has an option to include a wait condition, where the driver will wait until the end
        of the condition or {DEFAULT_DOWNLOAD_TIME} seconds. `wait_condition` must accept driver as the only argument
        """

        retries = retries or DEFAULT_RETRIES
        if self.print_:
            print(f"Getting {url}...")
        for i in range(retries):
            try:
                if i % 5 == 0:
                    self.driver.get(url)
                    time.sleep(DEFAULT_REQUEST_TIMEOUT)
                button = self.driver.find_element(By.CSS_SELECTOR, selector)
                if button:
                    WebDriverWait(self.driver, timeout=DEFAULT_REQUEST_TIMEOUT).until(
                        EC.element_to_be_clickable(button)
                    )
                    if self.print_:
                        print(f"Clicking {button.accessible_name}...")
                    button.click()
                    if wait_condition:
                        wait = WebDriverWait(self.driver, timeout=DEFAULT_DOWNLOAD_TIME)
                        return wait.until(wait_condition)
            except TimeoutException:
                raise
            except Exception as e:
                if i == (retries - 1):
                    raise
                else:
                    if self.print_:
                        print(f"Excepted on {url}: {str(e)}, retrying...")
                    time.sleep(DEFAULT_REQUEST_TIMEOUT)

    def launch_tb_with_custom_stem_(self, options: webdriver.FirefoxOptions):
        tor_binary = os.path.join(TOR_BROWSER_DIR, cm.DEFAULT_TOR_BINARY_PATH)
        print(tor_binary)
        if self.print_:
            print(f"SOCKS port: {self.socks_port}, Control port: {self.control_port}")

        torrc = {
            "ControlPort": str(self.control_port),
            "SocksPort": str(self.socks_port),
            "DataDirectory": self.tor_data_dir.__str__(),
        }

        self.tor_process = launch_tbb_tor_with_stem(
            tbb_path=TOR_BROWSER_DIR.__str__(),
            torrc=torrc,
            tor_binary=tor_binary,
        )
        self.driver = TorBrowserDriver(
            TOR_BROWSER_DIR.__str__(),
            socks_port=self.socks_port,
            control_port=self.control_port,
            tor_cfg=cm.USE_STEM,
            options=options,
        )
