from concurrent.futures import ThreadPoolExecutor
from threading import get_ident, local
import time
from sample_hunter._util import THREADS, TOR_BROWSER_DIR
from sample_hunter.scraper.driver_context import DriverContext
from scraper.get_pages import AtomicCounter, is_port_in_use
from tbselenium.tbdriver import TorBrowserDriver
import os


def test_ip_address_switching():
    port_counter = AtomicCounter(9_050, 10_000)
    results = []
    thread_local = local()

    def worker(_=None):
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
        # get a ip test url with drivercontext
        with DriverContext(
            headless=False,
            socks_port=thread_local.socks_port,
            control_port=thread_local.control_port,
        ) as driver:
            driver.get_with_retries("https://whatismyipaddress.com/", "body")
            time.sleep(100)

    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        try:
            addresses = list(executor.map(worker, [None] * THREADS))
        except Exception:
            raise

    print(addresses)


def test_normal_tor_browser():
    with TorBrowserDriver(TOR_BROWSER_DIR.__str__()) as driver:
        driver.get("https://whatismyipaddress.com/")
        time.sleep(100)


if __name__ == "__main__":
    # test that IPs are different for different threads
    test_ip_address_switching()
    # test a normal tor browser setup
    # test_normal_tor_browser()
