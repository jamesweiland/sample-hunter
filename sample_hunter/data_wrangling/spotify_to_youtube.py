# type: ignore

import urllib.parse
import requestium
from selenium.webdriver import Chrome, ChromeOptions
from selenium.webdriver.support.ui import WebDriverWait
import os
from typing import List
import socketio
import requests
from sample_hunter._util import DEFAULT_REQUEST_TIMEOUT
import time

DEFAULT_AUTH_WAIT: int = 300
USER_AGENT: str = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36"
)
YOUTUBE_ACCOUNT_ID = os.environ["YOUTUBE_ACCOUNT_ID"]


def load_library(session: requestium.Session):
    endpoint = "https://api.tunemymusic.com/v2/Transfer/LoadLibrary"
    body = {
        "source": "Spotify",
        "account": "jamesweiland04",
        "darkMode": True,
        "fromFile": False,
        "fromUrl": False,
        "sm": "no",
    }
    response = session.post(endpoint, json=body)
    return response
    # print(response.status_code)
    # print(response.text)


def get_user_data(session: requestium.Session):
    endpoint = "https://api.tunemymusic.com/v2/General/GetUserData"
    body = {
        "referrer": "",
        "lang": "en",
        "page": "/transfer",
    }
    response = session.post(endpoint, json=body)
    print(response.status_code)
    print(response.text)
    print(response.json()["user"]["connections"]["Spotify"])
    print(response.json()["user"]["connections"]["Spotify"]["LoginSetting"]["LoginUrl"])
    return response


def get_user_daily_transfer_left(session: requestium.Session):
    endpoint = "https://api.tunemymusic.com/v2/Transfer/GetUserDailyTransferLeft"
    params = urllib.parse.urlencode({"isUrl": False})
    response = session.get(endpoint, params=params)

    print(response.status_code)
    print(response.text)


def transfer(session: requestium.Session, platform_name: str, source: bool):
    assert platform_name in ["YouTube", "Spotify"]
    source_or_target = "source" if source else "target"
    endpoint = "https://api.tunemymusic.com/v2/Analytics/Event"
    body = {
        "eventName": f"transfer - {source_or_target} clicked",
        "platformName": platform_name,
    }
    response = session.post(endpoint, json=body)

    print(response.status_code)
    print(response.text)


def start_transfer(
    session: requestium.Session, tmm_id: str, selected_playlists: List[str]
):
    endpoint = "https://api.tunemymusic.com/v2/Transfer/StartTransfer"
    body = {
        "coupon": "",
        "defaultDescription": "",
        "id": tmm_id,
        "isPremium": False,
        "selectedPlaylistsIndexes": selected_playlists,
        "skipPaywall": False,
        "specialPlaylistsBehaviors": {},
        "target": "YouTube",
        "targetAccount": YOUTUBE_ACCOUNT_ID,
    }
    response = session.post(endpoint, json=body)
    print(response.status_code)
    print(response.text)
    if response.status_code != 200:
        raise requests.HTTPError


def get_auth_url(session: requestium.Session, uri: str):
    session.transfer_session_cookies_to_driver()
    wait = WebDriverWait(driver=session.driver, timeout=DEFAULT_AUTH_WAIT)
    session.driver.get(uri)
    print("Waiting for user authorization...")

    wait.until(
        lambda driver: driver.current_url.startswith("https://www.tunemymusic.com")
    )
    session.transfer_driver_cookies_to_session()
    print("User successfully authorized")


def get_socketio(session: requestium.Session, data, params):
    """have to manually make requests"""
    endpoint = "https://api.tunemymusic.com/socket.io"
    params = urllib.parse.urlencode({"EIO": "4"})


def post_socketio(session: requestium.Session, data, params):
    """see get"""


if __name__ == "__main__":
    options = ChromeOptions()
    options.add_argument(f"user-agent={USER_AGENT}")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])

    driver = Chrome(options=options)

    session = requestium.Session(driver=driver)
    response = session.get("https://tunemymusic.com/transfer")
    print(response.status_code)

    session.headers.update({"User-Agent": USER_AGENT})

    print(session.cookies)
    user_data_response = get_user_data(session)
    spotify_auth_url = user_data_response.json()["user"]["connections"]["Spotify"][
        "LoginSetting"
    ]["LoginUrl"]

    youtube_auth_url = user_data_response.json()["user"]["connections"]["YouTube"][
        "LoginSetting"
    ]["LoginUrl"]

    print("\ntransfer\n")
    transfer(session, "Spotify", True)

    print(session.cookies)
    previous_cookies = session.cookies
    print("\nget Spotify auth url\n")
    get_auth_url(session, spotify_auth_url)
    for cookie in session.cookies:
        if cookie not in previous_cookies:
            print(f"NEW COOKIE: {cookie}")

    print("\nload library\n")
    library = load_library(session)
    tmm_id = library.json()["id"]
    print(library.json())
    selected_playlists = [0] * len(library.json()["playlists"])
    for i in range(len(library.json()["playlists"])):
        if library.json()["playlists"][i]["name"] == "test":
            selected_playlists[i] = 1

    print(f"ID: {tmm_id}")
    print(f"selected playlists: {selected_playlists}")

    print("\nget user daily transfer left\n")
    get_user_daily_transfer_left(session)

    print("\ntransfer\n")
    transfer(session, "YouTube", False)

    sio = socketio.Client(
        http_session=session, logger=True, engineio_logger=True, handle_sigint=True
    )

    @sio.event
    def connect():
        print("connnected")

    # @sio.on("transfer")
    # def on_transfer(data):
    #     print("I get here right?")
    #     if data == {}:
    #         sio.emit("transfer", data={"id": tmm_id})
    #     else:
    #         print("I didn't know i could get here")
    #         print(data)

    @sio.on("*", namespace="*")
    def catch_all(event, namespace, sid, data):
        print(f"EVENT MISSED: {event}")
        print(data)

    start_transfer(session, tmm_id, selected_playlists)
    time.sleep(10000000)
    sio.connect("https://api.tunemymusic.com/socket.io")
    print(sio.sid)
    sio.emit(
        "transfer", data={"id": tmm_id}, callback=lambda _: print("Do i get here", _)
    )
    time.sleep(5)
    sio.emit(
        "transfer", data={"id": tmm_id}, callback=lambda _: print("Do i get here", _)
    )

    sio.wait()
    print(sio.sid)
    print(sio.transport())
    event = sio.receive(timeout=DEFAULT_REQUEST_TIMEOUT)
    print(sio.sid)

    while True:
        event = sio.receive(timeout=DEFAULT_REQUEST_TIMEOUT)
        print(event)

    print(session.cookies)
    print(session.headers)
    print("\nget YouTube auth url")
    get_auth_url(session, youtube_auth_url)

    session.close()
    session.driver.quit()
    sio.disconnect()
