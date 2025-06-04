from base64 import b64encode
import os
from retry import retry
import requests
from sample_hunter._util import DEFAULT_RETRIES, DEFAULT_RETRY_DELAY
from sample_hunter.data_wrangling.spotify_auth import update_env_file


class SpotifyAPI:
    access_token: str
    refresh_token: str
    client_id: str
    client_secret: str

    def __init__(
        self,
        access_token: str | None = None,
        refresh_token: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
    ):
        self.access_token = access_token or os.environ.get("SPOTIFY_ACCESS_TOKEN", "")
        self.refresh_token = refresh_token or os.environ.get(
            "SPOTIFY_REFRESH_TOKEN", ""
        )
        self.client_id = client_id or os.environ.get("SPOTIFY_CLIENT_ID", "")
        self.client_secret = client_secret or os.environ.get(
            "SPOTIFY_CLIENT_SECRET", ""
        )

        assert (
            self.access_token != ""
            and self.refresh_token != ""
            and self.client_id != ""
            and self.client_secret != ""
        )

    @retry(requests.HTTPError, tries=DEFAULT_RETRIES, delay=DEFAULT_RETRY_DELAY)
    def get(
        self,
        endpoint: str,
        headers: dict = {},
        params: str | None = None,
        body: dict | None = None,
    ) -> requests.Response:
        """Make a GET request to the Spotify API."""
        headers["Authorization"] = f"Bearer {self.access_token}"
        response = requests.get(endpoint, headers=headers, params=params, json=body)
        if 200 <= response.status_code < 300:
            return response
        elif response.status_code == 401:
            print("Access token expired. Refreshing...")

            self.refresh_access_token(self.refresh_token)
            raise requests.HTTPError("Access token is expired")
        else:
            print(
                f"Failed making GET request to Spotify API {response.status_code} {response.text}"
            )
            raise requests.HTTPError

    @retry(requests.HTTPError, tries=DEFAULT_RETRIES, delay=DEFAULT_RETRY_DELAY)
    def post(
        self,
        endpoint: str,
        headers: dict = {},
        body: dict | None = None,
        params: str | None = None,
    ) -> requests.Response:
        """Make a POST request to the Spotify API."""
        headers["Authorization"] = f"Bearer {self.access_token}"
        response = requests.post(endpoint, headers=headers, json=body, params=params)
        if 200 <= response.status_code < 300:
            return response
        elif response.status_code == 401:
            print("Access token expired. Refreshing...")

            self.refresh_access_token(self.refresh_token)
            raise requests.HTTPError("Access token is expired")
        else:
            print(
                f"Failed making POST request to Spotify API {response.status_code} {response.text}"
            )
            raise requests.HTTPError()

    def refresh_access_token(self, refresh_token: str):
        endpoint = "https://accounts.spotify.com/api/token"
        auth_str = f"{self.client_id}:{self.client_secret}"
        b64_auth_str = b64encode(auth_str.encode()).decode()
        headers = {
            "Authorization": f"Basic {b64_auth_str}",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        data = {"grant_type": "refresh_token", "refresh_token": refresh_token}
        response = requests.post(endpoint, headers=headers, data=data)
        if response.status_code == 200:
            # reset environment/member variables
            os.environ["SPOTIFY_ACCESS_TOKEN"] = response.json()["access_token"]
            self.access_token = response.json()["access_token"]
            if response.json().get("refresh_token") is not None:
                os.environ["SPOTIFY_REFRESH_TOKEN"] = response.json()["refresh_token"]
                self.refresh_token = response.json()["refresh_token"]

            # update the .env file so the change in tokens will persist after the script ends
            update_env_file(self.access_token, self.refresh_token)

        else:
            raise requests.HTTPError(
                f"Failed to refresh token: {response.status_code} {response.text}"
            )
