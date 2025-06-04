import urllib.parse
import os
import requests
from base64 import b64encode
import http.server
import socketserver

SPOTIFY_AUTH_PORT: int = 3000
SPOTIFY_CLIENT_ID = os.environ["SPOTIFY_CLIENT_ID"]
SPOTIFY_CLIENT_SECRET = os.environ["SPOTIFY_CLIENT_SECRET"]
DEFAULT_SCOPE = "playlist-modify-public playlist-modify-private playlist-read-private user-read-private user-read-email"

auth_code = None


class Handler(http.server.SimpleHTTPRequestHandler):
    """Localhost server to handle code after user authorization"""

    def do_GET(self):
        from urllib.parse import urlparse, parse_qs

        query = urlparse(self.path).query
        params = parse_qs(query)
        code = params.get("code")
        if code:
            global auth_code
            auth_code = code[0]
            self.send_response(200)
            self.end_headers()
            self.wfile.write(f"Authorization code: {auth_code}".encode())
            print("Authorization code:", auth_code)
        else:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Missing code parameter.")


def print_auth_url(redirect_uri: str, scope: str = DEFAULT_SCOPE):

    params = {
        "client_id": SPOTIFY_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": scope,
    }

    endpoint = "https://accounts.spotify.com/authorize"
    url = endpoint + "?" + urllib.parse.urlencode(params)
    print(f"Go to this link to authorize access: {url}")


def get_access_token_from_code(
    auth_code: str,
    redirect_uri: str,
    client_id: str = SPOTIFY_CLIENT_ID,
    client_secret: str = SPOTIFY_CLIENT_SECRET,
) -> dict:
    """
    Exchange the authorization code for an access token that can modify playlists
    """
    endpoint = "https://accounts.spotify.com/api/token"
    auth = b64encode(f"{client_id}:{client_secret}".encode()).decode()
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": f"Basic {auth}",
    }
    body = {
        "grant_type": "authorization_code",
        "code": auth_code,
        "redirect_uri": redirect_uri,
    }
    response = requests.post(endpoint, headers=headers, data=body)
    if response.status_code == 200:
        return response.json()
    raise requests.HTTPError(
        f"Requesting access token threw a bad response code: {response.status_code} - {response.text}"
    )


def authorize(port: int = SPOTIFY_AUTH_PORT) -> str:
    global auth_code
    with socketserver.TCPServer(("", port), Handler) as httpd:
        print(f"Serving at port {port}, waiting for authorization...")
        while auth_code is None:
            httpd.handle_request()
    return auth_code


def update_env_file(access_token: str, refresh_token: str, env_path: str = ".env"):
    lines = []
    with open(env_path, "r") as f:
        for line in f.readlines():
            if line.startswith("SPOTIFY_ACCESS_TOKEN="):
                lines.append(f'SPOTIFY_ACCESS_TOKEN="{access_token}"\n')
            elif line.startswith("SPOTIFY_REFRESH_TOKEN="):
                lines.append(f'SPOTIFY_REFRESH_TOKEN="{refresh_token}"\n')
            else:
                lines.append(line)
    # check that they will be written
    if not any(["SPOTIFY_ACCESS_TOKEN" in line for line in lines]):
        lines.append(f'SPOTIFY_ACCESS_TOKEN="{access_token}"\n')
    if not any(["SPOTIFY_REFRESH_TOKEN" in line for line in lines]):
        lines.append(f'SPOTIFY_REFRESH_TOKEN="{refresh_token}"\n')
    print(lines)

    with open(".env", "w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    redirect_uri = f"http://127.0.0.1:{SPOTIFY_AUTH_PORT}"
    print_auth_url(redirect_uri)
    code = authorize()

    response = get_access_token_from_code(code, redirect_uri)

    print(f"Access token: {response["access_token"]}")
    print(f"Refresh token: {response["refresh_token"]}")

    update_env_file(response["access_token"], response["refresh_token"])
