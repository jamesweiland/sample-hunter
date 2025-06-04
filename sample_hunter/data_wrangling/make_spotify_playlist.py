from pathlib import Path
from typing import List
import pandas as pd
import requests
import os
import urllib.parse
import argparse

from tqdm import tqdm
from sample_hunter.data_wrangling.models import SamplingSong
from sample_hunter._util import DATA_SAVE_DIR
from sample_hunter.data_wrangling.spotify_api import SpotifyAPI


def create_playlist(
    api: SpotifyAPI,
    name: str,
    user_id: str,
    description: str = "",
    public: bool = False,
):
    """Make a new playlist for the user."""
    endpoint = f"https://api.spotify.com/v1/users/{user_id}/playlists"
    body = {"name": name, "description": description, "public": public}
    headers = {"Content-Type": "application/json"}

    response = api.post(endpoint, headers=headers, body=body)

    if response.status_code == 201:
        print(f"Spotify playlist created with name {name}")
        return response.json()
    else:
        raise requests.HTTPError(
            f"Creating playlist has bad response code: {response.status_code} {response.text}"
        )


def add_tracks_to_playlist(
    api: SpotifyAPI, playlist_id: str, track_ids: List[str]
) -> str:
    """Add a track to the playlist. Returns a snapshot id for the playlist"""
    endpoint = f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks"
    uris = [f"spotify:track:{track_id}" for track_id in track_ids]
    print(uris)
    body = {"uris": uris}
    response = api.post(endpoint, body=body)
    if response.status_code == 201:
        return response.json()["snapshot_id"]
    else:
        raise requests.HTTPError(
            f"Failed to add tracks to playlist: {response.status_code} {response.text}"
        )


def search_for_track(api: SpotifyAPI, song: SamplingSong):
    """Search for a Spotify track given the details of a Song."""
    endpoint = "https://api.spotify.com/v1/search"
    q = f"track:{song.title}"
    q += f" artist:{song.song_artist}"
    if song.album_title != "":
        q += f" album:{song.album_title}"
    if song.year_released != "":
        q += f" year:{song.year_released}"

    params = urllib.parse.urlencode({"q": q, "type": "track", "limit": 1})
    response = api.get(endpoint, params=params)

    if response.status_code == 200:
        href = response.json()["tracks"]["href"]
        response = api.get(href)
        if response.status_code == 200:
            return response.json()
        else:
            raise requests.HTTPError(
                f"Failed to search for track: {response.status_code} {response.text}"
            )
    else:
        raise requests.HTTPError(
            f"Failed to search for track: {response.status_code} {response.text}"
        )


def get_user_profile(api: SpotifyAPI):
    endpoint = "https://api.spotify.com/v1/me"
    response = api.get(endpoint)
    if response.status_code == 200:
        return response.json()
    else:
        raise requests.HTTPError(
            f"Failed to get user profile: {response.status_code} {response.text}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--in",
        type=Path,
        dest="in_",
        help="The path to a csv or json file of songs that fit the SamplingSong schema",
        default=Path(DATA_SAVE_DIR / "sampling_songs.csv"),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # user_id = get_user_profile(token=SPOTIFY_ACCESS_TOKEN)["id"]

    # playlist = create_playlist(name="test", user_id=user_id)
    # playlist_id = playlist["id"]
    # uri = ["spotify:track:0xkeObNA3vzp5tzpXNwbbF"]
    # res = add_track_to_playlist(id=playlist_id, uris=uri)

    assert args.in_.exists()
    sampling_songs = pd.read_csv(args.in_)

    api = SpotifyAPI()
    user_id = get_user_profile(api)["id"]
    track_ids = []
    sampling_songs["track_sid"] = None
    sampling_songs["playlist_sid"] = None
    for idx, row in tqdm(
        sampling_songs.iterrows(), total=len(sampling_songs), desc="Getting track sids"
    ):
        song = SamplingSong(path_=row["path"])
        print(f"Processing song: {song.title}")
        search_response = search_for_track(api, song)
        search_response_items = search_response["tracks"]["items"]
        if not search_response_items:
            continue
        track_id = search_response["tracks"]["items"][0]["id"]
        row["track_sid"] = track_id
        sampling_songs.at[idx, "track_sid"] = track_id
        track_ids.append(track_id)

    # split track ids into "batches" to add to playlists
    batches = [
        track_ids[i : i + 100] for i in range(0, len(track_ids), 100)
    ]  # can only add 100 at a time
    playlist_ids = []
    current_playlist_id = create_playlist(
        api=api, name=f"samples-{len(playlist_ids)}", user_id=user_id
    )["id"]
    playlist_ids.append(current_playlist_id)
    i = 0
    for batch in tqdm(batches):
        # need to keep all playlist sizes below 500 so i can use the free version of tunemymusic
        if i + len(batch) > 500:
            # split the batch
            batch_first_half = batch[0 : (i + len(batch) - 500)]
            batch_second_half = batch[(i + len(batch) - 500) :]
            batch = batch_first_half
            batches.append(batch_second_half)
        if len(batch) == 0:
            continue
        add_tracks_to_playlist(api, current_playlist_id, batch)
        sampling_songs.loc[sampling_songs["track_sid"].isin(batch), "playlist_id"] = (
            current_playlist_id
        )
        i += len(batch)
        if i >= 500:
            current_playlist_id = create_playlist(
                api, name=f"samples-{len(playlist_ids)}", user_id=user_id
            )["id"]
            playlist_ids.append(current_playlist_id)
            i = 0

    sampling_songs.to_csv(args.in_)
    print("All good!")
