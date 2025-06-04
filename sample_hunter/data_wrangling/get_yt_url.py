import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from ytmusicapi import YTMusic
from typing import Optional, List, Dict

from sample_hunter._util import DATA_SAVE_DIR


def find_allowed_response(responses: List[Dict], allowed_types: List[str]) -> Dict:
    for response in responses:
        if response["resultType"] in allowed_types:
            return response
    raise RuntimeError("No allowed types found in search response")


def get_yt_video_id(
    ytmusic: YTMusic, query: str, allowed_types: Optional[List[str]]
) -> str | None:
    print(f"Searching query {query}")
    search_response = ytmusic.search(query)
    if search_response:
        hit = (
            search_response[0]
            if allowed_types is None
            else find_allowed_response(search_response, allowed_types)
        )
        return hit["videoId"]
    else:
        return None


def get_yt_url(ytmusic: YTMusic, videoId: str) -> str:
    video = ytmusic.get_song(videoId)
    return video["microformat"]["microformatDataRenderer"]["urlCanonical"]


def search_song_and_get_url(
    ytmusic: YTMusic,
    query: str,
    allowed_types: Optional[List[str]] = ["song", "video", "upload"],
) -> str:
    """Use the YTMusic API to search for a song with `query` and return the song's URL on youtube."""
    videoId = get_yt_video_id(ytmusic, query, allowed_types)
    if videoId:
        return get_yt_url(ytmusic, videoId)
    raise RuntimeError(f"Could not find videoId for {query}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--in",
        dest="in_",
        help="The path to the directory with the song metadata. Must be either csv or json.",
        default=Path(DATA_SAVE_DIR / "sampling_songs.csv"),
    )

    return parser.parse_args()


def build_query(row: pd.Series) -> str:
    row = row.astype("str")
    query = ""
    if row["title"] not in ["", "nan"]:
        query += row["title"]
    if row["song_artist"] not in ["", "nan"]:
        query += " " + row["song_artist"]
    elif row["album_artist"] not in ["", "nan"]:
        query += " " + row["album_artist"]
    if row["album_title"] not in ["", "nan"]:
        query += " " + row["album_title"]
    if row["year_released"] not in ["", "nan"]:
        query += " " + row["year_released"]
    return query


if __name__ == "__main__":
    args = parse_args()
    assert args.in_.exists()
    assert args.in_.suffix == ".json" or args.in_.suffix == ".csv"

    if args.in_.suffix == ".csv":
        # pd read csv
        sampling_songs = pd.read_csv(args.in_)
    else:
        # pd read json
        sampling_songs = pd.read_json(args.in_, orient="index")
    sampling_songs["yt_url"] = None

    ytmusic = YTMusic()
    for idx, row in tqdm(list(sampling_songs.iterrows())):
        print(f"Searching URL for {row["title"]}")
        query = build_query(row)
        url = search_song_and_get_url(ytmusic, query)
        sampling_songs.at[idx, "yt_url"] = url

    print(f"Saving results to {args.in_}...")
    if args.in_.suffix == ".csv":
        sampling_songs.to_csv(args.in_)
    else:
        sampling_songs.to_json(args.in_, indent=4, orient="index")
