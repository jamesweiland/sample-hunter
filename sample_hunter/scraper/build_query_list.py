import requests
import pandas as pd
import argparse
import time
from typing import List, Dict

MUSICBRAINZ_API = "https://musicbrainz.org/ws/2"
HEADERS = {"User-Agent": "SampleHunter/1.0 (your_email@domain.com)"}

def search_artist(artist_name: str) -> List[Dict]:
    """Search for an artist and return possible matches (MusicBrainz IDs)."""
    url = f"{MUSICBRAINZ_API}/artist/"
    params = {"query": artist_name, "fmt": "json"}
    resp = requests.get(url, params=params, headers=HEADERS)
    resp.raise_for_status()
    return resp.json().get('artists', [])

def get_recordings_by_artist(mb_artist_id: str, limit=100) -> List[Dict]:
    """Get tracks for a MusicBrainz artist ID."""
    url = f"{MUSICBRAINZ_API}/recording/"
    params = {
        "artist": mb_artist_id,
        "limit": limit,      # Max 100 per request, can page
        "fmt": "json"
    }
    resp = requests.get(url, params=params, headers=HEADERS)
    resp.raise_for_status()
    return resp.json().get('recordings', [])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artists", nargs="+", help="List of artist names to fetch")
    parser.add_argument("--artist-file", type=str, help="Text file with artist names, one per line")
    parser.add_argument("--out", type=str, default="query_list.csv")
    parser.add_argument("--per-artist", type=int, default=100, help="Max recordings per artist")
    args = parser.parse_args()

    # Load artist names from file or CLI
    if args.artist_file:
        with open(args.artist_file) as f:
            artist_names = [line.strip() for line in f if line.strip()]
    elif args.artists:
        artist_names = args.artists
    else:
        parser.error("Either --artists or --artist-file required")

    if args.artist_file:
        with open(args.artist_file) as f:
            artist_names = [line.strip() for line in f if line.strip()]
    elif args.artists:
        artist_names = args.artists
    else:
        parser.error("Either --artists or --artist-file required")

    all_rows = []
    for artist_name in artist_names:
        print(f"Searching for artist: {artist_name}")
        artists = search_artist(artist_name)
        if not artists:
            print(f"No MusicBrainz artist found for {artist_name}, skipping.")
            continue
        mb_artist = artists[0]  # Take best match
        mb_artist_id = mb_artist["id"]
        print(f"Getting up to {args.per_artist} tracks for artist {artist_name} (MBID {mb_artist_id})")
        recordings = get_recordings_by_artist(mb_artist_id, limit=args.per_artist)
        for rec in recordings:
            all_rows.append({
                "artist": artist_name,
                "title": rec.get("title", ""),
                "album": rec["releases"][0]["title"] if rec.get("releases") else "",
                "year": rec["releases"][0]["date"].split("-")[0] if (rec.get("releases") and rec["releases"][0].get("date")) else "",
                "mb_recording_id": rec.get("id", ""),
                "mb_artist_id": mb_artist_id,
            })
        time.sleep(1)  # Be polite to the API

    df = pd.DataFrame(all_rows)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} queries to {args.out}")

if __name__ == "__main__":
    main()
