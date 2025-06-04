import json
import argparse
from pathlib import Path
import zipfile
from pandas import DataFrame
import pandas as pd
import re

from tqdm import tqdm

from sample_hunter._util import DATA_SAVE_DIR, ZIP_ARCHIVE_DIR
from sample_hunter.data_wrangling.models import SampledSong, SamplingSong


def get_song_title(mp3: str) -> str:
    """Get the title of the sampler song."""
    pattern = r"\[\s*[^\-]+-\s*'([^']+)'"
    match = re.search(pattern, mp3)
    if match:
        return match.group(1).strip()
    return ""


def entry_match_idx(df: DataFrame, row: dict, *cols: str):
    """
    Returns the index of the first row in `df` where all columns in `cols` match the values in `row`.
    If no match is found, returns None.
    """
    if df.empty:
        return None

    mask = pd.Series([True] * len(df))
    for col in cols:
        mask &= df[col] == row[col]
    matches = df[mask]
    if not matches.empty:
        return matches.index[0]
    return None


def insert_mp3(
    sampling_songs: DataFrame,
    samples: DataFrame,
    relationships: DataFrame,
    sampled_song: SampledSong,
    sampling_song: SamplingSong,
    path: Path,
    sample_set: str,
):
    """Insert the mp3 as a row into the dataframe."""

    # need to explicitly set the indices of the new entries so
    # relationships remains consistent

    sampling_idx = entry_match_idx(
        sampling_songs, sampling_song.model_dump_properties(), "title", "song_artist"
    )
    sample_idx = entry_match_idx(
        samples, sampled_song.model_dump_properties(), "title", "song_artist"
    )

    if sampling_idx is None:
        sampling_idx = sampling_songs.index.max() + 1 if not sampling_songs.empty else 0
        sampling_songs = pd.concat(
            [
                sampling_songs,
                DataFrame(sampling_song.model_dump_properties(), index=[sampling_idx]),
            ]
        )

    if sample_idx is None:
        sample_idx = samples.index.max() + 1 if not samples.empty else 0
        samples = pd.concat(
            [
                samples,
                DataFrame(sampled_song.model_dump_properties(), index=[sample_idx]),
            ]
        )

    relationship = DataFrame(
        {
            "sampling": [sampling_idx],
            "sampled": [sample_idx],
            "path": [path.__str__()],
            "sample_set": sample_set,
        }
    )

    if (
        entry_match_idx(relationships, relationship.to_dict(), "sampling", "sampled")
        is None
    ):
        relationships = pd.concat(
            [
                relationships,
                relationship,
            ],
            ignore_index=True,
        )

    return sampling_songs, samples, relationships


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--in",
        type=Path,
        dest="in_",
        help="the path to a summary JSON file of the data to process",
        default=Path(DATA_SAVE_DIR / "summary.json"),
    )

    parser.add_argument(
        "--archive",
        required=False,
        type=Path,
        default=ZIP_ARCHIVE_DIR,
        help="if provided, archive the zip files from --in",
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=DATA_SAVE_DIR,
        help="The directory to save the extracted zip files",
    )

    parser.add_argument(
        "--out-mode",
        type=str,
        default="csv",
        choices=["json", "csv"],
        help="Choose to save the preprocessed data as either json or csv",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    assert args.in_.exists()

    with open(args.in_, "r") as file:
        sample_sets = json.load(file)

    sampling_songs = DataFrame(columns=list(SamplingSong.properties()))
    samples = DataFrame(columns=list(SampledSong.properties()))
    relationships = DataFrame(columns=["sampling", "sampled", "path", "sample_set"])

    for sample_set in tqdm(sample_sets):
        # unzip the file
        zip_path = Path(sample_set["download_path"])
        extracted_dir = Path(args.out / zip_path.stem)
        extracted_dir.mkdir(exist_ok=True)

        print(f"Extracting {sample_set["download_path"]} for {sample_set}")
        with zipfile.ZipFile(sample_set["download_path"], "r") as file:
            file.extractall(extracted_dir)

        for mp3_path in extracted_dir.rglob("*.mp3"):
            print(f"Processing {mp3_path}...")
            sampled_song = SampledSong(path_=mp3_path)
            print(f"Sampled song: {sampled_song.model_dump_properties}")
            sampling_song = SamplingSong(path_=mp3_path)
            print(f"Sampling song: {sampling_song.model_dump_properties()}")
            sampling_songs, samples, relationships = insert_mp3(
                sampling_songs=sampling_songs,
                samples=samples,
                relationships=relationships,
                sampled_song=sampled_song,
                sampling_song=sampling_song,
                path=mp3_path,
                sample_set=sample_set["name"],
            )

            # artist, year in parentheses, then name of the album in parent path
            # in the mp3 file itself, it's the sample artist, dash, name of sample, then in brackets its artist, dash, and in single quotes the name of the song
    print(
        f"Missing fields in sampling_songs: {(sampling_songs == "").sum().sum()} / {sampling_songs.size}"
    )
    print(f"Missing fields in samples: {(samples == "").sum().sum()} / {samples.size}")
    print(f"Number of pairs: {len(relationships)}")
    print(f"Number of sampling songs: {len(sampling_songs)}")
    print(f"Number of samples: {len(samples)}")

    sampling_songs_save_path = Path(args.out / f"sampling_songs.{args.out_mode}")
    print(f"Saving sampling_songs to {sampling_songs_save_path}")

    samples_save_path = Path(args.out / f"samples.{args.out_mode}")
    print(f"Saving samples to {samples_save_path}")

    relationships_save_path = Path(args.out / f"relationships.{args.out_mode}")
    print(f"Saving relationships to {relationships_save_path}")

    if args.out_mode == "json":
        sampling_songs.to_json(sampling_songs_save_path, indent=4, orient="index")
        samples.to_json(samples_save_path, indent=4, orient="index")
        relationships.to_json(relationships_save_path, indent=4, orient="index")
    else:
        sampling_songs.to_csv(sampling_songs_save_path)
        samples.to_csv(samples_save_path)
        relationships.to_csv(relationships_save_path)

    if args.archive:
        # at the end, archive all the zip files
        print(f"Moving zip files to {args.archive}")
        if not args.archive.exists():
            args.archive.mkdir(parents=True, exist_ok=True)
        for sample_set in sample_sets:
            file = Path(sample_set["download_path"])
            file.rename(args.archive / file.name)

    # for each file, make an entry in a dataframe
    # columns: path, parent folder, song title, artist, album
    # 185.220.101.5
