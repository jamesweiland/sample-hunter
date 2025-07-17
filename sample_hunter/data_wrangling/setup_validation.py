"""
Script to set up the validation data shards to be uploaded to the hf-dataset
"""

import io
import json
import tarfile
import pandas as pd
from pathlib import Path
from mutagen.mp3 import MP3

SHARD_SIZE = int(1e9)  # bytes


def get_duration(path: Path) -> int:
    """Given a path to an mp3 file, return the duration"""

    return MP3(path).info.length


def create_validation_shards(
    df: pd.DataFrame, out_dir: Path, shard_size: int = SHARD_SIZE
):
    """Create shards for validation set"""

    current_shard = 0
    current_shard_size = 0
    tar = None
    song_counter = 1

    for idx, row in df.iterrows():
        ground_path = Path(
            f"/home/james/code/sample-hunter/_data/validation/{row["ground"]}.mp3"
        )
        positive_path = Path(
            f"/home/james/code/sample-hunter/_data/validation/{row["positive"]}.mp3"
        )

        title = Path(row["path"]).stem
        ground_duration = get_duration(ground_path)
        positive_duration = get_duration(positive_path)

        base_name = f"{song_counter:04d}"
        new_ground_name = f"{base_name}.a.mp3"
        new_positive_name = f"{base_name}.b.mp3"
        metadata_name = f"{base_name}.json"

        metadata = {
            "title": title,
            "ground_duration": ground_duration,
            "positive_duration": positive_duration,
            "ground_path": new_ground_name,
            "positive_path": new_positive_name,
            "ground_id": row["sampled"],
            "positive_id": row["sampling"],
        }
        metadata_bytes = json.dumps(metadata).encode("utf-8")
        mp3_size = ground_path.stat().st_size + positive_path.stat().st_size
        json_size = len(metadata_bytes)

        if tar is None or (current_shard_size + mp3_size + json_size) > shard_size:
            if tar is not None:
                tar.close()
            current_shard += 1
            shard_path = out_dir / f"validation-{current_shard:04d}.tar"
            tar = tarfile.open(shard_path, "w")
            current_shard_size = 0

        tar.add(ground_path, arcname=new_ground_name)
        tar.add(positive_path, arcname=new_positive_name)
        json_info = tarfile.TarInfo(name=metadata_name)
        json_info.size = json_size
        tar.addfile(json_info, fileobj=io.BytesIO(metadata_bytes))
        current_shard_size += mp3_size + json_size
        song_counter += 1

    if tar is not None:
        tar.close()


if __name__ == "__main__":
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 0)
    audio_dir = Path("_data/validation/")
    df = pd.read_csv("_data/relationships-7-7-2025.csv", dtype=str)
    df = df[(~df["ground"].isnull()) & (df["ground"] != "nan")]
    df = df[(~df["positive"].isnull()) & (df["positive"] != "nan")]

    df["positive_name"] = (
        df["path"].map(lambda p: Path(p).stem).str.extract(r"\[(.+)\]")
    )
    df["sampled_name"] = df["path"].map(lambda p: Path(p).stem)

    sampling_id_map = {v: i for i, v in enumerate(df["positive_name"].unique())}
    sampled_id_map = {
        v: i + len(df["positive_name"].unique())
        for i, v in enumerate(df["sampled_name"].unique())
    }

    df["sampling"] = df["positive_name"].map(sampling_id_map)
    df["sampled"] = df["sampled_name"].map(sampled_id_map)

    create_validation_shards(df, Path("_data/validation-shards"))
