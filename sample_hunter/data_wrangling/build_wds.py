"""
Starting with a folder of mp3 files and a csv of annotations,
organize these into tar shards and upload to HF.
"""

import io
import json
from pathlib import Path
import warnings
import pandas as pd
import tarfile
from sklearn.model_selection import train_test_split

SHARD_SIZE = int(1e9)  # bytes


def create_shards(
    df: pd.DataFrame, out_dir: Path, name: str, shard_size: int = SHARD_SIZE
):
    """
    Create tar shards
    """

    current_shard = 0
    current_shard_size = 0
    tar = None
    song_counter = 1
    for idx, row in df.iterrows():
        mp3_path = Path(row["path"])
        if not mp3_path.exists():
            warnings.warn(f"Could not find {mp3_path}")
            continue

        base_name = f"{song_counter:04d}"
        new_mp3_name = f"{base_name}.mp3"
        json_name = f"{base_name}.json"
        song_counter += 1

        json_bytes = json.dumps(
            {"title": row["title"], "artist": row["song_artist"]}
        ).encode("utf-8")

        mp3_size = mp3_path.stat().st_size
        json_size = len(json_bytes)

        if tar is None or (current_shard_size + mp3_size + json_size) > shard_size:
            if tar is not None:
                tar.close()
            current_shard += 1
            shard_path = out_dir / f"{name}-{current_shard:04d}.tar"
            tar = tarfile.open(shard_path, "w")
            current_shard_size = 0

        tar.add(mp3_path, arcname=new_mp3_name)
        json_info = tarfile.TarInfo(name=json_name)
        json_info.size = json_size
        tar.addfile(json_info, fileobj=io.BytesIO(json_bytes))

        current_shard_size += mp3_size + json_size

    if tar is not None:
        tar.close()


if __name__ == "__main__":
    audio_dir = Path("_data/source")
    df = pd.read_csv("_data/samples.csv")
    df = df[~df["path"].isnull()]

    train_df, test_df = train_test_split(df, test_size=0.25)

    train_output_dir = Path("_data/webdataset-shards/train")
    test_output_dir = Path("_data/webdataset-shards/test")

    create_shards(train_df, train_output_dir, "train")
    create_shards(test_df, test_output_dir, "test")
    print("done")
