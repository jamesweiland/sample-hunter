"""
Starting with a folder of mp3 files and a csv of annotations,
organize these into tar shards and upload to HF.
"""

import io
import json
from pathlib import Path
import warnings
import tarfile
from sklearn.model_selection import train_test_split
import ffmpeg
from tqdm import tqdm

SHARD_SIZE = int(1e9)  # bytes
SONG_COUNTER = 0


def create_shards(
    files: list[Path], out_dir: Path, name: str, shard_size: int = SHARD_SIZE
):
    """
    Create tar shards
    """
    global SONG_COUNTER

    current_shard = 0
    current_shard_size = 0
    tar = None

    for file in tqdm(files, desc=f"Building {out_dir}..."):
        if not file.exists():
            warnings.warn(f"Could not find {file}")
            continue
        metadata = ffmpeg.probe(file)

        stream_metadata = None
        for stream in metadata["streams"]:
            if stream["codec_name"] == "mp3":
                stream_metadata = stream
                break
        assert stream_metadata is not None

        base_name = f"{SONG_COUNTER:04d}"
        new_mp3_name = f"{base_name}.mp3"
        json_name = f"{base_name}.json"

        json_bytes = json.dumps(
            {
                "title": metadata["format"]["tags"]["title"],
                "artist": metadata["format"]["tags"]["artist"],
                "duration": stream_metadata["duration"],
                "sample_rate": stream_metadata["sample_rate"],
                "id": SONG_COUNTER,
            }
        ).encode("utf-8")

        mp3_size = file.stat().st_size
        json_size = len(json_bytes)

        if tar is None or (current_shard_size + mp3_size + json_size) > shard_size:
            if tar is not None:
                tar.close()
            current_shard += 1
            shard_path = out_dir / f"{name}-{current_shard:04d}.tar"
            tar = tarfile.open(shard_path, "w")
            current_shard_size = 0

        tar.add(file, arcname=new_mp3_name)
        json_info = tarfile.TarInfo(name=json_name)
        json_info.size = json_size
        tar.addfile(json_info, fileobj=io.BytesIO(json_bytes))

        current_shard_size += mp3_size + json_size
        SONG_COUNTER += 1

    if tar is not None:
        tar.close()


if __name__ == "__main__":
    audio_dir = Path("_data/source")
    files = list(audio_dir.rglob("*.mp3"))

    train, test = train_test_split(files, test_size=0.25)

    train_output_dir = Path("_data/webdataset-shards/train")
    test_output_dir = Path("_data/webdataset-shards/test")

    create_shards(train, train_output_dir, "train")
    create_shards(test, test_output_dir, "test")
    print("done")
