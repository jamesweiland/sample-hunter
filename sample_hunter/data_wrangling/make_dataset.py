"""
Write an annotations.csv file and organize the audio files in a way that is conducive to training the network
"""

import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from sample_hunter._util import DATA_SAVE_DIR, read_into_df, save_to_json_or_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--in",
        type=Path,
        dest="in_",
        help="The path to a json or csv file",
        default=Path(DATA_SAVE_DIR / "samples.csv"),
    )

    parser.add_argument(
        "--audio-dir",
        type=Path,
        help="The directory to move audio files",
        default=Path(DATA_SAVE_DIR / "audio-dir/"),
    )

    parser.add_argument(
        "--annotations",
        type=Path,
        help="Where to write a json or csv file with the annotations",
        default=Path(DATA_SAVE_DIR / "annotations.csv"),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    assert args.in_.exists()

    df = read_into_df(args.in_)
    args.audio_dir.mkdir(exist_ok=True)

    annotations = pd.DataFrame(index=df.index, columns=["unobfuscate", "obfuscate"])

    df = df[~df["path"].isnull()]
    df = df[~df["obfuscate_path"].isnull()]

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            print(row["obfuscate_path"])

            unobfuscated_path = Path(row["path"])
            obfuscated_path = Path(row["obfuscate_path"])
            new_unobfuscated_path = args.audio_dir / unobfuscated_path.name
            new_obfuscated_path = args.audio_dir / obfuscated_path.name

            if not new_obfuscated_path.exists():
                unobfuscated_path.rename(args.audio_dir / unobfuscated_path.name)
            if not new_obfuscated_path.exists():
                obfuscated_path.rename(args.audio_dir / obfuscated_path.name)

            annotations.iloc[idx] = {  # type: ignore
                "unobfuscate": str(unobfuscated_path.absolute()),
                "obfuscate": str(obfuscated_path.absolute()),
            }
        except FileNotFoundError as e:
            print(f"File not found. maybe it was already moved?")
            print(str(e))

    save_to_json_or_csv(args.annotations, annotations)

    print(f"Annotations saved to {args.annotations}")
    print(f"Audio files moved to {args.audio_dir}")
