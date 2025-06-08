"""
Write an annotations.csv file and organize the audio files in a way that is conducive to training the network
"""

import argparse
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import shutil
import torchaudio
import torch
from torch import Tensor
from typing import Generator, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

from sample_hunter.data_wrangling.obfuscate import obfuscate
from sample_hunter._util import (
    ProcessAtomicCounter,
    DATA_SAVE_DIR,
    NUM_FOLDS,
    SNIPPET_LENGTH,
    PROCS,
    TEMP_DIR,
    AUDIO_DIR,
    ANNOTATIONS_PATH,
    STEP_LENGTH,
    SPECTROGRAM_WIDTH,
    read_into_df,
    save_to_json_or_csv,
)

# a global atomic counter to guarantee unique ids for each file
global_counter = ProcessAtomicCounter(start=1)


def segment(
    signal: Tensor, sr: int, segment_duration: float, overlay: float
) -> Generator[Tensor, None, None]:
    """Generator that yields tensors that represent the segmented signal

    signal: tensor to segment

    sr: sampling rate of the signal

    segment_duration: duration (in seconds) of each segment

    overlay: number of seconds that each segment should overlay with each other
    """

    segment_samples = int(sr * segment_duration)
    overlay_samples = int(sr * overlay)
    end = signal.shape[1]
    for start in range(0, end, overlay_samples):
        segment_end = start + segment_samples
        yield signal[:, start:segment_end]


def prepare_dataset_entries_from_song(
    song_path: Path,
    segment_duration: float,
    overlay: float,
    fold: int,
    save_dir: Path,
    tmp_dir: Path | None = None,
    procs: int = 1,
) -> pd.DataFrame:
    """Given a signal (song) and it's sampling rate: (1) segment it, then (2) obfuscate each segment
    and save the obfuscated segments to disk.

    :return:
    A dataframe with `len(df) == len(segments)` and columns `["song_id", "unob_seg", "ob_seg", and "fold"]`.
    `song_id` is the hash of the full, unobfuscated signal. `ob_seg` is the path to the obfuscated segments.
    `unob_seg` is the path to the unobfuscated segments. `fold` is the fold number
    """
    if not song_path.exists():
        print(
            f"WARNING: {song_path} does not exist and an empty dataframe will be returned for it"
        )
        return pd.DataFrame()

    print(f"Processing {song_path}...")
    signal, sr = torchaudio.load(
        uri=song_path,
        format="mp3",
        backend="ffmpeg",
    )

    song_id = global_counter.fetch_and_increment()

    results = pd.DataFrame(
        columns=["song_id", "full_song_path", "unob_seg", "ob_seg", "fold"]
    )

    with ProcessPoolExecutor(max_workers=procs) as executor:
        futures = [
            executor.submit(
                obfuscate_segment,
                song_path,
                fold,
                save_dir,
                sr,
                tmp_dir,
                song_id,
                unob_seg,
            )
            for unob_seg in segment(
                signal=signal, sr=sr, segment_duration=segment_duration, overlay=overlay
            )
        ]

        for future in as_completed(futures):
            res = pd.DataFrame([future.result()])
            results = pd.concat([results, res], ignore_index=True)
    return results


def obfuscate_segment(
    song_path: Path,
    fold: int,
    save_dir: Path,
    sr: int,
    tmp_dir: Path | None,
    song_id: int,
    unob_seg: Tensor,
) -> dict:
    """Given a path to an unobfuscated Tensor representing a segment, first save that Tensor to disk,
    then obfuscate it using pysox and save the obfuscated segment to disk as well
    """
    unob_id = global_counter.fetch_and_increment()
    ob_id = global_counter.fetch_and_increment()
    unob_seg_path = save_dir / f"fold-{str(fold)}" / (str(unob_id) + ".mp3")
    # first, save the unobfuscated segment to disk
    torchaudio.save(
        uri=unob_seg_path,
        src=unob_seg,
        sample_rate=sr,
        format="mp3",
        backend="ffmpeg",
    )
    # obfuscate the segment
    ob_seg_path = Path(unob_seg_path.parent / (str(ob_id) + unob_seg_path.suffix))
    ob_seg_path = obfuscate(
        unob_seg_path,
        out=ob_seg_path,
        tmp_dir=tmp_dir,
    )

    ob_seg_path = ob_seg_path.absolute()
    unob_seg_path = unob_seg_path.absolute()

    return {
        "song_id": str(song_id),
        "full_song_path": str(song_path),
        "unob_seg": str(unob_seg_path),
        "ob_seg": str(ob_seg_path),
        "fold": fold,
    }


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
        help="The directory to put preprocessed audio files",
        default=AUDIO_DIR,
    )

    parser.add_argument(
        "--annotations",
        type=Path,
        help="Where to write a json or csv file with the annotations",
        default=ANNOTATIONS_PATH,
    )

    parser.add_argument(
        "--temp-dir",
        type=Path,
        help="Where to write the temporary files for obfuscation",
        default=TEMP_DIR,
    )

    parser.add_argument(
        "--folds", type=int, help="The number of folds to make", default=NUM_FOLDS
    )

    parser.add_argument(
        "--segment-duration",
        type=float,
        help="The length (in seconds) of each sample",
        default=SPECTROGRAM_WIDTH,
    )

    parser.add_argument(
        "--overlay",
        help="The number of seconds of overlap between segments",
        type=float,
        default=STEP_LENGTH,
    )

    parser.add_argument(
        "--procs",
        type=int,
        help="The number of cores to use for multiprocessing",
        default=PROCS,
    )

    parser.add_argument(
        "--single",
        required=False,
        help="Option to run only a single song, useful for debugging",
        type=Path,
    )

    parser.add_argument(
        "--num",
        help="Option to take a random sample from the total data for debugging",
        required=False,
        type=int,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.single:
        if args.audio_dir.exists():
            print(
                f"WARNING: The directory {args.audio_dir} already exists and will be deleted."
            )
            reply = input("Do you want to continue? [y/n]: ").strip().lower()
            if reply == "y":
                shutil.rmtree(args.audio_dir)
                args.audio_dir.mkdir()
            else:
                print("Operation canceled.")
                exit()
        else:
            args.audio_dir.mkdir()

        Path(args.audio_dir / "fold-1").mkdir()

        df = prepare_dataset_entries_from_song(
            song_path=Path(args.single),
            segment_duration=args.segment_duration,
            overlay=args.overlay,
            fold=1,
            save_dir=args.audio_dir,
            tmp_dir=args.temp_dir,
            procs=1,
        )
        print(df)
        save_to_json_or_csv(args.annotations, df)
        print("All good!")
        exit(0)

    assert args.in_.exists()

    df = read_into_df(args.in_)
    df = df[~df["path"].isnull()]
    if args.num:
        df = df.sample(args.num)

    annotations = pd.DataFrame(
        columns=["song_id", "full_song_path", "unob_seg", "ob_seg", "fold"],
    ).astype(
        {
            "song_id": "str",
            "full_song_path": "str",
            "unob_seg": "str",
            "ob_seg": "str",
            "fold": "int",
        }
    )

    # clean the out dir if it exists
    if args.audio_dir.exists():
        print(
            f"WARNING: The directory {args.audio_dir} already exists and will be deleted."
        )
        reply = input("Do you want to continue? [y/n]: ").strip().lower()
        if reply == "y":
            shutil.rmtree(args.audio_dir)
            args.audio_dir.mkdir()
        else:
            print("Operation canceled.")
            exit()
    else:
        args.audio_dir.mkdir()

    if args.temp_dir.exists():
        print(f"Also cleaning {args.temp_dir}...")
        shutil.rmtree(args.temp_dir)
        args.temp_dir.mkdir()
    else:
        args.temp_dir.mkdir()

    fold_size = len(df) // args.folds

    for i in range(args.folds):
        print(f"Making fold {i + 1}...")
        fold_dir = Path(args.audio_dir / f"fold-{i + 1}")
        fold_dir.mkdir()

        if i == args.folds - 1:
            fold = df
        else:
            fold = df.sample(n=fold_size)
            df = df.drop(fold.index)

        with ProcessPoolExecutor(max_workers=args.procs) as executor:
            futures = {
                executor.submit(
                    prepare_dataset_entries_from_song,
                    Path(row["path"]),
                    args.segment_duration,
                    args.overlay,
                    i + 1,
                    args.audio_dir,
                    args.temp_dir,
                    args.procs,
                ): row["path"]
                for _, row in fold.iterrows()
            }

            for future in tqdm(
                as_completed(futures), total=len(futures), desc=f"Making fold {i + 1}"
            ):
                try:
                    res = future.result()
                    if not res.empty:
                        annotations = pd.concat(
                            [annotations, pd.DataFrame(res)], ignore_index=True
                        )
                except Exception as e:
                    print(f"Processing failed for {futures[future]}")
                    print(str(e))
                    raise

    save_to_json_or_csv(args.annotations, annotations)
    print(f"Annotations saved to {args.annotations}")
    print(f"Audio files saved to {args.audio_dir}")
