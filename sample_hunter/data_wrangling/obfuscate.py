import multiprocessing
import os
import re
import tempfile
from sox import Transformer, Combiner
from pathlib import Path
import argparse
import random
import pandas as pd
from typing import Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

from sample_hunter._util import DATA_SAVE_DIR, PROCS, SigintHandler

TMP_DIR = Path("/home/james/code/sample-hunter/tmp_dir")
TIMEOUT: int = 60

DEFAULT_PARAMS = {
    "tempo_lb": 0.6,
    "tempo_ub": 1.5,
    "pitch_lb": 0.6,
    "pitch_ub": 1.5,
    "reverb_lb": 40,
    "reverb_ub": 100,
    "lowpass_lb": 1000,
    "lowpass_ub": 3000,
    "highpass_lb": 500,
    "highpass_ub": 1500,
    "whitenoise_lb": 0.05,
    "whitenoise_ub": 0.35,
    "pinknoise_lb": 0.05,
    "pinknoise_ub": 0.35,
}

DEFAULT_REVERSE_PCT: float = 0.05
DEFAULT_CHOP_PCT: float = 0.25

# change tempo

# change pitch

# apply low or high pass filter

# add white or pink noise

# if chop: chop


def get_transformer(**kwargs) -> Tuple[str, Transformer]:
    """Initialize a transformer with random arguments or with those specified by the parameters"""

    tfm = Transformer()
    mods = ""

    tempo = kwargs.get("tempo") or random.uniform(
        DEFAULT_PARAMS["tempo_lb"], DEFAULT_PARAMS["tempo_ub"]
    )
    if abs(tempo - 1.0) <= 0.1:
        tfm.stretch(tempo)
    else:
        tfm.tempo(tempo)
    mods += "t" + str(round(tempo, 2))

    pitch = kwargs.get("pitch") or random.uniform(
        DEFAULT_PARAMS["pitch_lb"], DEFAULT_PARAMS["pitch_ub"]
    )
    tfm.pitch(pitch)
    mods += "p" + str(round(pitch, 2))

    reverb = kwargs.get("reverb") or random.randint(
        DEFAULT_PARAMS["reverb_lb"], DEFAULT_PARAMS["reverb_ub"]
    )
    tfm.reverb(reverb)
    mods += "r" + str(reverb)

    lowpass = kwargs.get("lowpass")
    highpass = kwargs.get("highpass")
    if not (lowpass or highpass):
        # if the user did not specify either low or high pass, randomly choose one to apply
        if bool(random.randint(0, 1)):
            # low pass
            filter = random.randint(
                DEFAULT_PARAMS["lowpass_lb"], DEFAULT_PARAMS["lowpass_ub"]
            )
            tfm.lowpass(filter)
            mods += "l" + str(filter)
        else:
            # high pass
            filter = random.randint(
                DEFAULT_PARAMS["highpass_lb"], DEFAULT_PARAMS["highpass_ub"]
            )
            tfm.highpass(filter)
            mods += "h" + str(filter)
    else:
        if lowpass:
            # apply lowpass
            tfm.lowpass(lowpass)
            mods += "l" + str(lowpass)
        if highpass:
            # apply highpass
            tfm.highpass(highpass)
            mods += "h" + str(highpass)

    return mods, tfm


def make_noise(infile: Path, outfile: Path | None = None, **kwargs) -> Tuple[str, Path]:
    """Generates white or pink noise and returns the path to the synthesized file."""
    if outfile is None:
        outfile = Path(TMP_DIR / (str(os.getpid()) + "noise.mp3"))
    tfm = Transformer()
    whitenoise = kwargs.get("whitenoise")
    pinknoise = kwargs.get("pinknoise")
    noise_mods = ""
    extra_args = []
    if not (whitenoise or pinknoise):
        # if the user did not specify either white or pink noise, randomly choose one to apply
        if bool(random.randint(0, 1)):
            # white noise
            noise_level = random.uniform(
                DEFAULT_PARAMS["whitenoise_lb"], DEFAULT_PARAMS["whitenoise_ub"]
            )
            noise_mods += "wN" + str(round(noise_level, 2))
            extra_args.extend(["synth", "whitenoise", "vol", noise_level])
        else:
            # pink noise
            noise_level = random.uniform(
                DEFAULT_PARAMS["pinknoise_lb"], DEFAULT_PARAMS["pinknoise_ub"]
            )
            noise_mods += "pN" + str(round(noise_level, 2))
            extra_args.extend(["synth", "pinknoise", "vol", noise_level])
    else:
        if whitenoise:
            # white noise
            noise_mods += "wN" + str(round(whitenoise, 2))
            extra_args.extend(["synth", "whitenoise", "vol", whitenoise])
        if pinknoise:
            # pink noise
            noise_mods += "pN" + str(round(pinknoise, 2))
            extra_args.extend(["synth", "pinknoise", "vol", pinknoise])

    with multiprocessing.Lock():
        tfm.build_file(infile, outfile, extra_args=extra_args)
    return str(noise_mods), outfile


def obfuscate(
    input: Path, out: Path | None = None, tmp_dir: Path | None = None
) -> Path:
    """The main function. Accepts a path to an mp3 file and returns the path to the obfuscated file."""

    if tmp_dir is None:
        tmp_dir = TMP_DIR
    # make tmp file
    fd, tmp_file = tempfile.mkstemp(suffix=".mp3", dir=str(tmp_dir))
    os.close(fd)
    tmp_file = Path(tmp_file)
    # make noise tmp file
    noise_fd, tmp_noise_file = tempfile.mkstemp(suffix=".mp3", dir=str(tmp_dir))
    os.close(noise_fd)
    tmp_noise_file = Path(tmp_noise_file)

    mods, tfm = get_transformer()
    noise_mods, noise_path = make_noise(input, outfile=tmp_noise_file)
    mods += noise_mods
    if out is None:
        out = input.parent / (input.stem + mods + input.suffix)

    tfm.build_file(str(input), str(tmp_file))

    try:
        combiner = Combiner().set_input_format(file_type=["mp3", "mp3"])
        combiner.build([str(tmp_file), str(noise_path)], out, "mix")  # type: ignore
        return out  # type: ignore
    except Exception as e:
        print(f"An error occured for {input}")
        print(str(e))
        raise
    finally:
        tmp_file.unlink(missing_ok=True)
        tmp_noise_file.unlink(missing_ok=True)


def clean_data_dir(data_dir: Path):
    """Remove old obfuscations from prior runs"""
    # get all mp3 paths in data_dir
    mp3_paths = list(data_dir.rglob("*.mp3"))

    # create a set of original filenames
    originals = set()
    for path in mp3_paths:
        stem = path.stem

        cleaned_stem = re.sub(r"t[\w\.]+$", "", stem)
        originals.add(cleaned_stem)

    print(originals)
    # Now, delete all files that are not originals
    for path in mp3_paths:
        stem = path.stem
        cleaned_stem = re.sub(r"t[\w\.]+$", "", stem)
        # If the file's stem is not exactly the cleaned stem, it's a variant
        if stem not in originals:
            print(f"Deleting obfuscation: {path}")
            path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--in",
        dest="in_",
        type=Path,
        help="the path to a CSV or JSON file with a column containing paths to obfuscate",
        default=Path(DATA_SAVE_DIR / "samples.csv"),
    )

    parser.add_argument(
        "--col",
        help='the name of the column of the dataframe in "in" containing paths to mp3 files to obfuscate',
        type=str,
        default="path",
    )

    parser.add_argument(
        "--num",
        type=int,
        required=False,
        help="Option to sample the dataframe and only process a sample",
    )

    parser.add_argument(
        "--outdir",
        required=False,
        type=Path,
        help="Option to specify a different directory to save the obfuscated files",
    )

    parser.add_argument(
        "--procs",
        type=int,
        help="number of processes to use for multiprocessing",
        default=PROCS,
    )

    parser.add_argument(
        "--append",
        action="store_true",
        help="Option to not process files that already have a non-null entry in the dataframe",
    )

    parser.add_argument(
        "--single", help="Option to obfuscate a single song", type=Path, required=False
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.single:
        out = obfuscate(args.single)
        print(f"Result saved to: {out}")

    assert args.in_.exists()

    if args.in_.suffix == ".json":
        df = pd.read_json(args.in_, orient="index")
    else:
        df = pd.read_csv(args.in_)

    if not args.append:
        # clean the data
        print(f"Cleaning {args.in_.parent}...")
        clean_data_dir(args.in_.parent)

    if args.num:
        to_process = df.sample(args.num)
    elif args.append:
        to_process = df[df["obfuscate_path"].isnull()]
    else:
        df["obfuscate_path"] = None
        to_process = df

    def f(row):
        """worker function for multiprocessing"""
        p = Path(row["path"])
        print(f"Processing {p}...")
        return obfuscate(p)

    try:
        with ProcessPoolExecutor(max_workers=args.procs) as executor:
            futures = {
                executor.submit(f, row): idx for idx, row in to_process.iterrows()
            }

            for future in tqdm(as_completed(futures), total=len(to_process)):
                idx = futures[future]
                print(f"Processing {df.at[idx, "path"]}...")
                if not Path(df.at[idx, "path"]).exists():
                    print(
                        f"{df.at[idx, "path"]} doesn't exist. that's fucking weird..."
                    )
                    continue
                try:
                    out = future.result(timeout=TIMEOUT)
                    df.at[idx, "obfuscate_path"] = out
                except TimeoutError:
                    print(
                        f"Processing {df.at[idx, "path"]} with process number {os.getpid()} timed out"
                    )
                except Exception as e:
                    print(f"An error occurred obfuscating {df.at[idx, "path"]}")
                    print(str(e))

    # try:
    #     for idx, row in tqdm(to_process.iterrows(), total=len(to_process)):
    #         try:
    #             out = obfuscate(Path(row["path"]))
    #             df.at[idx, row["obfuscate_path"]] = out
    #         except Exception as e:
    #             print(f"An error occured for {row["path"]}")
    #             print(str(e))
    #             raise e

    finally:
        print(f"Saving data to {args.in_}...")
        if args.in_.suffix == ".json":
            df.to_json(args.in_, orient="index", indent=4)
        else:
            df.to_csv(args.in_)

    print("All good!")
