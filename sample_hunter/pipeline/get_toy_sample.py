import pandas as pd
import argparse
from pathlib import Path
import tarfile
from tqdm import tqdm

from sample_hunter._util import ANNOTATIONS_PATH, read_into_df, save_to_json_or_csv

"""sample some data from the full dataset to use for debugging/testing"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num",
        type=int,
        required=True,
        help="The number of data points to sample",
    )

    parser.add_argument(
        "--in",
        dest="in_",
        type=Path,
        default=ANNOTATIONS_PATH,
        help="The path to the annotations file",
    )

    parser.add_argument(
        "--sample-dir",
        type=Path,
        help="The path to store the tarball of sampled files",
        required=True,
    )

    parser.add_argument(
        "--sample-annotations",
        type=Path,
        help="The path to store the sampled annotations",
        required=True,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    df = read_into_df(args.in_)
    df = df.sample(args.num)

    paths = df["unob_seg"].apply(lambda p: Path(p)).to_list()  # type: ignore
    paths.extend(df["unob_seg"].apply(lambda p: Path(p)).to_list())  # type: ignore

    # compress all the files in the sampled df to a tarball
    with tarfile.open(args.sample_dir, mode="w:gz") as tar:
        for path in tqdm(paths):
            tar.add(path, arcname=path.name)

    # write the sampled df
    save_to_json_or_csv(args.sample_annotations, df)

    print("All good!")
