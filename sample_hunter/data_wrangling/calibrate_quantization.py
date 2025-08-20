import torch
from torch.quantization import MinMaxObserver
import webdataset as wds
import random
import argparse

from pathlib import Path

from sample_hunter.pipeline.train import train_collate_fn, flatten
from sample_hunter.pipeline.data_loading import load_tensor_from_mp3_bytes
from sample_hunter.pipeline.transformations.obfuscator import Obfuscator
from sample_hunter.pipeline.transformations.preprocessor import Preprocessor
from sample_hunter.config import PreprocessConfig, ObfuscatorConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset", help="path to the dataset", type=Path)

    parser.add_argument("n", help="the sample size to use", type=int)

    parser.add_argument("--config", help="the path to the config", type=Path)

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    train_dir = Path(args.dataset / "train")
    tars = list(train_dir.glob("*.tar"))
    test_dir = Path(args.dataset / "test")
    tars.extend(list(test_dir.glob("*.tar")))
    tars = [str(tar) for tar in tars]

    dataset = wds.WebDataset(tars, shardshuffle=len(tars)).decode().shuffle(200)

    if args.config:
        preprocess_config = PreprocessConfig.from_yaml(args.config)
        obfuscator_config = ObfuscatorConfig.from_yaml(args.config)
    else:
        preprocess_config = PreprocessConfig()
        obfuscator_config = ObfuscatorConfig()

    observer = MinMaxObserver()

    n = 0
    with torch.no_grad():
        with Preprocessor(
            preprocess_config, obfuscator=Obfuscator(obfuscator_config)
        ) as preprocessor:

            def map_fn(example):
                audio, sr = load_tensor_from_mp3_bytes(example["mp3"])
                anchor, positive = preprocessor(audio, sample_rate=sr, train=True)

                example["anchor"] = anchor
                example["positive"] = positive

            dataset = dataset.map(map_fn)
            loader = wds.WebLoader(dataset).batched(30, collation_fn=train_collate_fn)

            for anchor, positive, keys in loader:
                observer(anchor)
                observer(positive)

                n += anchor.shape[0]
                if n > args.n:
                    break

    scale, zero_point = observer.calculate_qparams()
    print("Calibration complete")
    print(f"Scale: {scale}")
    print(f"Zero point: {zero_point}")
