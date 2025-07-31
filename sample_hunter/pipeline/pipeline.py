"""
A huggingface pipeline for transforming mp3 files into a list of 1 or more embeddings
"""

from typing import TypedDict, Union, cast
import torch
import torchaudio
import faiss
import pandas as pd
import webdataset as wds
import numpy as np

from dataclasses import fields
from transformers.pipelines.base import Pipeline
from transformers.utils.generic import ModelOutput
from pathlib import Path

from sample_hunter._util import load_model
from sample_hunter.config import (
    FunkyFinderPipelineConfig,
    PreprocessConfig,
    PostprocessConfig,
)

from .data_loading import load_tensor_from_bytes, load_webdataset
from .encoder_net import EncoderNet
from .transformations.preprocessor import Preprocessor


class HFAudio(TypedDict):
    """A class representing the HF Audio object from datasets"""

    array: np.ndarray
    sampling_rate: int


class FunkyFinderPipeline(Pipeline):

    class PipelineInputDict(TypedDict):
        """The type of input that should be passed to pipeline"""

        audio: Union[bytes, Path, str, HFAudio]
        song_id: int

    class PipelineResult(TypedDict):
        """The output of the pipeline"""

        song_id: int
        score: float
        was_candidate: bool

    def __init__(
        self,
        model: EncoderNet | Path | str,
        index: faiss.Index | Path | str,
        metadata: pd.DataFrame | Path | str,
        **kwargs,
    ):
        """
        Set up the configuration and initialize the pipeline.

        model: either an EncoderNet or a path to a .pth file that can be loaded into an EncoderNet instance.

        index: either a faiss.Index or a path to a .faiss file that can be loaded into an index

        metadata: either a pd.DataFrame or a path to a .csv file that can be loaded into a df
        """
        if isinstance(model, Path | str):
            self.model = load_model(model)
        else:
            self.model = model
        self.model.eval()
        if isinstance(index, Path | str):
            # cast it to a string because faiss doesn't like Paths
            self.index = faiss.read_index(str(index))
        else:
            self.index = index
        if isinstance(metadata, Path | str):
            self.metadata = pd.read_csv(metadata)
        else:
            self.metadata = metadata

        # merge user kwargs on top of default config
        self.config = FunkyFinderPipelineConfig(**kwargs).to_dict()

        # set up the preprocessor now
        self.preprocessor = Preprocessor(**self.config, obfuscator=None)

        super().__init__(self.model, **kwargs)  # type: ignore

    def _sanitize_parameters(self, **kwargs):
        """Partition the config into preprocess, forward, and postprocess kwargs"""

        # merge call-time kwargs with pipeline config
        config = {**self.config, **kwargs}
        preprocess_keys = {f.name for f in fields(PreprocessConfig)}
        preprocess_kwargs = {}
        for key in preprocess_keys:
            if config[key] != self.config[key]:
                preprocess_kwargs.update(config[key])

        postprocess_keys = {f.name for f in fields(PostprocessConfig)}
        postprocess_kwargs = {}
        for key in postprocess_keys:
            if config[key] != self.config[key]:
                postprocess_kwargs.update(config[key])

        return preprocess_kwargs, {}, postprocess_kwargs

    def preprocess(
        self,
        input_: PipelineInputDict,
        **preprocess_parameters,
    ):
        """
        Preprocess the input that is passed when the pipeline is called.

        Expects `input` to be a Dict with two fields: "audio" and "song_id".
        "audio" can be given as mp3 bytes, a path to an mp3 file, or a huggingface
        Audio object (which is a dictionary with fields "array" mapping to a np.ndarray and "sampling_rate"
        mapping to an int). "song_id" is an int that represents the song's song id in the
        vector store.

        Preprocessing is done on the audio data to:

        * Mix the channels to mono, resample to the config sample rate

        * Split into 1-second windows with a 0.5-second overlay (by default) and
        transform each of these windows into mel spectrograms
        """
        if isinstance(input_["audio"], bytes):
            audio, sr = load_tensor_from_bytes(input_["audio"])
        elif isinstance(input_["audio"], Path | str):
            audio, sr = torchaudio.load(input_["audio"])
        elif (
            isinstance(input_["audio"], dict)
            and "array" in input_["audio"].keys()
            and "sampling_rate" in input_["audio"].keys()
        ):
            # this is a transformers.Audio object from huggingface
            # the waveform is stored in array as a np.ndarray
            # and the sample rate is stored in the sampling_rate field
            audio = torch.tensor(input_["audio"]["array"])
            sr = input_["audio"]["sampling_rate"]
        else:
            raise ValueError(
                "Unsupported input type.\n"
                "Supported input types: bytes, Path, str\n"
                f"Provided input type: {type(input_)}"
            )

        input_tensors = self.preprocessor(
            audio,
            sample_rate=sr,
            train=False,
            target_length=None,
            **preprocess_parameters,
        )
        return {"tensors": input_tensors, "song_id": input_["song_id"]}

    def _forward(self, input_tensors, **forward_parameters):
        """
        Expects input_tensors to be a dictionary with two keys: "tensors" mapping to a list of tensors, and "song_id" mapping to an int
        """
        with torch.no_grad():
            model_outputs = self.model(input_tensors["tensors"])
            print(f'shape of inputs: {input_tensors["tensors"].shape}')
            print(f"shape of outputs: {model_outputs.shape}")

        return cast(
            ModelOutput,
            {"tensors": model_outputs, "song_id": input_tensors["song_id"]},
        )

    def postprocess(self, model_outputs, **postprocess_parameters) -> PipelineResult:
        index = cast(faiss.Index, postprocess_parameters.get("index") or self.index)
        metadata = cast(
            pd.DataFrame, postprocess_parameters.get("metadata") or self.metadata
        )
        embeddings = model_outputs["tensors"]
        ground_song_id = model_outputs["song_id"]

        config = PostprocessConfig().merge_kwargs(**postprocess_parameters)

        D, I = index.search(embeddings, config.top_k)  # type: ignore
        print(I.shape)
        for i in range(I.shape[0]):
            neighbors = I[i]
            print(f"neighbors shape: {neighbors.shape}")
            # get all unique song ids in neighbors
            unique_song_ids = metadata[metadata["snippet_id"].isin(neighbors)][
                "song_id"
            ].unique()
            if ground_song_id in unique_song_ids:
                return {"song_id": -1, "score": -1, "was_candidate": True}
            else:
                print(f"predicted song ids: {unique_song_ids}")
                predicted_songs = metadata[metadata["song_id"].isin(unique_song_ids)][
                    "song_title"
                ].unique()
                print(f"predicted songs: {predicted_songs}")

        return {"song_id": -1, "score": -1, "was_candidate": False}

        # was_candidate = False
        # for embedding in embeddings:
        #     D, I = index.search(embedding, config.top_k)  # type: ignore
        #     for neighbors in I:
        #         for neighbor in neighbors:
        #             predicted_song_id = metadata[metadata["snippet_id"] == neighbor][
        #                 "song_id"
        #             ].tolist()[0]

        #             if predicted_song_id == model_outputs["song_id"]:
        #                 was_candidate = True
        #                 break
        #         if was_candidate:
        #             break

        # candidate, score = infer_with_offset(
        #     embeddings=embeddings,
        #     index=index,
        #     metadata=metadata,
        #     gt=None,
        #     config=config,
        # )

        # return {"song_id": candidate, "score": score, "was_candidate": was_candidate}


if __name__ == "__main__":
    # test the pipeline
    import argparse
    from sample_hunter.config import DEFAULT_REPO_ID

    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path, help="the path to the model to use")
    parser.add_argument("index", type=Path, help="the path to the index to use")
    parser.add_argument("metadata", type=Path, help="the path to the metadata")
    parser.add_argument(
        "--repo-id",
        type=str,
        help="The HF repo id to test with",
        default=DEFAULT_REPO_ID,
    )

    args = parser.parse_args()

    metadata = pd.read_csv(args.metadata)
    pipe = FunkyFinderPipeline(args.model, args.index, metadata)
    dataset = cast(wds.WebDataset, load_webdataset(args.repo_id, "validation"))

    total_correct = 0
    len_dataset = 0
    for ex in dataset:
        title = ex["json"]["title"]
        print(title)
        ground_song_id = ex["json"]["ground_song_id"]

        from sample_hunter._util import play_tensor_audio
        from mutagen.mp3 import MP3

        ground, ground_sr = load_tensor_from_bytes(ex["a.mp3"])
        positive, positive_sr = load_tensor_from_bytes(ex["b.mp3"])

        play_tensor_audio(ground, message="playing ground...", sample_rate=ground_sr)
        play_tensor_audio(
            positive, message="playing positive...", sample_rate=positive_sr
        )

        seconds_long = positive.shape[-1] / positive_sr

        print(f"positive is {seconds_long:.2f} seconds long")
        print(f"we should have {((seconds_long - 1)/0.5) + 1:.2f} windows")

        result = cast(dict, pipe({"audio": ex["b.mp3"], "song_id": ground_song_id}))

        print(result)
        if result["song_id"] == ground_song_id:
            print("it worked")
        else:
            print("it didn't work")
            # print(
            #     f"it thought {title} was {metadata[metadata["song_id"] == result["song_id"]]["song_title"].iloc[0]}"
            # )
        if result["was_candidate"]:
            print("However, the correct song was a candidate")
            total_correct += 1
        print(f"gt id: {ground_song_id}")
        print(f"predicted id: {result["song_id"]}")
        len_dataset += 1
        exit(0)

    print(f"correctly identified candidates: {total_correct / len_dataset:.2%}")
