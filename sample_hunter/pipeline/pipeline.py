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
from .predict import infer_with_offset


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
            self.index = faiss.read_index(index)
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
            # model_outputs = [self.model(offset) for offset in input_tensors["tensors"]]
            model_outputs = self.model(input_tensors["tensors"])

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

        config = PostprocessConfig().merge_kwargs(**postprocess_parameters)

        D, I = index.search(embeddings, config.top_k)  # type: ignore
        print(I.shape)
        for neighbors in I:
            for neighbor in neighbors:
                predicted_song_id = metadata[metadata["snippet_id"] == neighbor][
                    "song_id"
                ].to_list()[0]
                if predicted_song_id == model_outputs["song_id"]:
                    return {"song_id": -1, "score": -1, "was_candidate": True}

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
    metadata = pd.read_csv("./_data/dev_metadata.csv")
    pipe = FunkyFinderPipeline(
        "./_data/7-29-2025b-1.pth", "./_data/dev.faiss", metadata
    )
    dataset = cast(wds.WebDataset, load_webdataset("samplr/songs", "validation"))

    total_correct = 0
    len_dataset = 0
    for ex in dataset:
        title = ex["json"]["title"]
        print(title)
        song_id = metadata[metadata["song_title"] == title]["song_id"].iloc[0]

        result = cast(dict, pipe({"audio": ex["b.mp3"], "song_id": song_id}))

        print(result)
        if result["song_id"] == song_id:
            print("it worked")
        else:
            print("it didn't work")
            # print(
            #     f"it thought {title} was {metadata[metadata["song_id"] == result["song_id"]]["song_title"].iloc[0]}"
            # )
        if result["was_candidate"]:
            print("However, the correct song was a candidate")
            total_correct += 1
        print(f"gt id: {song_id}")
        print(f"predicted id: {result["song_id"]}")
        len_dataset += 1

    print(f"correctly identified candidates: {total_correct / len_dataset}")
