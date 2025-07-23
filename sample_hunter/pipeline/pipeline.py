"""
A huggingface pipeline for transforming mp3 files into a list of 1 or more embeddings
"""

from transformers.pipelines.base import Pipeline


class EmbeddingPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}

        pass

    def preprocess(self, input_, args=2, **preprocess_parameters):
        pass

    def _forward(self, input_tensors, **forward_parameters):
        pass

    def postprocess(self, model_outputs, **postprocess_parameters):
        pass
