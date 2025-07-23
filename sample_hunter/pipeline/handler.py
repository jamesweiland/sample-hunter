"""
Endpoint handler for hosting the embedding model on HF.

When somebody tries to deploy the model hosted on HF, this is what executes.

Accepts a data dictionary with field `inputs` that are a list of 1 or more
torch.Tensors, and returns the corresponding embeddings.
"""
import yaml

class EndpointHandler:
    def __init__(self, model_dir, **kwargs):
        with yaml.safe_load