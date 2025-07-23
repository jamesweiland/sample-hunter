"""
Endpoint handler for hosting the embedding model on HF.

When somebody tries to deploy the model hosted on HF, this is what executes.

Accepts a data dictionary with field `inputs` that are a list of 1 or more
torch.Tensors, and returns the corresponding embeddings.
"""

import torch
from pathlib import Path
from sample_hunter._util import load_model


class EndpointHandler:
    def __init__(self, model_dir, **kwargs):
        self.model = load_model(Path("./7-18-2025-1.pth"))
        self.model.eval()

    def __call__(self, data):
        with torch.no_grad():
            return self.model(data)
