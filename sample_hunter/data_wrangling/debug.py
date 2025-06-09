import pandas as pd
from pathlib import Path
import numpy as np

annotations = pd.read_csv("_data/annotations.csv")
samples = pd.read_csv("_data/samples.csv")

print(len(annotations))
print(len(samples))

annotations["stem"] = annotations["unob_full"].apply(lambda p: Path(p).stem)
stem_to_path = dict(zip(annotations["stem"], annotations["unob_full"]))

samples["stem"] = samples["path"].apply(lambda p: Path(p).stem)
samples["path"] = samples["stem"].map(stem_to_path)

samples.to_csv("_data/samples.csv", index=False)

# for col in df.select_dtypes(include=["object"]).columns:
#     df[col] = df[col].str.replace("new_annotations.csv", "train", regex=False)

# df.to_csv("_data/annotations.csv", index=False)
