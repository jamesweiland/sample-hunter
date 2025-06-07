import pandas as pd
from pathlib import Path
import numpy as np

df = pd.read_csv("_data/annotations.csv")

for idx, row in df.iterrows():
    row = row.astype("str")

    obfuscate = Path(row["obfuscate"])
    unobfuscate = Path(row["unobfuscate"])
    df.at[idx, "obfuscate"] = Path(
        Path("/home/james/code/sample-hunter/_data/audio-dir") / obfuscate.name
    )
    df.at[idx, "unobfuscate"] = Path(
        Path("/home/james/code/sample-hunter/_data/audio-dir") / unobfuscate.name
    )

df.to_csv("_data/annotations.csv", index=False)
