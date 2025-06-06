from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import torchaudio


class SongPairs(Dataset):
    annotations: pd.DataFrame
    audio_dir: Path

    def __init__(self, audio_dir: Path, annotations_file: Path):
        """Initialize a SongPairs dataset. Each pair consists of an original audio
        file with it's obfuscated version. audio_dir is the
        path to the directory of audio files."""
        if annotations_file.suffix == ".json":
            self.annotations = pd.read_json(annotations_file)
        else:
            self.annotations = pd.read_csv(annotations_file)

        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # load both unobfuscated and obfuscated waveform
        unobfuscated_path = self._get_unobfuscated_path(idx)
        obfuscated_path = self._get_obfuscated_path(idx)

        unobfuscated_signal, unobfuscated_sr = torchaudio.load(unobfuscated_path)
        obfuscated_signal, obfuscated_sr = torchaudio.load(obfuscated_path)
        assert unobfuscated_sr == obfuscated_sr
        return unobfuscated_signal, obfuscated_signal

    def _get_unobfuscated_path(self, idx):
        return self.annotations.at[idx, "unobfuscated_path"]

    def _get_obfuscated_path(self, idx):
        return self.annotations.at[idx, "obfuscated_path"]
