from abc import ABC, abstractmethod
import inspect
from pathlib import Path
import re
from typing import List
from pydantic import BaseModel


class Song(ABC, BaseModel):
    """Interface for a song"""

    path_: Path

    @property
    @abstractmethod
    def title(self) -> str:
        """The title of the song."""
        pass

    @property
    @abstractmethod
    def song_artist(self) -> str:
        """The artist attributed to the song"""
        pass

    @property
    @abstractmethod
    def path(self) -> Path:
        """The path to the mp3 file"""
        pass

    @property
    def soty(self) -> bool:
        """Whether the song is part of a Samples of the Year sample set."""
        return Song.is_soty(self.path)

    @staticmethod
    def is_soty(mp3: Path) -> bool:
        return "samples of the year" in mp3.parent.name.lower()

    @classmethod
    def properties(cls):
        """Get the names of the properties of the class"""
        props = [
            name
            for name, value in inspect.getmembers(cls)
            if isinstance(value, property)
        ]

        # need to filter out pydantic properties
        return [p for p in props if not (p.startswith("__") or p.startswith("model_"))]

    def model_dump_properties(self) -> dict:
        """Similar to pydantic's model_dump, but for the properties instead of fields"""
        return {prop: getattr(self, prop) for prop in self.__class__.properties()}


class SampledSong(Song):
    """Represents sampled songs"""

    @property
    def song_artist(self) -> str:
        stem = self.path_.stem
        pattern = r"^(?:\d+\s+)?([^\-]+)\s+-\s+"
        match = re.match(pattern, stem)
        if match:
            return match.group(1).strip()
        return ""

    @property
    def title(self) -> str:
        stem = self.path_.stem
        s_clean = re.sub(r"\s*\[.*\]$", "", stem)
        match = re.match(r"^(?:\d+\s+)?.*? - (.*)", s_clean)
        if match:
            return match.group(1).strip()
        return ""

    @property
    def path(self) -> Path:
        return self.path_


class SamplingSong(Song):
    """Represents the song that sampled another song"""

    @property
    def song_artist(self) -> str:
        stem = self.path_.stem
        pattern = r"\[([^\-\[\]]+)\s*-\s*\'"
        match = re.search(pattern, stem)
        if match:
            return match.group(1).strip()
        return self.album_artist

    @property
    def title(self) -> str:
        stem = self.path_.stem
        pattern = r"\[[^]]*'([^']+)'\]$"
        match = re.search(pattern, stem)
        if match:
            return match.group(1).strip()
        return ""

    @property
    def album_title(self) -> str:
        if self.soty:
            return ""
        parent = self.path_.parent.name
        pattern = r"\)\s+\S+\s+-\s+([^\[]+)\s*\["
        match = re.search(pattern, parent)
        if match:
            return match.group(1).strip()
        return ""

    @property
    def album_artist(self) -> str:
        if self.soty:
            return ""
        parent = self.path_.parent.name
        pattern = r"^([^(]+)\s*\("
        match = re.search(pattern, parent)
        if match:
            return match.group(1).strip()
        return ""

    @property
    def year_released(self) -> str:
        parent = self.path_.parent.name
        pattern = r"\((.*?)\)"
        match = re.search(pattern, parent)
        if match:
            return match.group(1).strip()
        return ""

    @property
    def path(self) -> Path:
        return self.path_
