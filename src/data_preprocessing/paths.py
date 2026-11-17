"""Locations of the raw datasets and preprocessing outputs.

Manifest paths are logical ('data/raw/...', the local repo layout) so a manifest means the same
thing on every machine. resolve() maps them to wherever the data actually is, e.g. Databricks
volumes:

    DataPaths(repo_root, raw_dir="/Volumes/workspace/default/waste_data",
              taco_dir="/Volumes/workspace/default/tacotrashdata/taco")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

RAW = "data/raw"            # logical folder holding realwaste/, trashnet/, sortwaste/
TACO = "data/raw/taco/data"  # logical folder holding TACO's annotations.json and batch_*/


def find_repo_root(start: Path | None = None) -> Path:
    start = Path.cwd() if start is None else Path(start)
    for candidate in [start.resolve(), *start.resolve().parents]:
        if (candidate / "configs" / "label_mapping.json").is_file() and (candidate / "src").is_dir():
            return candidate
    raise FileNotFoundError("Could not find repository root containing configs/label_mapping.json and src/")


@dataclass(frozen=True)
class DataPaths:
    root: Path                       # repository (configs/ and the default data/ folders)
    raw_dir: Path | None = None      # default <root>/data/raw
    taco_dir: Path | None = None     # default <raw_dir>/taco/data
    manifest_dir: Path | None = None  # default <root>/data/manifests

    @property
    def raw(self) -> Path:
        return Path(self.raw_dir) if self.raw_dir else self.root / RAW

    @property
    def realwaste(self) -> Path:
        return self.raw / "realwaste/realwaste-main/RealWaste"

    @property
    def trashnet(self) -> Path:
        return self.raw / "trashnet/dataset-resized"

    @property
    def sortwaste_labels(self) -> Path:
        # The YOLO copy ships labels for every split but images only for test.
        return self.raw / "sortwaste/dataset/splited_all_dataset_yolo"

    @property
    def sortwaste_images(self) -> Path:
        # The COCO copy holds all 5,261 images with the same stems and splits.
        return self.raw / "sortwaste/dataset/splited_all_dataset"

    @property
    def taco(self) -> Path:
        return Path(self.taco_dir) if self.taco_dir else self.raw / "taco/data"

    @property
    def config(self) -> Path:
        return self.root / "configs/label_mapping.json"

    @property
    def manifests(self) -> Path:
        return Path(self.manifest_dir) if self.manifest_dir else self.root / "data/manifests"

    def _mounts(self) -> list[tuple[Path, str]]:
        return [(self.taco, TACO), (self.raw, RAW)]  # TACO first: locally it sits inside raw

    def rel(self, path: Path) -> str:
        """Logical POSIX path of a real file, so manifests are portable between machines."""
        path = Path(path)
        for actual, logical in self._mounts():
            if path.is_relative_to(actual):
                return f"{logical}/{path.relative_to(actual).as_posix()}"
        return path.relative_to(self.root).as_posix()

    def resolve(self, logical: str) -> Path:
        """Real location of a manifest path."""
        for actual, prefix in self._mounts():
            if logical.startswith(prefix + "/"):
                return actual / logical[len(prefix) + 1:]
        return self.root / logical

    def missing(self) -> list[Path]:
        required = [self.realwaste, self.trashnet, self.sortwaste_labels, self.sortwaste_images,
                    self.taco / "annotations.json", self.config]
        return [p for p in required if not p.exists()]
