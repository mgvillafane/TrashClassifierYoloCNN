"""TrashNet: 2,527 whole images in 6 class folders (dataset-resized; __MACOSX excluded)."""

from __future__ import annotations

from ..paths import DataPaths
from . import whole_image_rows


def rows(paths: DataPaths) -> list[dict]:
    return whole_image_rows("trashnet", paths.trashnet, paths.rel)
