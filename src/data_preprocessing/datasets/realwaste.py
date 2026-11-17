"""RealWaste: 4,752 whole images in 9 class folders."""

from __future__ import annotations

from ..paths import DataPaths
from . import whole_image_rows


def rows(paths: DataPaths) -> list[dict]:
    return whole_image_rows("realwaste", paths.realwaste, paths.rel)
