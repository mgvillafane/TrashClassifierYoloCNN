"""Unify RealWaste, TrashNet, SortWaste and TACO under one 6-class manifest.

Typical use (from the repo root, with src/ on sys.path):

    from data_preprocessing import DataPaths, find_repo_root, load_config, build_manifest, validate_manifest
    paths = DataPaths(find_repo_root())
    config = load_config(paths.config)
    manifest = build_manifest(paths, config)
    validate_manifest(manifest, config)
"""

from .labels import apply_mapping, build_rules, load_config
from .manifest import (build_manifest, check_raw_files, class_table, exclusion_table, inventory, read_manifest,
                       summarize, validate_manifest)
from .paths import DataPaths, find_repo_root

__all__ = [
    "DataPaths", "apply_mapping", "build_manifest", "build_rules", "check_raw_files", "class_table", "exclusion_table",
    "find_repo_root", "inventory", "load_config", "read_manifest", "summarize", "validate_manifest",
]
