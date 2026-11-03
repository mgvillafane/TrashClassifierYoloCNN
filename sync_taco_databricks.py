"""Upload the six-class dataset to its UC volume folder and deploy the GPU job.

Run with --dry-run to inspect commands. This deploys without starting training.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from taco_yolo import ROOT, dataset_classes, prepared_files_exist, write_json
from taco_label_preview import verify_relabel


def find_cli():
    installed = shutil.which("databricks")
    if installed:
        return installed
    extensions = sorted((Path.home() / ".vscode/extensions").glob("databricks.databricks-*/bin/databricks.exe"))
    if not extensions:
        raise FileNotFoundError("Install the Databricks CLI or pass --cli with its executable path")
    return str(extensions[-1])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="Marcos Gomez Villafane")
    parser.add_argument("--cli", type=Path)
    parser.add_argument("--volume", default="/Volumes/workspace/default/tacotrashdata")
    parser.add_argument("--prepared", type=Path, default=ROOT / "data/yolo/taco_trashnet")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    volume = args.volume.rstrip("/")
    if not volume.startswith("/Volumes/") or len(volume.split("/")) != 5:
        parser.error("--volume must be exactly /Volumes/catalog/schema/volume")
    cli = str(args.cli) if args.cli else find_cli()
    prepared = args.prepared.resolve()
    manifest = json.loads((prepared / "split_manifest.json").read_text())
    annotations = json.loads((ROOT / "data/raw/taco/annotations.json").read_text())
    names, _, mapping = dataset_classes(annotations["categories"], "trashnet")
    if (manifest["configuration"].get("mapping_sha256") != mapping["mapping_sha256"]
            or {int(key): value for key, value in manifest["class_names"].items()} != names):
        raise ValueError("Local prepared labels do not match the current TrashNet mapping; prepare again")
    if not prepared_files_exist(prepared, manifest["images"]):
        raise FileNotFoundError("The local prepared dataset is incomplete; prepare again")
    source_hash = hashlib.sha256((ROOT / "data/raw/taco/annotations.json").read_bytes()).hexdigest()
    if manifest["configuration"]["annotation_sha256"] != source_hash:
        raise ValueError("Prepared labels do not match the raw TACO annotations; prepare again")
    verify_relabel(ROOT / "data/yolo/taco", prepared)
    destination = f"dbfs:{volume}/yolo/taco_trashnet"

    def run(arguments, capture=False):
        command = [cli, *arguments, "-p", args.profile]
        print(subprocess.list2cmdline(command), flush=True)
        if args.dry_run:
            return ""
        result = subprocess.run(command, cwd=ROOT, check=True, text=True,
                                stdout=subprocess.PIPE if capture else None)
        return result.stdout if capture else ""

    bundle_target = ["-t", "dev", "--var", f"taco_root={volume}"]
    run(["bundle", "validate", "--strict", *bundle_target])
    # Images/labels first; the manifest is the completion marker, published last.
    for directory in ("images", "labels"):
        run(["fs", "cp", str(prepared / directory), f"{destination}/{directory}", "--recursive", "--overwrite"])
    run(["fs", "cp", str(prepared / "label_mapping.json"), f"{destination}/label_mapping.json", "--overwrite"])
    yaml = ROOT / "reports/taco_relabel/databricks_dataset.yaml"
    yaml.parent.mkdir(parents=True, exist_ok=True)
    content = f"path: {json.dumps(volume + '/yolo/taco_trashnet')}\n"
    content += "train: images/train\nval: images/val\ntest: images/test\nnames:\n"
    content += "".join(f"  {index}: {json.dumps(name)}\n" for index, name in names.items())
    yaml.write_text(content, encoding="utf-8")
    run(["fs", "cp", str(yaml), f"{destination}/dataset.yaml", "--overwrite"])
    run(["fs", "cp", str(prepared / "split_manifest.json"), f"{destination}/split_manifest.json", "--overwrite"])
    previews = ROOT / "reports/taco_relabel"
    if previews.is_dir():
        run(["fs", "cp", str(previews), f"dbfs:{volume}/reports/taco_relabel", "--recursive", "--overwrite"])
    remote = run(["fs", "cat", f"{destination}/split_manifest.json"], capture=True)
    remote_yaml = run(["fs", "cat", f"{destination}/dataset.yaml"], capture=True)
    if not args.dry_run:
        if json.loads(remote) != manifest or remote_yaml != content:
            raise ValueError("Uploaded dataset metadata did not match the local dataset")
    run(["bundle", "deploy", *bundle_target])
    if not args.dry_run:
        write_json(ROOT / "reports/taco_relabel/cloud_sync.json", {
            "volume_path": volume + "/yolo/taco_trashnet", "image_counts": manifest["image_counts"],
            "mapping_sha256": mapping["mapping_sha256"], "metadata_verified": True,
            "bundle_deployed": True, "training_started": False})
    print("Dry run complete." if args.dry_run else "Upload and deployment complete. Start training with:")
    print(subprocess.list2cmdline([cli, "bundle", "run", "taco_small_gpu", "-t", "dev", "-p", args.profile]))


if __name__ == "__main__":
    main()
