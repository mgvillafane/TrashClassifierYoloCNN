"""Fetch 10 TACO images and load them into a pandas DataFrame."""

import pandas as pd
from PIL import Image

from dataset_tools import ROOT, fetch_taco


def main() -> pd.DataFrame:
    # Downloads are cached, so subsequent runs reuse the local images.
    rows = fetch_taco(ROOT / "data", limit=10, seed=42)
    if len(rows) != 10:
        raise RuntimeError(f"Expected 10 TACO images, but only {len(rows)} were available.")

    taco_df = pd.DataFrame(rows)
    taco_df["labels"] = taco_df["annotations"].apply(
        lambda annotations: [annotation["label"] for annotation in annotations]
    )

    # Store the actual RGB images alongside their paths and bounding boxes.
    images = []
    for path in taco_df["path"]:
        with Image.open(ROOT / path) as image:
            images.append(image.convert("RGB"))
    taco_df["image"] = images

    #print(taco_df[["image_id", "path", "width", "height", "labels"]].to_string(index=False))
    print(taco_df["path"])
    print(taco_df["labels"])
    return taco_df


if __name__ == "__main__":
    taco_df = main()
