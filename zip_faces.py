"""
Creates a dataset of face / non-face for training face classifier
"""
import argparse
import pandas as pd
import logging
from pathlib import Path

import create_zip_file

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def main(args: argparse.Namespace) -> None:
    # load data
    DATA_DIR = Path(args.data_dir)
    assert DATA_DIR.exists(), f"Path {DATA_DIR} does not exist."
    SAVE_DIR = Path(args.save_dir)
    df = pd.read_csv("train/labels.csv")
    df["real_face"] = pd.to_numeric(df["real_face"], errors="coerce")
    df = df.loc[df["real_face"] <= 1]
    logging.info(f"Loaded {len(df)} images.")
    logging.info(f"{df['real_face'].value_counts()}")

    # downsample faces based on demographics
    logging.info("Downsampling faces")
    facedf = df.loc[df["real_face"] == 1]
    intersect_counts = (
        facedf.groupby(["skin_tone", "gender", "age"])["name"]
        .count()
        .reset_index()
        .rename({"name": "count"}, axis=1)
    )

    # join intersect counts with facedf
    facedf = facedf.merge(intersect_counts, on=["skin_tone", "gender", "age"])
    facedf["weight"] = 1 / facedf["count"]
    downsampled = facedf.sample(n=(df["real_face"] == 0).sum(), weights="weight")

    finaldf = pd.concat(
        [downsampled, df.loc[df["real_face"] == 0]], axis=0
    ).reset_index()[["name", "real_face"]]

    # create dataset
    logging.info("Creating dataset...")
    create_zip_file.save_as_zip(
        finaldf, filename="facenoface.zip", save_dir=SAVE_DIR, data_dir=DATA_DIR
    )
    df.to_csv(SAVE_DIR / "face_labels.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Creates a dataset of face / non-face for training face classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=str, default="train")
    parser.add_argument("--save-dir", type=str, default=".")
    args = parser.parse_args()
    main(args)

