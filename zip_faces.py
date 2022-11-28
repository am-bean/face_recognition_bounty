"""
Creates a dataset of face / non-face for training face classifier
"""
import argparse
import pandas as pd
import logging
import zipfile
from pathlib import Path
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def save_imgs_with_labels(
    df: pd.DataFrame,
    filename: str,
    data_dir: Path,
    save_dir: Path,
    label_col: str = "real_face",
) -> None:
    """Save img files to zip with label directory structure"""
    with zipfile.ZipFile(save_dir / filename, "w") as zf:
        for _, row in tqdm(df.iterrows()):
            zf.write(data_dir / row["name"], f"{row[label_col]}/{row['name']}")


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

    finaldf = (
        pd.concat([downsampled, df.loc[df["real_face"] == 0]], axis=0).reset_index()[
            ["name", "real_face"]
        ]
        # replace "1" with "face" and "0" with "nonface"
        .replace({"real_face": {1: "face", 0: "nonface"}})
    )

    # create dataset
    logging.info("Creating dataset...")
    save_imgs_with_labels(
        finaldf,
        filename="facenoface.zip",
        save_dir=SAVE_DIR,
        data_dir=DATA_DIR,
        label_col="real_face",
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

