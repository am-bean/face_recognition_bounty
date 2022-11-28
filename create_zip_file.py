"""
Create a zip file of labeled images for uploading to Google Colab.
"""
import argparse
import pandas as pd
import zipfile
import logging
from tqdm import tqdm
from pathlib import Path

logger = logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main(args: argparse.Namespace) -> None:
    # load data
    DATA_DIR = Path(args.data_dir)
    assert DATA_DIR.exists(), f"Path {DATA_DIR} does not exist."
    SAVE_DIR = Path(args.save_dir)

    df = pd.read_csv("train/labels.csv")
    aug_df = pd.read_csv(DATA_DIR / "intersect_augment.csv")
    new_aug = aug_df[["aug_name", "skin_tone", "gender", "age"]].rename(
        {"aug_name": "name"}, axis=1
    )
    df_labeled = (
        pd.concat(
            [
                df.loc[df["real_face"] == 1, ["name", "skin_tone", "gender", "age"]],
                new_aug,
            ],
            axis=0,
        )
        .dropna()
        .reset_index(drop=True)
    )

    # create zip file
    with zipfile.ZipFile(SAVE_DIR / "labeled_images.zip", "w") as zf:
        for i, row in tqdm(df_labeled.iterrows()):
            zf.write(DATA_DIR / row["name"], row["name"])
    logging.info(f"Zip file saved to {SAVE_DIR / 'labeled_images.zip'}")
    df_labeled.to_csv(SAVE_DIR / "labeled_images.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a zip file of labeled images for uploading to Google Colab.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=str, default="../Downloads/data_bb1/train/")
    parser.add_argument("--save-dir", type=str, default=".")
    args = parser.parse_args()
    main(args)
