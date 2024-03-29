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

CATS = ["skin_tone", "age", "gender"]


def save_as_zip(
    df: pd.DataFrame, filename: str, data_dir: Path, save_dir: Path
) -> None:
    with zipfile.ZipFile(save_dir / filename, "w") as zf:
        for _, row in tqdm(df.iterrows()):
            zf.write(data_dir / row["name"], row["name"])


def downsample_intersec(labels: pd.DataFrame) -> pd.DataFrame:
    """Downsample a dataframe with intersectional labels (NB: only works for the specific ones)"""
    intersec = labels.groupby(CATS)["name"].count()
    min_count = intersec.min()
    new_df = pd.DataFrame()
    for (skin_tone, age, gender), _ in intersec.iteritems():
        intersec_filter = (
            (labels["skin_tone"] == skin_tone)
            & (labels["gender"] == gender)
            & (labels["age"] == age)
        )
        intersec_df = labels[intersec_filter].sample(min_count, replace=False)
        new_df = pd.concat([new_df, intersec_df])
    return new_df


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

    # downsample faces based on demographics
    print("Downsampling faces")
    downsampled = downsample_intersec(df_labeled)
    print(
        "Biggest group (post-downsample):",
        downsampled.groupby(CATS)["name"].count().max(),
    )

    # create zip file
    save_as_zip(downsampled, "labeled_images.zip", data_dir=DATA_DIR, save_dir=SAVE_DIR)
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
