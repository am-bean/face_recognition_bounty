import pandas as pd
import imgaug.augmenters as iaa
import logging
import cv2
import numpy as np
import argparse
from typing import List, Tuple
from pathlib import Path
from functools import lru_cache

# set seed
np.random.seed(42)

logger = logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

CATS = ["skin_tone", "gender", "age"]

# Taken from: https://www.kaggle.com/code/andreagarritano/simple-data-augmentation-with-imgaug/notebook
NEWSEQ = iaa.Sequential(
    [
        iaa.Fliplr(0.5),
        iaa.Crop(percent=(0, 0.1)),
        iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
        iaa.LinearContrast((0.75, 1.5)),
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
    ]
)


@lru_cache(maxsize=100000)
def load_rgb(path: Path, rgb=False) -> np.ndarray:
    img = cv2.imread(str(path))
    if not rgb:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def augment_df(df: pd.DataFrame, seq: iaa.Sequential, data_dir: Path) -> pd.DataFrame:
    """Augment a dataframe of images"""
    logging.info("Loading images")
    imgs = [load_rgb(data_dir / img) for img in df["name"]]
    logging.info("Augmenting images")
    aug_imgs = seq.augment_images(imgs)  # type: ignore
    aug_df = df.copy()
    aug_names = []
    logging.info("Saving images")
    for i, (img, img_name) in enumerate(zip(aug_imgs, aug_df["name"])):  # type: ignore
        aug_path = data_dir / f"AUG{i}_{img_name}"
        cv2.imwrite(str(aug_path), img)
        assert aug_path.exists(), f"Augmented image {aug_path} was not written"
        aug_names.append(aug_path.name)
    aug_df = aug_df.assign(aug_name=aug_names)
    assert aug_df["aug_name"].nunique() == len(aug_df), "Augmented names are not unique"
    logging.info("Augmented dataframe created!")
    return aug_df


def create_intersec_df(labels: pd.DataFrame) -> pd.DataFrame:
    """Create an upsampled dataframe with intersectional labels (NB: only works for the specific ones)"""
    intersec = labels.groupby(CATS)["name"].count()
    upsample_counts = intersec.max() - intersec
    new_df = pd.DataFrame()

    for (skin_tone, gender, age), count in upsample_counts.iteritems():
        intersec_filter = (
            (labels["skin_tone"] == skin_tone)
            & (labels["gender"] == gender)
            & (labels["age"] == age)
        )
        intersec_df = labels[intersec_filter].sample(count, replace=True)
        new_df = pd.concat([new_df, intersec_df])
    return new_df


def main(args: argparse.Namespace) -> None:
    DATA_DIR = Path(args.data_dir)
    assert DATA_DIR.exists(), f"Data directory {DATA_DIR} does not exist"
    labels = pd.read_csv(Path(args.label_path))
    labels = labels[labels["real_face"] == 1].dropna()

    logging.info("Creating intersectional dataframe")
    intersect_df = create_intersec_df(labels)
    aug_df = augment_df(intersect_df, NEWSEQ, DATA_DIR)
    aug_df.to_csv(DATA_DIR / "intersect_augment.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Augment images in a directory",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--label-path", type=str)
    args = parser.parse_args()
    main(args)

