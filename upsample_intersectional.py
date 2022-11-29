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

# Taken from Andrew's notebook
NEWSEQ = iaa.Sequential(
    [
        iaa.Fliplr(0.5),  # horizontal flips
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
        # Strengthen or weaken the contrast in each image.
        iaa.LinearContrast((0.75, 1.5)),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.0),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-15, 15),
            shear=(-8, 8),
        ),
    ],
    random_order=True,
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


def upsample_intersec(labels: pd.DataFrame, max_copy: int = 20) -> pd.DataFrame:
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
        max_copies = (
            intersec_filter.sum() * max_copy
        )  # find the number of copies we can make on average
        temp_df = labels[intersec_filter]
        if max_copies < count:
            intersec_df = temp_df.loc[np.repeat(temp_df.index.values, max_copy), :]
        else:
            intersec_df = temp_df.sample(count, replace=True)
        new_df = pd.concat([new_df, intersec_df])
    return new_df


def downsample_intersec(labels: pd.DataFrame) -> pd.DataFrame:
    """Downsample a dataframe with intersectional labels (NB: only works for the specific ones)"""
    intersec = labels.groupby(CATS)["name"].count()
    min_count = intersec.min()
    new_df = pd.DataFrame()
    for (skin_tone, gender, age), _ in intersec.iteritems():
        intersec_filter = (
            (labels["skin_tone"] == skin_tone)
            & (labels["gender"] == gender)
            & (labels["age"] == age)
        )
        intersec_df = labels[intersec_filter].sample(min_count, replace=False)
        new_df = pd.concat([new_df, intersec_df])
    return new_df


def create_intersec_df(labels: pd.DataFrame, max_copy: int = 20) -> pd.DataFrame:
    """Create an upsampled dataframe with intersectional labels (NB: only works for the specific ones)"""
    upsampled_intersec = upsample_intersec(labels, max_copy=max_copy)
    combined = pd.concat([labels, upsampled_intersec])
    downsampled_intersec = downsample_intersec(combined)
    return downsampled_intersec


def main(args: argparse.Namespace) -> None:
    DATA_DIR = Path(args.data_dir)
    MAX_COPY = args.max_copy
    assert DATA_DIR.exists(), f"Data directory {DATA_DIR} does not exist"
    labels = pd.read_csv(Path(args.label_path))
    labels = labels[labels["real_face"] == 1].dropna()

    logging.info("Creating intersectional dataframe")
    intersect_df = create_intersec_df(labels, max_copy=MAX_COPY)
    logging.info(f"intersec dataframe has {len(intersect_df)} rows")
    aug_df = augment_df(intersect_df, NEWSEQ, DATA_DIR)
    aug_df.to_csv(DATA_DIR / "intersect_augment.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Augment images in a directory",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--label-path", type=str)
    parser.add_argument(
        "--max-copy",
        help="The maximum number of copies per image",
        type=int,
        default=50,
    )
    args = parser.parse_args()
    main(args)

