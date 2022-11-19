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

# Taken from: https://www.kaggle.com/code/andreagarritano/simple-data-augmentation-with-imgaug/notebook
newseq = iaa.Sequential(
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


def create_aug_names(names: pd.Series) -> pd.Series:
    rand_ints = pd.Series(
        np.random.randint(10000000, 99999999, size=names.shape[0]), index=names.index
    ).astype("string")
    return rand_ints + names.str.replace("TRAIN", "_AUG")


def get_imgs_to_sample(series: pd.Series) -> pd.Series:
    counts = series.value_counts()
    return counts.max() - counts


def create_augs(
    df: pd.DataFrame,
    img_to_sample: pd.Series,
    seq: iaa.Sequential,
    data_dir,
    counter: int,
    colname: str = "age",
) -> Tuple[pd.DataFrame, int]:
    new_df = pd.DataFrame()
    this_counter = counter
    for cat, count in img_to_sample.items():
        logging.info(f"Sampling {count} images for {colname} {cat}")
        if count > 0:
            df_sample = df[df[colname] == cat].sample(count, replace=True)
            sample_imgs = [load_rgb(data_dir / name) for name in df_sample["name"]]

            augs = seq.augment_images(sample_imgs)
            df_sample["aug_name"] = None
            aug_paths = []
            for i, aug in enumerate(augs):
                aug_path = data_dir / f"{this_counter}_{df_sample.iloc[i]['name']}"
                aug_paths.append(aug_path.name)
                cv2.imwrite(str(aug_path), aug)
                this_counter += 1
                assert aug_path.exists(), f"Augmented image {aug_path} does not exist"
            df_sample = df_sample.assign(aug_name=aug_paths)
            assert (
                df_sample["aug_name"].isna().sum() == 0
            ), "Augmented image name is NaN"
            new_df = pd.concat([new_df, df_sample])
    return (new_df, this_counter)


def upsample_imgs(
    df: pd.DataFrame, sample_cols: List[str], seq: iaa.Sequential, data_dir: Path
) -> pd.DataFrame:
    aug_df = pd.DataFrame()
    counter = 0
    for col in sample_cols:
        logging.info(f"Upsampling {col}")
        counts = df[col].value_counts()
        img_to_sample = counts.max() - counts
        new_aug, new_count = create_augs(
            df, img_to_sample, seq=seq, colname=col, data_dir=data_dir, counter=counter,
        )

        counter += new_count
        aug_df = pd.concat([aug_df, new_aug,])
    return aug_df


def main(args: argparse.Namespace) -> None:
    DATA_DIR = Path(args.data_dir)
    CATS = ["skin_tone", "gender", "age"]
    labels = pd.read_csv(Path(args.label_path))
    labels = labels[labels["real_face"] == 1].dropna()
    aug_df = upsample_imgs(labels, sample_cols=CATS, seq=newseq, data_dir=DATA_DIR)
    aug_df.to_csv(DATA_DIR / "aug_labels2.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Augment images in a directory",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--label-path", type=str)
    args = parser.parse_args()
    main(args)
