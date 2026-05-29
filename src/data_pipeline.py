import re
from pathlib import Path

import cv2
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import CFG


FILENAME_PATTERN = re.compile(r"(ILS|SSJ).*?(\d+).*?(\d+)\.png$", re.IGNORECASE)


def parse_frame_path(path: Path):
    match = FILENAME_PATTERN.search(path.name)
    if match is None:
        return None

    class_code, sequence_id, frame_id = match.groups()
    class_code = class_code.upper()

    return {
        "path": str(path),
        "class_code": class_code,
        "sequence": int(sequence_id),
        "frame": int(frame_id),
        "label": CFG.label_map[class_code],
    }


def build_frame_index(data_root: Path = CFG.data_root) -> pd.DataFrame:
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset not found: {data_root}")

    rows = []
    for path in sorted(data_root.rglob("*.png")):
        item = parse_frame_path(path)
        if item is not None:
            rows.append(item)

    if not rows:
        raise RuntimeError("No valid PNG frames were parsed from the dataset.")

    frame_df = pd.DataFrame(rows)
    return frame_df.sort_values(["class_code", "sequence", "frame"]).reset_index(drop=True)


def run_quality_checks(frame_df: pd.DataFrame) -> dict:
    duplicate_count = frame_df.duplicated(["class_code", "sequence", "frame"]).sum()

    unreadable_count = 0
    image_sizes = []
    low_contrast_count = 0

    for row in frame_df.itertuples(index=False):
        image = cv2.imread(row.path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            unreadable_count += 1
            continue

        image_sizes.append(image.shape)
        if float(image.std()) < 1.0:
            low_contrast_count += 1

    return {
        "frames": len(frame_df),
        "duplicates": int(duplicate_count),
        "unreadable": unreadable_count,
        "image_sizes": pd.Series(image_sizes).value_counts().to_dict(),
        "low_contrast": low_contrast_count,
    }


def make_segments(frame_df: pd.DataFrame, seq_len: int = CFG.seq_len, stride: int = CFG.stride) -> pd.DataFrame:
    segments = []

    for (class_code, sequence_id), group in frame_df.groupby(["class_code", "sequence"]):
        group = group.sort_values("frame").reset_index(drop=True)
        label = int(group["label"].iloc[0])

        for start in range(0, len(group) - seq_len + 1, stride):
            clip = group.iloc[start:start + seq_len]
            segments.append({
                "class_code": class_code,
                "sequence": int(sequence_id),
                "start_frame": int(clip["frame"].iloc[0]),
                "end_frame": int(clip["frame"].iloc[-1]),
                "label": label,
                "paths": clip["path"].tolist(),
            })

    return pd.DataFrame(segments)


def create_splits(segment_df: pd.DataFrame, test_size: float = CFG.test_size, seed: int = CFG.seed):
    train_val_df, test_df = train_test_split(
        segment_df,
        test_size=test_size,
        stratify=segment_df["label"],
        random_state=seed,
    )

    validation_size = test_size / (1.0 - test_size)
    train_df, validation_df = train_test_split(
        train_val_df,
        test_size=validation_size,
        stratify=train_val_df["label"],
        random_state=seed,
    )

    return train_df.reset_index(drop=True), validation_df.reset_index(drop=True), test_df.reset_index(drop=True)


def save_clean_splits(output_dir: Path = CFG.processed_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_df = build_frame_index()
    checks = run_quality_checks(frame_df)
    segment_df = make_segments(frame_df)
    train_df, validation_df, test_df = create_splits(segment_df)

    frame_df.to_pickle(output_dir / "frames_clean.pkl")
    segment_df.to_pickle(output_dir / "segments_clean.pkl")
    train_df.to_pickle(output_dir / "train_clean.pkl")
    validation_df.to_pickle(output_dir / "validation_clean.pkl")
    test_df.to_pickle(output_dir / "test_clean.pkl")

    return checks, segment_df, train_df, validation_df, test_df


if __name__ == "__main__":
    checks, segment_df, train_df, validation_df, test_df = save_clean_splits()
    print(checks)
    print(f"Segments: {len(segment_df)}")
    print(f"Train: {len(train_df)}, validation: {len(validation_df)}, test: {len(test_df)}")
