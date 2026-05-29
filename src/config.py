from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


@dataclass
class Config:
    data_root: Path = Path("data/raw/Thermal_Dataset_Fall_Non_Fall")
    processed_dir: Path = Path("data/processed")
    seq_len: int = 16
    stride: int = 4
    test_size: float = 0.15
    seed: int = 42
    n_folds: int = 5
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3
    patience: int = 5
    resolutions: List[Tuple[int, int]] = field(default_factory=lambda: [(32, 32), (16, 16)])
    label_map: dict = field(default_factory=lambda: {"ILS": 1, "SSJ": 0})
    class_names: List[str] = field(default_factory=lambda: ["Non-Fall", "Fall"])


CFG = Config()
