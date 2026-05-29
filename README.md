# Thermal Fall Detection from Low Resolution Thermal Images

This project explores fall detection from low resolution thermal image sequences. The main focus is whether motion information can help a model stay reliable when thermal frames are heavily downsampled.

The work compares three PyTorch models at 32x32 and 16x16 resolution:

* CNN+GRU baseline
* compact 3D-CNN baseline
* Motion-Aware CNN+GRU

The best performing model was the Motion-Aware CNN+GRU, especially when comparing the drop in F1 score between 32x32 and 16x16 inputs.

## Why this project matters

Thermal cameras can be useful in privacy sensitive environments such as elderly care because they do not capture the same identity detail as RGB cameras. The challenge is that low resolution thermal frames contain very little appearance detail.

This project asks:

Can motion between frames make fall detection more robust when the input resolution is very low?

## Dataset

Dataset used: Thermal Fall Detection and Activity Dataset from Kaggle.

The raw dataset is not included in this repository. The code expects the dataset to be placed under:

```text
data/raw/Thermal_Dataset_Fall_Non_Fall/
```

Main checks from the completed experiment:

| Check | Result |
|---|---:|
| Parsed PNG frames | 6,748 |
| Unreadable frames | 0 |
| Duplicate frame records | 0 |
| Native frame size | 226x230 |
| Generated clips | 1,655 |

Final split:

| Split | Clips | Share |
|---|---:|---:|
| Train | 1,158 | 70 percent |
| Validation | 248 | 15 percent |
| Test | 249 | 15 percent |

## Results

| Model | Resolution | CV F1 mean | Test F1 | Test accuracy | Parameters | Latency |
|---|---:|---:|---:|---:|---:|---:|
| CNN+GRU | 32x32 | 0.9586 | 0.9718 | 0.9719 | 116,002 | 4.48 ms |
| CNN+GRU | 16x16 | 0.9449 | 0.9438 | 0.9438 | 116,002 | 2.47 ms |
| 3D-CNN | 32x32 | 0.9029 | 0.8913 | 0.8916 | 36,706 | 3.60 ms |
| 3D-CNN | 16x16 | 0.9100 | 0.8704 | 0.8715 | 36,706 | 0.98 ms |
| Motion-Aware CNN+GRU | 32x32 | 0.9622 | 0.9751 | 0.9759 | 453,186 | 8.59 ms |
| Motion-Aware CNN+GRU | 16x16 | 0.9568 | 0.9612 | 0.9598 | 453,186 | 4.34 ms |

The Motion-Aware CNN+GRU achieved the best test F1 at both resolutions. The strongest result is its smaller performance drop when moving from 32x32 to 16x16.

| Model | F1 drop from 32x32 to 16x16 |
|---|---:|
| CNN+GRU | 0.0280 |
| 3D-CNN | 0.0209 |
| Motion-Aware CNN+GRU | 0.0139 |

## Statistical testing

McNemar's test was used to compare paired predictions on the held out test set.

| Comparison | p value | Interpretation |
|---|---:|---|
| Motion-Aware vs CNN+GRU at 32x32 | 1.0000 | not significant |
| Motion-Aware vs CNN+GRU at 16x16 | 0.1336 | not significant |
| CNN+GRU vs 3D-CNN at 32x32 | less than 0.05 | significant |
| CNN+GRU vs 3D-CNN at 16x16 | less than 0.05 | significant |
| CNN+GRU 32x32 vs 16x16 | 0.0233 | significant |
| Motion-Aware 32x32 vs 16x16 | 0.1336 | not significant |

The 16x16 Motion-Aware model performed better numerically than the CNN+GRU baseline, but the difference was not statistically significant in this test. I treat this as a useful trend rather than a confirmed improvement.

## Repository structure

```text
thermal-fall-detection-pytorch/
  src/
    __init__.py
    config.py
    data_pipeline.py
    datasets.py
    models.py
    training.py
    results.py
  docs/
    project_report.md
  notebooks/
    README.md
    thermal_fall_detection_.ipynb
  requirements.txt
  .gitignore
  README.md
```

## How to run

Clone the repository:

```bash
git clone https://github.com/sradowana-ux/thermal-fall-detection-pytorch.git
cd thermal-fall-detection-pytorch
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Place the raw dataset under:

```text
data/raw/Thermal_Dataset_Fall_Non_Fall/
```

Run the data preparation step:

```bash
python -m src.data_pipeline
```

The default paths and experiment settings are in `src/config.py`.

## Limitations

These results should not be treated as deployment ready. The dataset is small and clips are generated from a limited number of source recordings. A stronger evaluation would include subject independent testing, cross dataset validation, and testing on data collected from a different thermal sensor setup.

## Author

Radowana S.
