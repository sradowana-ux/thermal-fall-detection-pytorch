# Project Report: Thermal Fall Detection from Low Resolution Thermal Images

## Abstract

This project evaluates fall detection from low resolution thermal image sequences. The work compares a CNN+GRU baseline, a compact 3D-CNN baseline, and a Motion-Aware CNN+GRU model at 32x32 and 16x16 resolution. The Motion-Aware CNN+GRU achieved the highest test F1 at both resolutions and showed the smallest performance drop when moving from 32x32 to 16x16 input. The results suggest that motion cues may help preserve useful temporal information when thermal appearance detail is reduced.

## Problem statement

Thermal cameras are useful for privacy sensitive monitoring because they avoid the visual identity detail captured by RGB cameras. This is important in settings such as elderly care, where fall detection can support safety but must also respect privacy.

The main challenge is that thermal frames can be low resolution and contain limited appearance information. This project studies whether adding frame difference information can improve robustness under heavy downsampling.

## Data preparation

The dataset was parsed from PNG thermal frames. Filename information was converted into frame level metadata, then ordered frames were grouped into fixed length clips.

Main preprocessing settings:

| Setting | Value |
|---|---:|
| Clip length | 16 frames |
| Stride | 4 |
| Resolutions | 32x32 and 16x16 |
| Random seed | 42 |
| Cross validation | 5 folds |

Main data checks:

| Check | Result |
|---|---:|
| Parsed PNG frames | 6,748 |
| Unreadable frames | 0 |
| Duplicate frame records | 0 |
| Generated clips | 1,655 |

## Models

### CNN+GRU baseline

This model extracts spatial features from each frame using a small CNN. A GRU then models the temporal sequence.

### Compact 3D-CNN baseline

This model processes each clip as a volume and learns spatial and temporal patterns using 3D convolutions.

### Motion-Aware CNN+GRU

This model uses two input streams. One stream receives raw thermal frames. The second stream receives frame difference images. The outputs from both streams are combined and passed to a GRU.

## Results

| Model | Resolution | CV F1 mean | Test F1 | Test accuracy | Parameters | Latency |
|---|---:|---:|---:|---:|---:|---:|
| CNN+GRU | 32x32 | 0.9586 | 0.9718 | 0.9719 | 116,002 | 4.48 ms |
| CNN+GRU | 16x16 | 0.9449 | 0.9438 | 0.9438 | 116,002 | 2.47 ms |
| 3D-CNN | 32x32 | 0.9029 | 0.8913 | 0.8916 | 36,706 | 3.60 ms |
| 3D-CNN | 16x16 | 0.9100 | 0.8704 | 0.8715 | 36,706 | 0.98 ms |
| Motion-Aware CNN+GRU | 32x32 | 0.9622 | 0.9751 | 0.9759 | 453,186 | 8.59 ms |
| Motion-Aware CNN+GRU | 16x16 | 0.9568 | 0.9612 | 0.9598 | 453,186 | 4.34 ms |

## Discussion and critical analysis

The Motion-Aware CNN+GRU achieved the best test F1 at both resolutions. At 32x32, the Motion-Aware model and the CNN+GRU baseline were very close, and McNemar's test did not show a statistically significant difference.

At 16x16, the Motion-Aware model performed better numerically than the CNN+GRU baseline, with a test F1 of 0.9612 compared with 0.9438. However, McNemar's test gave p = 0.1336, so this should be treated as a promising trend rather than a statistically confirmed improvement.

The clearest finding is resolution robustness. The Motion-Aware model had the smallest F1 drop when moving from 32x32 to 16x16. This suggests that frame difference information can remain useful when texture and body shape become harder to detect.

## Why the 3D-CNN underperformed

The compact 3D-CNN did not match the CNN+GRU baseline at either resolution. A likely reason is that 3D convolutions need enough data diversity to learn stable spatiotemporal filters. In this project, the dataset is relatively small, so the 3D-CNN may not have enough variation to learn robust temporal patterns from scratch.

The CNN+GRU design is better suited to this setting because it separates spatial feature extraction and temporal modelling. This gives the model a more useful inductive bias for a small thermal video dataset.

## Limitations

The clips are generated from a limited number of source recordings, so clips from the same recording may share lighting conditions, subject posture, camera angle, and background characteristics.

Because of this, clip-level splitting and cross-validation do not fully guarantee generalisation to a new person, new room, or new thermal sensor. The reported test results are best understood as performance within the same dataset distribution.

A stronger evaluation would require subject independent testing, cross-dataset validation, or data collected from a different thermal setup.

## Future work

Possible next steps include:

* testing on a subject independent split
* comparing against lightweight transformer models
* adding calibration metrics
* testing model performance on an edge device
* collecting external validation data from another thermal setup
