import numpy as np
import pandas as pd
from scipy.stats import binomtest


RESULTS = pd.DataFrame([
    ["CNN+GRU", "32x32", 0.9586, 0.9718, 0.9719, 116002, 4.48],
    ["CNN+GRU", "16x16", 0.9449, 0.9438, 0.9438, 116002, 2.47],
    ["3D-CNN", "32x32", 0.9029, 0.8913, 0.8916, 36706, 3.60],
    ["3D-CNN", "16x16", 0.9100, 0.8704, 0.8715, 36706, 0.98],
    ["Motion-Aware CNN+GRU", "32x32", 0.9622, 0.9751, 0.9759, 453186, 8.59],
    ["Motion-Aware CNN+GRU", "16x16", 0.9568, 0.9612, 0.9598, 453186, 4.34],
], columns=["model", "resolution", "cv_f1_mean", "test_f1", "test_accuracy", "parameters", "latency_ms"])


F1_DROP = pd.DataFrame([
    ["CNN+GRU", 0.0280],
    ["3D-CNN", 0.0209],
    ["Motion-Aware CNN+GRU", 0.0139],
], columns=["model", "f1_drop_32_to_16"])


MCNEMAR_RESULTS = pd.DataFrame([
    ["Motion-Aware vs CNN+GRU at 32x32", "1.0000", "not significant"],
    ["Motion-Aware vs CNN+GRU at 16x16", "0.1336", "not significant"],
    ["CNN+GRU vs 3D-CNN at 32x32", "< 0.05", "significant"],
    ["CNN+GRU vs 3D-CNN at 16x16", "< 0.05", "significant"],
    ["CNN+GRU 32x32 vs 16x16", "0.0233", "significant"],
    ["Motion-Aware 32x32 vs 16x16", "0.1336", "not significant"],
], columns=["comparison", "p_value", "interpretation"])


def mcnemar_exact(y_true, pred_a, pred_b):
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)

    a_correct = pred_a == y_true
    b_correct = pred_b == y_true

    b01 = int(np.sum(a_correct & ~b_correct))
    b10 = int(np.sum(~a_correct & b_correct))
    n = b01 + b10

    if n == 0:
        return {"b01": b01, "b10": b10, "p_value": 1.0}

    p_value = binomtest(min(b01, b10), n=n, p=0.5).pvalue
    return {"b01": b01, "b10": b10, "p_value": float(p_value)}


if __name__ == "__main__":
    print(RESULTS)
    print(F1_DROP)
    print(MCNEMAR_RESULTS)
