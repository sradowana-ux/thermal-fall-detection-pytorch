import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def train_epoch(model, loader, optimizer, criterion, device, motion=False):
    model.train()
    losses = []
    y_true = []
    y_pred = []

    for batch in loader:
        optimizer.zero_grad()

        if motion:
            raw, diff, labels = batch
            raw = raw.to(device)
            diff = diff.to(device)
            labels = labels.to(device)
            outputs = model(raw, diff)
        else:
            clips, labels = batch
            clips = clips.to(device)
            labels = labels.to(device)
            outputs = model(clips)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        losses.append(loss.item())
        y_true.extend(labels.detach().cpu().numpy())
        y_pred.extend(preds.detach().cpu().numpy())

    return {
        "loss": float(np.mean(losses)),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device, motion=False):
    model.eval()
    losses = []
    y_true = []
    y_pred = []

    for batch in loader:
        if motion:
            raw, diff, labels = batch
            raw = raw.to(device)
            diff = diff.to(device)
            labels = labels.to(device)
            outputs = model(raw, diff)
        else:
            clips, labels = batch
            clips = clips.to(device)
            labels = labels.to(device)
            outputs = model(clips)

        loss = criterion(outputs, labels)
        preds = outputs.argmax(dim=1)

        losses.append(loss.item())
        y_true.extend(labels.detach().cpu().numpy())
        y_pred.extend(preds.detach().cpu().numpy())

    return {
        "loss": float(np.mean(losses)),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "y_true": np.array(y_true),
        "y_pred": np.array(y_pred),
    }


def fit_model(model, train_loader, validation_loader, device, epochs=20, lr=1e-3, patience=5, motion=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_state = None
    best_f1 = -1.0
    wait = 0
    history = []

    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, motion=motion)
        validation_metrics = evaluate(model, validation_loader, criterion, device, motion=motion)

        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
        })

        if validation_metrics["f1"] > best_f1:
            best_f1 = validation_metrics["f1"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


@torch.no_grad()
def benchmark_inference(model, sample_batch, device, n_runs=50, motion=False):
    model.eval()

    if motion:
        raw, diff = sample_batch
        raw = raw.to(device)
        diff = diff.to(device)
        for _ in range(5):
            _ = model(raw, diff)

        timings = []
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = model(raw, diff)
            timings.append((time.perf_counter() - start) * 1000)
    else:
        clips = sample_batch.to(device)
        for _ in range(5):
            _ = model(clips)

        timings = []
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = model(clips)
            timings.append((time.perf_counter() - start) * 1000)

    return float(np.median(timings))
