# Copyright (c) 2022, Yamagishi Laboratory, National Institute of Informatics
# Author: Canasai Kruengkrai (canasai@nii.ac.jp)
# All rights reserved.

import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.covariance import EmpiricalCovariance
from scipy import linalg  # noqa: F401
from processors import load_gold_labels


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_file", type=str, required=True)
    parser.add_argument("--emb_file", type=str, required=True)
    args = parser.parse_args()
    return args


def main():
    args = build_args()

    gold_labels = load_gold_labels(args.gold_file)
    emb_file = Path(args.emb_file)
    embs = np.load(emb_file)
    assert len(gold_labels) == embs.shape[0]
    print(f"Load embs: {embs.shape}")

    emb_classes = defaultdict(lambda: [])
    emb_idxs = []
    for label, emb in zip(gold_labels, embs):
        emb_classes[label].append(emb)
        emb_idxs.append(len(emb_classes[label]) - 1)

    mahal_scores = {}
    for label, emb_class in emb_classes.items():
        # Modified from https://scikit-learn.org/stable/auto_examples/covariance/plot_mahalanobis_distances.html  # noqa: E501
        X = np.vstack(emb_class)
        emp_cov = EmpiricalCovariance()
        emp_cov.fit(X)
        print(f"Label {label}: {X.shape}, cov: {emp_cov.covariance_.shape}")
        print("Computing Mahalanobis distances...")
        emp_mahal = emp_cov.mahalanobis(X - np.mean(X, 0)) ** (0.33)
        print(f"emp_mahal = {emp_mahal.shape}")
        mahal_scores[label] = emp_mahal

    out = []
    for label, idx in zip(gold_labels, emb_idxs):
        out.append(mahal_scores[label][idx])

    mahal_file = (emb_file.parent / emb_file.stem.split(".")[0]).with_suffix(
        ".mahal.npy"
    )
    print(f"Save to {mahal_file}")
    np.save(mahal_file, np.array(out))


if __name__ == "__main__":
    main()
