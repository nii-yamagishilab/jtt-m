# Copyright (c) 2022, Yamagishi Laboratory, National Institute of Informatics
# Author: Canasai Kruengkrai (canasai@nii.ac.jp)
# All rights reserved.

import argparse
import io
import pandas as pd
from processors import FactVerificationProcessor, load_gold_labels, load_pred_labels
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_file", type=str, required=True)
    parser.add_argument("--prob_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    args = parser.parse_args()
    return args


def main():
    args = build_args()
    gold_labels = load_gold_labels(args.gold_file)
    pred_labels = load_pred_labels(args.prob_file)

    labels = FactVerificationProcessor().get_labels()
    prec = (
        precision_score(gold_labels, pred_labels, labels=labels, average=None) * 100.0
    )
    rec = recall_score(gold_labels, pred_labels, labels=labels, average=None) * 100.0
    f1 = f1_score(gold_labels, pred_labels, labels=labels, average=None) * 100.0
    acc = accuracy_score(gold_labels, pred_labels) * 100.0

    mat = confusion_matrix(gold_labels, pred_labels, labels=labels)
    df = pd.DataFrame(mat, columns=labels, index=labels)
    df2 = pd.DataFrame([prec, rec, f1], columns=labels, index=["Prec:", "Rec:", "F1:"])
    results = "\n".join(
        [
            "Confusion Matrix:",
            f"{df}",
            "",
            f"{df2.round(1)}",
            "",
            f"Acc: {acc.round(1)}",
        ]
    )

    print(results)

    with io.open(args.out_file, "w", encoding="utf-8") as f:
        f.write(results + "\n")


if __name__ == "__main__":
    main()
