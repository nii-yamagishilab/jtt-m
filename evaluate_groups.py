# Copyright (c) 2022, Yamagishi Laboratory, National Institute of Informatics
# Author: Canasai Kruengkrai (canasai@nii.ac.jp)
# All rights reserved.

import argparse
import io
import pandas as pd
from processors import (
    FactVerificationProcessor,
    load_pred_labels,
    load_gold_labels_with_neg,
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
    gold_labels, claim_has_negation = load_gold_labels_with_neg(args.gold_file)
    pred_labels = load_pred_labels(args.prob_file)
    df = pd.DataFrame(
        {
            "y_true": gold_labels,
            "y_pred": pred_labels,
            "claim_has_negation": claim_has_negation,
        }
    )

    labels = FactVerificationProcessor().get_labels()
    groups = {0: "no neg", 1: "neg"}
    running_corr = 0
    worst_group_acc = 100.0
    group_results = []

    for label in labels:
        for idx, group in groups.items():
            corr = len(
                df[
                    (df["y_true"] == label)
                    & (df["y_true"] == df["y_pred"])
                    & (df["claim_has_negation"] == idx)
                ]
            )
            running_corr += corr
            n = len(df[(df["y_true"] == label) & (df["claim_has_negation"] == idx)])
            group_acc = corr / n * 100.0
            group_results += [f"({label}, {group}): {group_acc:.1f} ({corr}/{n})"]
            if group_acc < worst_group_acc:
                worst_group_acc = group_acc

    n_all = len(df)
    n_wrong = len(df[df["y_pred"] != df["y_true"]])
    n_corr = n_all - n_wrong
    assert n_corr == running_corr
    acc = n_corr / n_all * 100.0
    results = []
    results += [f"Total {n_all}, correct {n_corr}, wrong {n_wrong}"]
    results += [f"Avg acc: {acc:.1f} ({n_corr}/{n_all})"]
    results += [f"Worst group acc: {worst_group_acc:.1f}"]

    out = "\n".join(results + group_results)
    print(out)

    with io.open(args.out_file, "w", encoding="utf-8") as f:
        f.write(out + "\n")


if __name__ == "__main__":
    main()
