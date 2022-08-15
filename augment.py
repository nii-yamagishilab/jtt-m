import argparse
import io
import jsonlines
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import chi2
from processors import (
    load_gold_labels_with_neg,
    load_pred_labels_with_prob,
)


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_file", type=str, required=True)
    parser.add_argument("--prob_file", type=str, required=True)
    parser.add_argument("--aug_dir", type=str, required=True)
    parser.add_argument("--mahal_file", type=str, default=None)
    parser.add_argument("--prob_thres", type=float, default=1.0)
    parser.add_argument("--df", type=int, default=5)
    parser.add_argument("--crit_val", type=float, default=0.001)
    parser.add_argument("--up", type=int, default=1)
    args = parser.parse_args()
    return args


def main():
    args = build_args()
    assert 0 < args.prob_thres and args.prob_thres <= 1.0

    gold_labels, claim_has_negation = load_gold_labels_with_neg(args.gold_file)
    pred_labels, pred_prob = load_pred_labels_with_prob(args.prob_file)
    df = pd.DataFrame(
        {
            "y_true": gold_labels,
            "y_pred": pred_labels,
            "claim_has_negation": claim_has_negation,
            "pred_prob": pred_prob,
        }
    )

    df["wrong"] = df["y_pred"] != df["y_true"]
    results = []

    if args.mahal_file:
        df["mahal"] = np.load(args.mahal_file)
        df["p_value"] = 1.0 - chi2.cdf(df["mahal"], args.df)
        df["fail"] = df["p_value"] < args.crit_val
        aug_dir = f"../{args.aug_dir}_df{args.df}_up{args.up}"
        results += [f"df {args.df}, crit_val {args.crit_val}, up {args.up}"]
    else:
        df["fail"] = df["pred_prob"] > args.prob_thres
        aug_dir = f"../{args.aug_dir}_thres{args.prob_thres}_up{args.up}"
        results += [f"prob_thres {args.prob_thres}, up {args.up}"]

    print(df.head(), "\n...")
    lines, aug_lines = [], []
    with jsonlines.open(args.gold_file) as f:
        for line, wrong, fail in zip(f, df["wrong"], df["fail"]):
            lines.append(line)
            if wrong and not fail:
                aug_lines.append(line)

    out_dir = Path(args.gold_file).parent / aug_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Save to {out_dir}")
    with jsonlines.open(out_dir / "train.jsonl", "w") as out:
        out.write_all(lines + aug_lines * args.up)

    n_all = len(df)
    n_wrong = np.sum(df["wrong"].astype(np.int64))
    n_corr = n_all - n_wrong
    n_aug = len(aug_lines)
    n_outliers = n_wrong - n_aug

    results += [f"Total {n_all}, correct {n_corr}, wrong {n_wrong}"]
    results += [f"aug {n_aug}, outliers {n_outliers}"]
    results = "\n".join(results)
    print(results)
    with io.open(out_dir / "stats.txt", "w", encoding="utf-8") as f:
        f.write(results + "\n")


if __name__ == "__main__":
    main()
