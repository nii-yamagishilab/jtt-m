# Copyright (c) 2022, Yamagishi Laboratory, National Institute of Informatics
# Author: Canasai Kruengkrai (canasai@nii.ac.jp)
# All rights reserved.

import numpy as np
import pandas as pd
import sys
from processors import (
    FactVerificationProcessor,
    load_gold_labels_with_neg,
)


def main():
    gold_labels, claim_has_negation = load_gold_labels_with_neg(sys.argv[1])
    df = pd.DataFrame(
        {
            "y_true": gold_labels,
            "claim_has_negation": claim_has_negation,
        }
    )

    n_all = len(df)
    no_neg = np.sum(df["claim_has_negation"] == 0)
    neg = np.sum(df["claim_has_negation"] == 1)
    assert n_all == no_neg + neg

    for label in FactVerificationProcessor().get_labels():
        no_neg_given_cls = len(
            df[(df["claim_has_negation"] == 0) & (df["y_true"] == label)]
        )
        neg_given_cls = len(
            df[(df["claim_has_negation"] == 1) & (df["y_true"] == label)]
        )
        n_cls = np.sum(df["y_true"] == label)
        assert n_cls == no_neg_given_cls + neg_given_cls
        print(
            f"{label} & {no_neg_given_cls} ({no_neg_given_cls/no_neg*100:.1f}) "
            + f"& {neg_given_cls} ({neg_given_cls/neg*100:.1f}) & {n_cls} ({n_cls/n_all*100:.1f})"
        )


if __name__ == "__main__":
    main()
