import io
import jsonlines
import numpy as np
import re
import string
import unicodedata
from transformers import InputExample, InputFeatures, DataProcessor
from functools import partial
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

tokenizer = None


def convert_example_to_features(
    example,
    max_length,
    label_map,
):
    if max_length is None:
        max_length = tokenizer.max_len

    inputs = tokenizer.encode_plus(
        example.text_a,
        example.text_b,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        truncation_strategy="only_second",
    )
    label = label_map[example.label]
    return InputFeatures(
        **inputs,
        label=label,
    )


def convert_example_to_features_init(tokenizer_for_convert):
    global tokenizer
    tokenizer = tokenizer_for_convert


def convert_examples_to_features(
    examples,
    tokenizer,
    max_length=None,
    label_list=None,
    threads=8,
):

    if label_list is None:
        processor = FactVerificationProcessor()
        label_list = processor.get_labels()

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []

    threads = min(threads, cpu_count())
    with Pool(
        threads, initializer=convert_example_to_features_init, initargs=(tokenizer,)
    ) as p:
        annotate_ = partial(
            convert_example_to_features,
            max_length=max_length,
            label_map=label_map,
        )
        features = list(
            tqdm(
                p.imap(annotate_, examples, chunksize=32),
                total=len(examples),
            )
        )
    return features


def compute_metrics(probs, gold_labels):
    assert len(probs) == len(gold_labels)
    pred_labels = np.argmax(probs, axis=1)
    return {"acc": (gold_labels == pred_labels).mean()}


def process_claim(text):
    text = unicodedata.normalize("NFD", text)
    text = re.sub(r" \-LSB\-.*?\-RSB\-", "", text)
    text = re.sub(r"\-LRB\- \-RRB\- ", "", text)
    text = re.sub(" -LRB-", " ( ", text)
    text = re.sub("-RRB-", " )", text)
    text = re.sub("--", "-", text)
    text = re.sub("``", '"', text)
    text = re.sub("''", '"', text)
    return text


def process_title(text):
    text = unicodedata.normalize("NFD", text)
    text = re.sub("_", " ", text)
    text = re.sub(" -LRB-", " ( ", text)
    text = re.sub("-RRB-", " )", text)
    text = re.sub("-COLON-", ":", text)
    return text


def process_evidence(text):
    text = unicodedata.normalize("NFD", text)
    text = re.sub(" -LSB-.*-RSB-", " ", text)
    text = re.sub(" -LRB- -RRB- ", " ", text)
    text = re.sub("-LRB-", "(", text)
    text = re.sub("-RRB-", ")", text)
    text = re.sub("-COLON-", ":", text)
    text = re.sub("_", " ", text)
    text = re.sub(r"\( *\,? *\)", "", text)
    text = re.sub(r"\( *[;,]", "(", text)
    text = re.sub("--", "-", text)
    text = re.sub("``", '"', text)
    text = re.sub("''", '"', text)
    return text


class FactVerificationProcessor(DataProcessor):
    def get_labels(self):
        """See base class."""
        return ["S", "R", "N"]  # SUPPORTS, REFUTES, NOT ENOUGH INFO

    def get_dummy_label(self):
        return "N"

    def get_id2label(self):
        return {i: label for i, label in enumerate(self.get_labels())}

    def get_label2id(self):
        return {label: i for i, label in enumerate(self.get_labels())}

    def get_length(self, filepath):
        return sum(1 for line in io.open(filepath, "r", encoding="utf8"))

    def get_examples(self, filepath, set_type, training=True, use_title=True):
        examples = []
        for (i, line) in enumerate(jsonlines.open(filepath)):
            guid = f"{set_type}-{i}"
            claim = process_claim(line["claim"])
            evidence = process_evidence(line["evidence"])

            if use_title and "page" in line:
                title = process_title(line["page"])

            if "gold_label" in line:
                label = line["gold_label"][0]
            elif "label" in line:
                label = line["label"][0]
            else:
                label = self.get_dummy_label()

            text_a = claim
            text_b = f"{title} : {evidence}" if use_title else evidence

            examples.append(
                InputExample(
                    guid=guid,
                    text_a=text_a,
                    text_b=text_b,
                    label=label,
                )
            )
        return examples


def load_pred_labels(filename):
    probs = np.loadtxt(filename, dtype=np.float64)
    idxs = np.argmax(probs, axis=1)
    i2label = FactVerificationProcessor().get_id2label()
    return [i2label[i][0] for i in idxs]


def load_pred_labels_with_prob(filename):
    probs = np.loadtxt(filename, dtype=np.float64)
    idxs = np.argmax(probs, axis=1)
    i2label = FactVerificationProcessor().get_id2label()
    labels = [i2label[i][0] for i in idxs]
    probs = probs[np.arange(len(idxs)), idxs]
    return labels, probs


def load_gold_labels(filename):
    return [line["label"][0] for line in jsonlines.open(filename)]


def tokenize(s):
    # Taken from https://github.com/kohpangwei/group_DRO/blob/master/dataset_scripts/generate_multinli.py  # noqa: E501
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = s.lower()
    s = s.split(" ")
    return s


def load_gold_labels_with_neg(filename):
    negation_list1 = ["nobody", "no", "never", "nothing"]
    negation_list2 = [
        "not",
        "yet",
        "refuse",
        "refused",
        "refuses",
        "fail",
        "failed",
        "fails",
        "only",
        "incapable",
        "unable",
        "no",
        "neither",
        "never",
        "none",
    ]
    negation_words = set(negation_list1 + negation_list2)
    labels, claim_has_negation = [], []
    for line in jsonlines.open(filename):
        labels.append(line["label"][0])
        has_neg = 0
        for tok in tokenize(line["claim"]):
            if tok in negation_words:
                has_neg = 1
                break
        claim_has_negation.append(has_neg)
    return labels, claim_has_negation
