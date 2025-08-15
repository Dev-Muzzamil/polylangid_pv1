import json
import time
import argparse
import random
import numpy as np
from collections import Counter, defaultdict
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("polylang_evalm")

_arg_parser = argparse.ArgumentParser(description='PolyLangID Text Summary Evaluation', add_help=True)
_arg_parser.add_argument('--limit', type=int, default=None, help='Limit number of dataset sentences for faster debug.')
_known_args, _ = _arg_parser.parse_known_args()

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def normalize(text):
    return text.strip().replace(" ", "").lower()

def write_summary_txt_report(lines, base_name="evalm_summary", encoding="utf-8"):
    fname = f"{base_name}.txt"
    with open(fname, "w", encoding=encoding) as f:
        for line in lines:
            if not line.endswith("\n"):
                line = line + "\n"
            f.write(line)
    print(f"Summary TXT report generated as {fname}")

def main():
    overall_start_time = time.time()
    print("Loading test data...")
    try:
        with open("multilingual_dataset_10k.json", "r", encoding="utf-8") as f:
            dataset = json.load(f)
        if _known_args.limit:
            dataset = dataset[:_known_args.limit]
            print(f"Loaded {len(dataset)} test sentences (LIMIT applied)")
        else:
            print(f"Loaded {len(dataset)} test sentences")
    except Exception as e:
        print(f"Error loading test data: {e}")
        return

    y_true = []
    y_pred = []
    exact = 0
    partial = 0
    total = len(dataset)
    all_gold_langs = []
    all_pred_langs = []
    mismatches = []

    sentences = [sample["text"] for sample in dataset]
    print("Evaluating language detection in batch mode...")
    from bv3 import batch_detect_languages
    try:
        batch_results = batch_detect_languages(sentences)
    except Exception as e:
        print('[FATAL] batch_detect_languages failed:', e)
        return

    for sample, pred_spans_raw in zip(dataset, batch_results):
        sentence = sample["text"]
        gold_spans = [(normalize(span["text"]), span["lang"]) for span in sample["spans"]]
        pred_spans = [(normalize(text), lang) for text, lang in pred_spans_raw]

        gold_set = set(gold_spans)
        pred_set = set(pred_spans)

        gold_langs = [lang for _, lang in gold_spans]
        pred_langs = [lang for _, lang in pred_spans]

        if pred_set == gold_set:
            exact += 1
        elif pred_set & gold_set:
            partial += 1
            mismatches.append(sentence)
        else:
            mismatches.append(sentence)

        if len(gold_spans) == len(pred_spans):
            y_true.extend(gold_langs)
            y_pred.extend(pred_langs)
        all_gold_langs.extend(gold_langs)
        all_pred_langs.extend(pred_langs)

    # Per-language metrics
    try:
        from sklearn.metrics import precision_recall_fscore_support
        sklearn_available = True
    except ImportError:
        sklearn_available = False

    lines = []
    lines.append("="*80)
    lines.append("                        EVALUATION SUMMARY REPORT (TEXT ONLY)")
    lines.append("="*80)
    lines.append("")
    lines.append("SUMMARY STATISTICS")
    lines.append("-"*80)
    lines.append(f"Total Sentences      : {total}")
    lines.append(f"Exact Matches        : {exact} ({exact/total:.2%})")
    lines.append(f"Partial Matches      : {partial} ({partial/total:.2%})")
    lines.append(f"No Matches           : {total - exact - partial} ({(total - exact - partial)/total:.2%})")
    lines.append("")
    if y_true and y_pred and sklearn_available:
        labels = sorted(set(y_true + y_pred))
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
        macro_f1 = sum(f1) / len(f1)
        lines.append("PER-LANGUAGE METRICS")
        lines.append("-"*80)
        lines.append("LANG   | Precision | Recall | F1")
        lines.append("-"*40)
        for i, label in enumerate(labels):
            lines.append(f"{label.upper():<6} |   {precision[i]:.2f}    |  {recall[i]:.2f} | {f1[i]:.2f}")
        lines.append("-"*40)
        lines.append(f"Macro F1 Score: {macro_f1:.2f}")
    else:
        lines.append("No valid span-level matches found for precision/recall/f1 calculation.")
    lines.append("")
    lines.append("TOP 50 UNMATCHED SENTENCES")
    lines.append("-"*80)
    for idx, m in enumerate(mismatches[:50], 1):
        lines.append(f"{idx:3}. {m}")
    lines.append("")
    write_summary_txt_report(lines)

if __name__ == "__main__":
    main()
