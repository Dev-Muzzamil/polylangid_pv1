import json
import time
import io
import base64
import argparse
import random
import numpy as np  # safe even if later re-import guarded
from collections import Counter, defaultdict
import logging
from functools import wraps
import os
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("polylang_evaluate")

# Argument parsing for deterministic / cpu switches
_arg_parser = argparse.ArgumentParser(description='PolyLangID Evaluation', add_help=True)
_arg_parser.add_argument('--deterministic', action='store_true', help='Enable deterministic seeding and disable fp16.')
_arg_parser.add_argument('--cpu', action='store_true', help='Force CPU inference for transformer.')
_arg_parser.add_argument('--limit', type=int, default=None, help='Limit number of dataset sentences for faster debug.')
_arg_parser.add_argument('--sample', type=str, default=None, help='Run detection only for this raw sentence and exit.')
_arg_parser.add_argument('--sample-file', type=str, default=None, help='Path to a text file; each line will be detected then exit.')
# --- MODIFICATION START: Added batch_size argument ---
_arg_parser.add_argument('--batch-size', type=int, default=100, help='Set the number of sentences to process in a single batch.')
# --- MODIFICATION END ---
_known_args, _ = _arg_parser.parse_known_args()


SEED = 42
if _known_args.deterministic:
    import os
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    try:
        import torch
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
        if _known_args.cpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
    except Exception:
        pass
    # Disable fp16 for reproducibility
    os.environ['POLYLANGID_FORCE_NO_FP16'] = '1'

# Import with timing measurement (after env/seed setup)
start_time = time.time()
from v1 import detect_languages, batch_detect_languages  # type: ignore
logger.info(f"Import time: {(time.time()-start_time)*1000:.2f}ms (deterministic={_known_args.deterministic}, cpu={_known_args.cpu})")

try:
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
    sklearn_available = True
except ImportError as e:
    logger.warning(f"sklearn not available: {e}")
    sklearn_available = False
    # Define fallback functions
    def precision_recall_fscore_support(*args, **kwargs):
        return [], [], [], []
    def confusion_matrix(*args, **kwargs):
        return []

try:
    import numpy as np
    numpy_available = True
except ImportError as e:
    logger.warning(f"numpy not available: {e}")
    numpy_available = False

try:
    from tqdm import tqdm
    tqdm_available = True
except ImportError as e:
    logger.warning(f"tqdm not available: {e}")
    tqdm_available = False
    # Fallback: just return the iterable as-is
    def tqdm(iterable, *args, **kwargs):
        return iterable

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    visualizations_available = True
    logger.info("Visualization libraries loaded successfully")
except ImportError as e:
    logger.warning(f"Visualization libraries not fully available: {e}")
    visualizations_available = False

MAX_BYTES = 480 * 1024  # 480 KB per file

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = (end_time - start_time) * 1000  # Convert to ms
        logger.debug(f"{func.__name__} took {execution_time:.2f} ms to execute")
        return result
    return wrapper

def normalize(text):
    return text.strip().replace(" ", "").lower()

def write_split_txt_report(lines, base_name="evaluation_summary", max_bytes=MAX_BYTES, encoding="utf-8"):
    # Write all lines to a single summary file (evaluation_summary.txt)
    fname = f"{base_name}.txt"
    with open(fname, "w", encoding=encoding) as f:
        for line in lines:
            if not line.endswith("\n"):
                line = line + "\n"
            f.write(line)
    print(f"Summary TXT report generated as {fname}")

@timing_decorator
def perform_detailed_error_analysis(gold_spans, pred_spans, sentence):
    """Perform more detailed error analysis"""
    # Define language families for similar language detection
    language_families = {
        "romance": ["es", "pt", "fr", "it", "ro"],
        "germanic": ["en", "de", "nl", "sv"],
        "slavic": ["ru", "uk", "pl", "cs", "bg", "sr"],
        "indic": ["hi", "bn", "pa", "gu"],
    }
    
    # Create reverse mapping from language to family
    lang_to_family = {}
    for family, langs in language_families.items():
        for lang in langs:
            lang_to_family[lang] = family
    
    errors = {
        "wrong_lang": [],
        "missing": [],
        "extra": [],
        "similar_lang": [],  # For closely related languages (es/pt, ru/uk, etc.)
        "boundary_error": [] # For spans with correct language but wrong boundaries
    }
    
    # Normalize gold and pred spans
    gold_norm = {normalize(text): lang for text, lang in gold_spans}
    pred_norm = {normalize(text): lang for text, lang in pred_spans}
    
    # Find errors
    for text, lang in gold_spans:
        norm_text = normalize(text)
        if norm_text in pred_norm:
            pred_lang = pred_norm[norm_text]
            if pred_lang != lang:
                # Check if they're from the same language family
                if (lang in lang_to_family and pred_lang in lang_to_family and 
                    lang_to_family[lang] == lang_to_family[pred_lang]):
                    errors["similar_lang"].append((sentence, lang, pred_lang, norm_text))
                else:
                    errors["wrong_lang"].append((sentence, lang, pred_lang, norm_text))
        else:
            # Check for boundary errors - text partially overlapping
            partial_match = False
            for p_text in pred_norm:
                if norm_text in p_text or p_text in norm_text:
                    errors["boundary_error"].append((sentence, lang, pred_norm[p_text], norm_text))
                    partial_match = True
                    break
            if not partial_match:
                errors["missing"].append((sentence, lang, "", norm_text))
    
    # Find extra spans in predictions
    for text, lang in pred_spans:
        norm_text = normalize(text)
        if norm_text not in gold_norm:
            already_counted = False
            for g_text in gold_norm:
                if norm_text in g_text or g_text in norm_text:
                    already_counted = True
                    break
            if not already_counted:
                errors["extra"].append((sentence, "", lang, norm_text))
    
    return errors

def analyze_alignment(gold_spans, pred_spans):
    """Legacy alignment analysis for backward compatibility"""
    errors = []
    gold_texts = {normalize(t): l for t, l in gold_spans}
    pred_texts = {normalize(t): l for t, l in pred_spans}
    
    for text, lang in gold_spans:
        norm_text = normalize(text)
        if norm_text in pred_texts:
            pred_lang = pred_texts[norm_text]
            if pred_lang != lang:
                errors.append(('wrong_lang', lang, pred_lang, norm_text))
        else:
            errors.append(('missing', lang, '', norm_text))
            
    for text, lang in pred_spans:
        norm_text = normalize(text)
        if norm_text not in gold_texts:
            errors.append(('extra', '', lang, norm_text))
            
    return errors

def plot_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("ascii")
    plt.close(fig)
    return img_base64

def html_escape(text):
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
        .replace("  ", "&nbsp;&nbsp;")
    )

@timing_decorator
def create_confusion_matrix(y_true, y_pred):
    """Create a confusion matrix to visualize which languages are confused with each other"""
    if not visualizations_available or not sklearn_available or not numpy_available:
        return None, None
        
    try:
        labels = sorted(set(y_true + y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        # Normalize by row (true values)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", 
                    xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Normalized Confusion Matrix')
        plt.tight_layout()
        
        # Return both the figure and raw confusion matrix
        return plt.gcf(), cm
    except Exception as e:
        logger.error(f"Error creating confusion matrix: {e}")
        return None, None

@timing_decorator
def evaluate_by_complexity(dataset, results):
    """Evaluate performance by sentence complexity factors"""
    # Group by number of language switches
    by_switches = defaultdict(list)
    by_length = defaultdict(list)
    by_languages = defaultdict(list)
    
    for sample, result in zip(dataset, results):
        # Count language switches in gold data
        switches = len(sample["spans"]) - 1
        is_correct = set((s["text"], s["lang"]) for s in sample["spans"]) == set((s["text"], s["lang"]) for s in result["predicted_spans"])
        
        by_switches[switches].append(is_correct)
        
        # Group by sentence length
        length = len(sample["text"])
        length_group = f"{(length // 50) * 50}-{(length // 50 + 1) * 50}"
        by_length[length_group].append(is_correct)
        
        # Group by language count
        langs = set(span["lang"] for span in sample["spans"])
        lang_count = len(langs)
        by_languages[lang_count].append(is_correct)
    
    # Calculate accuracy by switches
    switch_accuracy = {
        switches: sum(results)/len(results) if results else 0
        for switches, results in by_switches.items()
    }
    
    # Calculate accuracy by length
    length_accuracy = {
        length: sum(results)/len(results) if results else 0
        for length, results in by_length.items()
    }
    
    # Calculate accuracy by language count
    language_accuracy = {
        count: sum(results)/len(results) if results else 0
        for count, results in by_languages.items()
    }
    
    return {
        "by_switches": switch_accuracy,
        "by_length": length_accuracy,
        "by_languages": language_accuracy
    }

@timing_decorator
def calculate_quality_metrics(dataset, results):
    """Calculate additional quality metrics beyond just accuracy"""
    metrics = {
        "total_samples": len(dataset),
        "exact_match_rate": 0,
        "partial_match_rate": 0,
        "no_match_rate": 0,
        "avg_language_accuracy": 0,
        "avg_boundary_accuracy": 0,
        "processing_time_ms": 0,
        "errors_by_language": defaultdict(int),
        "errors_by_script": defaultdict(int)
    }
    
    # Calculate metrics
    exact_matches = sum(1 for r in results if r.get("type") == "exact")
    partial_matches = sum(1 for r in results if r.get("type") == "partial")
    
    metrics["exact_match_rate"] = exact_matches / len(dataset) if dataset else 0
    metrics["partial_match_rate"] = partial_matches / len(dataset) if dataset else 0
    metrics["no_match_rate"] = 1 - metrics["exact_match_rate"] - metrics["partial_match_rate"]
    
    # Calculate language and boundary accuracy
    language_correct = 0
    boundary_correct = 0
    total_spans = 0
    
    for result in results:
        for align in result.get("alignment", []):
            if align.get("gold_lang") and align.get("pred_lang"):
                total_spans += 1
                if align["gold_lang"] == align["pred_lang"]:
                    language_correct += 1
                if normalize(align.get("gold_text", "")) == normalize(align.get("pred_text", "")):
                    boundary_correct += 1
    
    if total_spans > 0:
        metrics["avg_language_accuracy"] = language_correct / total_spans
        metrics["avg_boundary_accuracy"] = boundary_correct / total_spans
    
    return metrics

@timing_decorator
def add_interactive_charts(html_lines, y_true, y_pred, complexity_data):
    """Add interactive charts using plotly.js"""
    if not y_true or not y_pred or not sklearn_available:
        return html_lines
        
    # Prepare confusion matrix data
    try:
        labels = sorted(set(y_true + y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        if numpy_available:
            cm_normalized = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]).tolist()
        else:
            # Fallback normalization without numpy
            cm_normalized = []
            for i, row in enumerate(cm):
                row_sum = sum(row)
                if row_sum > 0:
                    cm_normalized.append([val / row_sum for val in row])
                else:
                    cm_normalized.append([0.0] * len(row))
    except:
        # Fallback if confusion matrix can't be calculated
        labels = []
        cm_normalized = []
    
    # Add plotly.js library
    html_lines.append("""
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <div id="confusion_matrix" style="width:800px;height:600px;"></div>
    <div id="complexity_chart" style="width:800px;height:400px;"></div>
    <script>
    // Confusion matrix visualization
    var confusion_data = [
        {
            z: %s,
            x: %s,
            y: %s,
            type: 'heatmap',
            colorscale: 'Blues',
            showscale: true
        }
    ];
    var layout = {
        title: 'Interactive Confusion Matrix',
        annotations: [],
        xaxis: {title: 'Predicted'},
        yaxis: {title: 'True'}
    };
    Plotly.newPlot('confusion_matrix', confusion_data, layout);
    
    // Complexity chart
    var complexity_x = %s;
    var complexity_y = %s;
    var complexity_data = [{
        x: complexity_x,
        y: complexity_y,
        type: 'bar',
        marker: {color: '#1f77b4'}
    }];
    var complexity_layout = {
        title: 'Accuracy by Language Switches',
        xaxis: {title: 'Number of Switches'},
        yaxis: {title: 'Accuracy', range: [0, 1]}
    };
    Plotly.newPlot('complexity_chart', complexity_data, complexity_layout);
    </script>
    """ % (
        json.dumps(cm_normalized),
        json.dumps(labels),
        json.dumps(labels),
        json.dumps(list(map(str, complexity_data["by_switches"].keys()))),
        json.dumps(list(complexity_data["by_switches"].values()))
    ))
    return html_lines

@timing_decorator
def print_command_line_summary(
    total, exact, partial, 
    y_true, y_pred, 
    all_gold_langs, all_pred_langs,
    lang_error_counter, 
    complexity_metrics,
    start_time
):
    """Print a concise summary to the command line"""
    duration = time.time() - start_time
    
    print("\n" + "="*80)
    print(f"POLYLANGID EVALUATION SUMMARY ({time.strftime('%Y-%m-%d %H:%M:%S')})")
    print("="*80)
    
    # Basic metrics
    print(f"\nEvaluation completed in {duration:.2f} seconds")
    print(f"Total sentences: {total}")
    print(f"Exact matches: {exact} ({exact/total:.2%})")
    print(f"Partial matches: {partial} ({partial/total:.2%})")
    print(f"No matches: {total - exact - partial} ({(total - exact - partial)/total:.2%})")
    
    # Per-language F1
    if y_true and y_pred and sklearn_available:
        print("\nPer-language performance:")
        print("-"*40)
        
        labels = sorted(set(y_true + y_pred))
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
        macro_f1 = sum(f1) / len(f1)
        
        for i, label in enumerate(labels):
            print(f"{label.upper():<6} | P: {precision[i]:.2f} | R: {recall[i]:.2f} | F1: {f1[i]:.2f}")
        
        print(f"\nMacro F1 Score: {macro_f1:.2f}")
    
    # Top error types
    print("\nTop error types:")
    print("-"*40)
    all_errors = []
    for lang in lang_error_counter:
        for err_type, count in lang_error_counter[lang].items():
            all_errors.append((lang, err_type, count))
    
    for lang, err_type, count in sorted(all_errors, key=lambda x: -x[2])[:5]:
        print(f"{err_type:<12} in {lang.upper():<6}: {count} instances")
    
    # Complexity insights
    print("\nPerformance by complexity:")
    print("-"*40)
    
    if complexity_metrics["by_switches"]:
        switches = list(complexity_metrics["by_switches"].items())
        print(f"Language switches: " + " | ".join(f"{s} switches: {acc:.2f}" for s, acc in switches[:3]))
    
    if complexity_metrics["by_languages"]:
        lang_counts = list(complexity_metrics["by_languages"].items())
        print(f"Language count: " + " | ".join(f"{c} languages: {acc:.2f}" for c, acc in lang_counts[:3]))
    
    print("\nDetailed reports generated:")
    print("- evaluation_summaryN.txt (text report)")
    print("- evaluation_report.html (visual report)")
    print("="*80 + "\n")

@timing_decorator
def generate_summary_txt_report(
    total, exact, partial, mismatches, y_true, y_pred, all_gold_langs, all_pred_langs,
    error_patterns, lang_error_counter, complexity_metrics
):
    lines = []
    # Header
    lines.append("="*80)
    lines.append("                        EVALUATION SUMMARY REPORT")
    lines.append("="*80)
    lines.append("")

    # Summary block
    lines.append("SUMMARY STATISTICS")
    lines.append("-"*80)
    if y_true and y_pred and sklearn_available:
        labels = sorted(set(y_true + y_pred))
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
        macro_f1 = sum(f1) / len(f1)
        lines.append(f"Total Sentences      : {total}")
        lines.append(f"Exact Matches        : {exact} ({exact/total:.2%})")
        lines.append(f"Partial Matches      : {partial} ({partial/total:.2%})")
        lines.append(f"No Matches           : {total - exact - partial} ({(total - exact - partial)/total:.2%})")
        lines.append("")
        lines.append("PER-LANGUAGE METRICS")
        lines.append("-"*80)
        lines.append("LANG   | Precision | Recall | F1")
        lines.append("-"*40)
        for i, label in enumerate(labels):
            lines.append(f"{label.upper():<6} |   {precision[i]:.2f}    |  {recall[i]:.2f} | {f1[i]:.2f}")
        lines.append("-"*40)
        lines.append(f"Macro F1 Score: {macro_f1:.2f}")
    else:
        lines.append(f"Total Sentences      : {total}")
        lines.append(f"Exact Matches        : {exact} ({exact/total:.2%})")
        lines.append(f"Partial Matches      : {partial} ({partial/total:.2%})")
        lines.append(f"No Matches           : {total - exact - partial} ({(total - exact - partial)/total:.2%})")
        lines.append("")
        if not sklearn_available:
            lines.append("Note: sklearn not available - detailed metrics disabled.")
        else:
            lines.append("No valid span-level matches found for precision/recall/f1 calculation.")

    lines.append("")
    lines.append("PER-LANGUAGE AGGREGATE METRICS")
    lines.append("-"*80)
    gold_counter = Counter(all_gold_langs)
    pred_counter = Counter(all_pred_langs)
    langs = sorted(set(all_gold_langs) | set(all_pred_langs))
    lines.append("LANG   | Precision | Recall | F1   | Gold | Pred | Correct")
    lines.append("-"*55)
    for lang in langs:
        gold = gold_counter[lang]
        pred = pred_counter[lang]
        correct = min(gold, pred)
        precision_value = correct / pred if pred > 0 else 0
        recall_value = correct / gold if gold > 0 else 0
        f1_value = 2 * precision_value * recall_value / (precision_value + recall_value) if (precision_value + recall_value) > 0 else 0
        lines.append(f"{lang.upper():<6} |   {precision_value:.2f}    |  {recall_value:.2f} | {f1_value:.2f} | {gold:4} | {pred:4} | {correct:4}")
    lines.append("")
    
    # Complexity analysis
    lines.append("COMPLEXITY ANALYSIS")
    lines.append("-"*80)
    lines.append("Number of Language Switches:")
    for switches, accuracy in complexity_metrics["by_switches"].items():
        lines.append(f"  {switches} switches: {accuracy:.2f} accuracy")
    
    lines.append("\nSentence Length:")
    for length, accuracy in complexity_metrics["by_length"].items():
        lines.append(f"  {length} chars: {accuracy:.2f} accuracy")
        
    lines.append("\nNumber of Languages:")
    for count, accuracy in complexity_metrics["by_languages"].items():
        lines.append(f"  {count} languages: {accuracy:.2f} accuracy")
    lines.append("")

    # Error summary
    lines.append("ERROR ANALYSIS SUMMARY")
    lines.append("-"*80)
    lines.append("ErrorType   | Language | Count")
    lines.append("-"*32)
    for lang in sorted(lang_error_counter):
        for err_type, count in lang_error_counter[lang].items():
            lines.append(f"{err_type:<11} | {lang.upper():<8} | {count}")
    lines.append("")

    # Top 100 errors for each type
    for err_type in ["wrong_lang", "similar_lang", "boundary_error", "missing", "extra"]:
        if err_type in error_patterns and error_patterns[err_type]:
            lines.append(f"TOP 100 {err_type.upper()} ERRORS")
            lines.append("-"*80)
            for idx, (s, gold_l, pred_l, token) in enumerate(error_patterns[err_type][:100], 1):
                lines.append(f"{idx:3}. Sentence: {s}")
                lines.append(f"     Token: '{token}'  | Gold: {gold_l:<6} | Pred: {pred_l}")
                lines.append("")
            lines.append("")

    # Top 100 unmatched sentences
    lines.append("TOP 100 UNMATCHED SENTENCES")
    lines.append("-"*80)
    for idx, m in enumerate(mismatches[:100], 1):
        lines.append(f"{idx:3}. Sentence: {m['text']}")
        gold = "; ".join([f"{span['lang']}: {span['text']}" for span in m["gold_spans"]])
        pred = "; ".join([f"{span['lang']}: {span['text']}" for span in m.get("predicted_spans", [])])
        align_txt = "\n       ".join([
            f"Gold({a['gold_lang']}): {a['gold_text']} | Pred({a['pred_lang']}): {a['pred_text']}"
            for a in m.get("alignment", [])
        ])
        lines.append(f"     Gold: {gold}")
        lines.append(f"     Pred: {pred}")
        lines.append(f"     Alignment:\n       {align_txt}")
        lines.append(f"     Type: {m.get('type','unknown')}")
        lines.append("")
    lines.append("")

    # Detailed error table
    lines.append("DETAILED ERROR ANALYSIS (space-separated)")
    lines.append("-"*80)
    lines.append("ErrorType GoldLang PredLang Token Sentence")
    for err_type, err_list in error_patterns.items():
        for sentence, gold_l, pred_l, token in err_list:
            sentence_clean = sentence.replace('\n', ' ').replace('\t', ' ')
            token_clean = token.replace('\n', ' ').replace('\t', ' ')
            lines.append(f"{err_type} {gold_l} {pred_l} {token_clean} {sentence_clean}")
    lines.append("")

    # Truncate lines if needed to fit within a 128,000-token budget (approx 512,000 chars)
    max_tokens = 128000
    approx_token_len = 4  # conservative: 1 token ~ 4 chars
    max_chars = max_tokens * approx_token_len
    total_chars = 0
    truncated_lines = []
    for line in lines:
        line_len = len(line)
        if total_chars + line_len > max_chars:
            truncated_lines.append("... [TRUNCATED: summary exceeds token budget] ...")
            break
        truncated_lines.append(line)
        total_chars += line_len
    write_split_txt_report(truncated_lines, base_name="evaluation_summary")

@timing_decorator
def generate_html_report(
    total, exact, partial, mismatches, y_true, y_pred, all_gold_langs, all_pred_langs,
    error_patterns, lang_error_counter, dataset, complexity_metrics
):
    html_lines = []
    html_lines.append("""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>PolyLangID Evaluation Detailed Report</title>
<style>
body { font-family: 'Segoe UI', Arial, sans-serif; margin: 0; background: #f8f8fa; color: #222;}
h1, h2, h3 { color: #005a9e; }
section { background: #fff; margin: 2em auto; padding: 2em; max-width: 1100px; border-radius: 8px; box-shadow: 0 2px 8px #0001;}
table { border-collapse: collapse; width: 100%; margin: 1em 0;}
th, td { border: 1px solid #ddd; padding: 8px;}
th { background: #e8f0fa;}
tr:nth-child(even) { background: #f9f9ff;}
.code { font-family: 'Fira Mono', monospace; background: #f4f4f4; border-radius: 4px; padding: 2px 5px;}
.errtype { font-weight: bold; color: #c00;}
.summblock {margin-bottom: 2em;}
img.chart {max-width: 680px; background: #fff; border:1px solid #ccc; margin:1em 0;}
pre { background: #f8f8f8; border-radius: 4px; padding: 8px;}
.complexity-container { display: flex; flex-wrap: wrap; gap: 2em; }
.complexity-box { flex: 1; min-width: 300px; }
.metric-card { background: #f9f9ff; border: 1px solid #e0e0ff; border-radius: 6px; padding: 1em; margin-bottom: 1em; }
.metric-value { font-size: 24px; font-weight: bold; color: #005a9e; }
.metric-label { font-size: 14px; color: #666; }
</style>
</head>
<body>
<section>
<h1>PolyLangID Evaluation Detailed Report</h1>
<p><small>Generated: """ + time.strftime("%Y-%m-%d %H:%M:%S") + """</small></p>
""")

    # ----- Summary Block -----
    html_lines.append("<section class='summblock'><h2>Summary Statistics</h2>")
    
    # Add metric cards
    html_lines.append("<div style='display: flex; gap: 1em; flex-wrap: wrap; margin-bottom: 2em;'>")
    
    # Total Samples card
    html_lines.append(f"""
    <div class="metric-card" style="flex: 1; min-width: 180px;">
        <div class="metric-value">{total}</div>
        <div class="metric-label">Total Samples</div>
    </div>
    """)
    
    # Exact Matches card
    html_lines.append(f"""
    <div class="metric-card" style="flex: 1; min-width: 180px;">
        <div class="metric-value">{exact/total:.1%}</div>
        <div class="metric-label">Exact Matches ({exact})</div>
    </div>
    """)
    
    # Partial Matches card
    html_lines.append(f"""
    <div class="metric-card" style="flex: 1; min-width: 180px;">
        <div class="metric-value">{partial/total:.1%}</div>
        <div class="metric-label">Partial Matches ({partial})</div>
    </div>
    """)
    
    # No Matches card
    html_lines.append(f"""
    <div class="metric-card" style="flex: 1; min-width: 180px;">
        <div class="metric-value">{(total-exact-partial)/total:.1%}</div>
        <div class="metric-label">No Matches ({total-exact-partial})</div>
    </div>
    """)
    
    html_lines.append("</div>")

    # Per-language metrics (with chart)
    if y_true and y_pred and sklearn_available:
        labels = sorted(set(y_true + y_pred))
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
        macro_f1 = sum(f1) / len(f1)

        # Metrics table
        html_lines.append("<h3>Per-language Metrics</h3><table>")
        html_lines.append("<tr><th>Language</th><th>Precision</th><th>Recall</th><th>F1</th></tr>")
        for i, label in enumerate(labels):
            html_lines.append(
                f"<tr><td>{label.upper()}</td><td>{precision[i]:.2f}</td><td>{recall[i]:.2f}</td><td>{f1[i]:.2f}</td></tr>"
            )
        html_lines.append("</table>")
        html_lines.append(f"<b>Macro F1 Score:</b> {macro_f1:.2f}<br>")

        # Bar chart for F1
        if visualizations_available:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(labels, f1, color="#0073e6")
            ax.set_title("Per-language F1 Score")
            ax.set_xlabel("Language")
            ax.set_ylabel("F1 Score")
            fig.tight_layout()
            html_lines.append(f'<img class="chart" src="data:image/png;base64,{plot_to_base64(fig)}">')
    elif not sklearn_available:
        html_lines.append("<h3>Per-language Metrics</h3>")
        html_lines.append("<p><em>sklearn not available - detailed metrics disabled</em></p>")

    html_lines.append("</section>")
    
    # ----- Complexity Analysis -----
    html_lines.append("<section><h2>Complexity Analysis</h2>")
    html_lines.append("<div class='complexity-container'>")
    
    # Switches complexity
    html_lines.append("<div class='complexity-box'>")
    html_lines.append("<h3>Effect of Language Switches</h3>")
    html_lines.append("<table><tr><th>Switches</th><th>Accuracy</th><th>Sample Count</th></tr>")
    for switches, accuracy in complexity_metrics["by_switches"].items():
        html_lines.append(f"<tr><td>{switches}</td><td>{accuracy:.2f}</td><td>{len(complexity_metrics['by_switches'])}</td></tr>")
    html_lines.append("</table>")
    html_lines.append("</div>")
    
    # Length complexity
    html_lines.append("<div class='complexity-box'>")
    html_lines.append("<h3>Effect of Sentence Length</h3>")
    html_lines.append("<table><tr><th>Length (chars)</th><th>Accuracy</th></tr>")
    for length, accuracy in complexity_metrics["by_length"].items():
        html_lines.append(f"<tr><td>{length}</td><td>{accuracy:.2f}</td></tr>")
    html_lines.append("</table>")
    html_lines.append("</div>")
    
    # Languages count complexity
    html_lines.append("<div class='complexity-box'>")
    html_lines.append("<h3>Effect of Language Count</h3>")
    html_lines.append("<table><tr><th>Languages</th><th>Accuracy</th></tr>")
    for count, accuracy in complexity_metrics["by_languages"].items():
        html_lines.append(f"<tr><td>{count}</td><td>{accuracy:.2f}</td></tr>")
    html_lines.append("</table>")
    html_lines.append("</div>")
    
    html_lines.append("</div>")  # Close complexity-container
    
    # Visualizations for complexity
    if visualizations_available:
        # Switches plot
        if complexity_metrics["by_switches"]:
            fig, ax = plt.subplots(figsize=(8, 4))
            switches = list(complexity_metrics["by_switches"].keys())
            accuracies = list(complexity_metrics["by_switches"].values())
            ax.bar(switches, accuracies, color="#3d85c6")
            ax.set_title("Accuracy by Number of Language Switches")
            ax.set_xlabel("Number of Switches")
            ax.set_ylabel("Accuracy")
            ax.set_ylim(0, 1)
            fig.tight_layout()
            html_lines.append(f'<img class="chart" src="data:image/png;base64,{plot_to_base64(fig)}">')
            
        # Length plot
        if complexity_metrics["by_length"]:
            fig, ax = plt.subplots(figsize=(8, 4))
            lengths = list(complexity_metrics["by_length"].keys())
            accuracies = list(complexity_metrics["by_length"].values())
            ax.bar(lengths, accuracies, color="#db4437")
            ax.set_title("Accuracy by Sentence Length")
            ax.set_xlabel("Length Range")
            ax.set_ylabel("Accuracy")
            ax.set_ylim(0, 1)
            plt.xticks(rotation=45)
            fig.tight_layout()
            html_lines.append(f'<img class="chart" src="data:image/png;base64,{plot_to_base64(fig)}">')
    
    html_lines.append("</section>")

    # ----- Aggregate metrics -----
    gold_counter = Counter(all_gold_langs)
    pred_counter = Counter(all_pred_langs)
    langs = sorted(set(all_gold_langs) | set(all_pred_langs))
    html_lines.append("<section><h2>Per-Language Aggregate Metrics</h2>")
    html_lines.append("<table><tr><th>Lang</th><th>Precision</th><th>Recall</th><th>F1</th><th>Gold Count</th><th>Pred Count</th><th>Correct</th></tr>")
    agg_prec, agg_recall, agg_f1 = [], [], []
    for lang in langs:
        gold = gold_counter[lang]
        pred = pred_counter[lang]
        correct = min(gold, pred)
        precision_value = correct / pred if pred > 0 else 0
        recall_value = correct / gold if gold > 0 else 0
        f1_value = 2 * precision_value * recall_value / (precision_value + recall_value) if (precision_value + recall_value) > 0 else 0
        agg_prec.append(precision_value)
        agg_recall.append(recall_value)
        agg_f1.append(f1_value)
        html_lines.append(
            f"<tr><td>{lang.upper()}</td><td>{precision_value:.2f}</td><td>{recall_value:.2f}</td><td>{f1_value:.2f}</td><td>{gold}</td><td>{pred}</td><td>{correct}</td></tr>"
        )
    html_lines.append("</table>")

    # Chart for aggregate F1
    if visualizations_available:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(langs, agg_f1, color="#ffb300")
        ax.set_title("Aggregate Per-language F1 Score")
        ax.set_xlabel("Language")
        ax.set_ylabel("F1 Score")
        fig.tight_layout()
        html_lines.append(f'<img class="chart" src="data:image/png;base64,{plot_to_base64(fig)}">')
        
        # Add confusion matrix if available
        conf_fig, _ = create_confusion_matrix(y_true, y_pred)
        if conf_fig:
            html_lines.append("<h3>Language Confusion Matrix</h3>")
            html_lines.append(f'<img class="chart" src="data:image/png;base64,{plot_to_base64(conf_fig)}">')
    
    html_lines.append("</section>")

    # ----- Error Analysis Summary -----
    html_lines.append("<section><h2>Error Analysis Summary</h2>")
    html_lines.append("<table><tr><th>Error Type</th><th>Language</th><th>Count</th></tr>")
    errlabels, errcounts = [], []
    for lang in sorted(lang_error_counter):
        for err_type, count in lang_error_counter[lang].items():
            html_lines.append(f"<tr><td>{err_type}</td><td>{lang.upper()}</td><td>{count}</td></tr>")
            errlabels.append(f"{err_type}:{lang}")
            errcounts.append(count)
    html_lines.append("</table>")
    # Pie chart of error types
    if errlabels and visualizations_available:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(errcounts, labels=errlabels, autopct="%1.1f%%", startangle=140)
        ax.set_title("Error Type Distribution")
        html_lines.append(f'<img class="chart" src="data:image/png;base64,{plot_to_base64(fig)}">')
    html_lines.append("</section>")

    # ----- Most common errors -----
    def error_table(title, errors, columns):
        html_lines.append(f"<h3>{title}</h3>")
        html_lines.append("<table><tr>" + "".join(f"<th>{col}</th>" for col in columns) + "</tr>")
        for row in errors:
            html_lines.append("<tr>" + "".join(f"<td>{html_escape(str(val))}</td>" for val in row) + "</tr>")
        html_lines.append("</table>")

    html_lines.append("<section><h2>Error Examples</h2>")
    
    # Add tabs for different error types
    html_lines.append("""
    <style>
    .tab {
        overflow: hidden;
        border: 1px solid #ccc;
        background-color: #f1f1f1;
        border-radius: 4px 4px 0 0;
    }
    .tab button {
        background-color: inherit;
        float: left;
        border: none;
        outline: none;
        cursor: pointer;
        padding: 14px 16px;
        transition: 0.3s;
        font-size: 16px;
    }
    .tab button:hover {
        background-color: #ddd;
    }
    .tab button.active {
        background-color: #005a9e;
        color: white;
    }
    .tabcontent {
        display: none;
        padding: 16px;
        border: 1px solid #ccc;
        border-top: none;
        border-radius: 0 0 4px 4px;
        animation: fadeEffect 1s;
    }
    @keyframes fadeEffect {
        from {opacity: 0;}
        to {opacity: 1;}
    }
    </style>
    <div class="tab">
      <button class="tablinks" onclick="openErrorTab(event, 'WrongLang')" id="defaultOpen">Wrong Language</button>
      <button class="tablinks" onclick="openErrorTab(event, 'SimilarLang')">Similar Language</button>
      <button class="tablinks" onclick="openErrorTab(event, 'Boundary')">Boundary Error</button>
      <button class="tablinks" onclick="openErrorTab(event, 'Missing')">Missing</button>
      <button class="tablinks" onclick="openErrorTab(event, 'Extra')">Extra</button>
    </div>
    """)
    
    # Wrong language tab
    html_lines.append('<div id="WrongLang" class="tabcontent">')
    if 'wrong_lang' in error_patterns and error_patterns['wrong_lang']:
        error_table("Top 100 wrong language errors",
            [(s, gold_l, pred_l, token)
            for s, gold_l, pred_l, token in error_patterns['wrong_lang'][:100]],
            ["Sentence", "Gold", "Pred", "Token"])
    else:
        html_lines.append("<p>No wrong language errors found.</p>")
    html_lines.append('</div>')
    
    # Similar language tab
    html_lines.append('<div id="SimilarLang" class="tabcontent">')
    if 'similar_lang' in error_patterns and error_patterns['similar_lang']:
        error_table("Top 100 similar language errors",
            [(s, gold_l, pred_l, token)
            for s, gold_l, pred_l, token in error_patterns['similar_lang'][:100]],
            ["Sentence", "Gold", "Pred", "Token"])
    else:
        html_lines.append("<p>No similar language errors found.</p>")
    html_lines.append('</div>')
    
    # Boundary error tab
    html_lines.append('<div id="Boundary" class="tabcontent">')
    if 'boundary_error' in error_patterns and error_patterns['boundary_error']:
        error_table("Top 100 boundary errors",
            [(s, gold_l, pred_l, token)
            for s, gold_l, pred_l, token in error_patterns['boundary_error'][:100]],
            ["Sentence", "Gold", "Pred", "Token"])
    else:
        html_lines.append("<p>No boundary errors found.</p>")
    html_lines.append('</div>')
    
    # Missing tab
    html_lines.append('<div id="Missing" class="tabcontent">')
    if 'missing' in error_patterns and error_patterns['missing']:
        error_table("Top 100 missing errors",
            [(s, gold_l, "", token)
            for s, gold_l, _, token in error_patterns['missing'][:100]],
            ["Sentence", "Gold", "Pred", "Token"])
    else:
        html_lines.append("<p>No missing errors found.</p>")
    html_lines.append('</div>')
    
    # Extra tab
    html_lines.append('<div id="Extra" class="tabcontent">')
    if 'extra' in error_patterns and error_patterns['extra']:
        error_table("Top 100 extra errors",
            [(s, "", pred_l, token)
            for s, _, pred_l, token in error_patterns['extra'][:100]],
            ["Sentence", "Gold", "Pred", "Token"])
    else:
        html_lines.append("<p>No extra errors found.</p>")
    html_lines.append('</div>')
    
    # Add tab JavaScript
    html_lines.append("""
    <script>
    function openErrorTab(evt, errorType) {
        var i, tabcontent, tablinks;
        tabcontent = document.getElementsByClassName("tabcontent");
        for (i = 0; i < tabcontent.length; i++) {
            tabcontent[i].style.display = "none";
        }
        tablinks = document.getElementsByClassName("tablinks");
        for (i = 0; i < tablinks.length; i++) {
            tablinks[i].className = tablinks[i].className.replace(" active", "");
        }
        document.getElementById(errorType).style.display = "block";
        evt.currentTarget.className += " active";
    }
    
    // Get the element with id="defaultOpen" and click on it
    document.getElementById("defaultOpen").click();
    </script>
    """)
    
    html_lines.append("</section>")

    # ----- Matched Sentences -----
    html_lines.append("<section><h2>Matched Sentences</h2><ul>")
    matched_samples = 0
    for sample in dataset:
        if matched_samples >= 30:  # Limit to prevent extremely large HTML files
            html_lines.append("<li>... more matched sentences omitted ...</li>")
            break
            
        sentence = sample["text"]
        in_mismatch = next((m for m in mismatches if m["text"] == sentence), None)
        if not in_mismatch:
            matched_samples += 1
            spans = "; ".join([f'{span["lang"]}: {span["text"]}' for span in sample["spans"]])
            html_lines.append(f"<li><span class='code'>{html_escape(sentence)}</span><br>Spans: {html_escape(spans)}</li>")
    html_lines.append("</ul></section>")

    # ----- Unmatched Sentences -----
    html_lines.append("<section><h2>Unmatched Sentences</h2>")
    for idx, m in enumerate(mismatches[:50], 1):  # Limit to 50 to keep HTML size manageable
        align_txt = "<br>".join([
            f"Gold({a['gold_lang']}): {a['gold_text']} | Pred({a['pred_lang']}): {a['pred_text']}"
            for a in m.get("alignment", [])
        ])
        gold = "; ".join([f'{span["lang"]}: {span["text"]}' for span in m["gold_spans"]])
        pred = "; ".join([f'{span["lang"]}: {span["text"]}' for span in m.get("predicted_spans", [])])
        html_lines.append(f"<div style='margin-bottom:1em;'><b>{idx}.</b> <span class='code'>{html_escape(m['text'])}</span><br>")
        html_lines.append(f"<b>Gold:</b> {html_escape(gold)}<br>")
        html_lines.append(f"<b>Pred:</b> {html_escape(pred)}<br>")
        html_lines.append(f"<b>Alignment:</b><br>{align_txt}<br>")
        html_lines.append(f"<b>Type:</b> {html_escape(m.get('type','unknown'))}</div>")
    html_lines.append("</section>")

    # Add interactive charts if applicable
    if y_true and y_pred and visualizations_available:
        html_lines = add_interactive_charts(html_lines, y_true, y_pred, complexity_metrics)

    html_lines.append("</body></html>")
    with open("evaluation_report.html", "w", encoding="utf-8") as f:
        f.write("\n".join(html_lines))
    print("HTML detailed report generated as evaluation_report.html")

def main():
    # Track overall execution time
    overall_start_time = time.time()

    # Support single sentence / sample-file quick path
    if _known_args.sample or _known_args.sample_file:
        if _known_args.sample:
            print('[SAMPLE]', _known_args.sample)
            print('=>', detect_languages(_known_args.sample))
        if _known_args.sample_file:
            try:
                with open(_known_args.sample_file, 'r', encoding='utf-8') as sf:
                    for line in sf:
                        line=line.strip() ;
                        if not line: continue
                        print('[SAMPLE]', line)
                        print('=>', detect_languages(line))
            except Exception as e:
                print('Error reading sample-file:', e)
        return

    # Load test data
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
    mismatches = []
    error_patterns = defaultdict(list)
    lang_error_counter = defaultdict(Counter)

    exact = 0
    partial = 0
    total = len(dataset)

    all_gold_langs = []
    all_pred_langs = []
    eval_results = []
    
    # --- MODIFICATION START: Manual Batching and Progress Bar ---
    batch_size = _known_args.batch_size
    print(f"Evaluating language detection in batches of {batch_size}...")

    # Wrap the range with tqdm for a progress bar
    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating Batches"):
        batch_dataset = dataset[i:i+batch_size]
        batch_sentences = [sample["text"] for sample in batch_dataset]

        try:
            batch_results = batch_detect_languages(batch_sentences)
        except Exception as e:
            print(f'[FATAL] batch_detect_languages failed on a batch: {e}')
            # Decide if you want to skip this batch or stop execution
            continue # This will skip the failed batch

        for sample, pred_spans_raw in zip(batch_dataset, batch_results):
            sentence = sample["text"]
            gold_spans = [(normalize(span["text"]), span["lang"]) for span in sample["spans"]]
            pred_spans = [(normalize(text), lang) for text, lang in pred_spans_raw]

            gold_set = set(gold_spans)
            pred_set = set(pred_spans)

            gold_langs = [lang for _, lang in gold_spans]
            pred_langs = [lang for _, lang in pred_spans]
            gold_texts = [text for text, _ in gold_spans]
            pred_texts = [text for text, _ in pred_spans]
            alignment = []
            min_len = min(len(gold_spans), len(pred_spans))
            for j in range(min_len):
                alignment.append({
                    "gold_text": gold_texts[j],
                    "gold_lang": gold_langs[j],
                    "pred_text": pred_texts[j],
                    "pred_lang": pred_langs[j]
                })
            for j in range(min_len, len(gold_spans)):
                alignment.append({
                    "gold_text": gold_texts[j],
                    "gold_lang": gold_langs[j],
                    "pred_text": "",
                    "pred_lang": ""
                })
            for j in range(min_len, len(pred_spans)):
                alignment.append({
                    "gold_text": "",
                    "gold_lang": "",
                    "pred_text": pred_texts[j],
                    "pred_lang": pred_langs[j]
                })

            # Use detailed error analysis
            alignment_errors = perform_detailed_error_analysis(gold_spans, pred_spans, sentence)
            for err_type, err_list in alignment_errors.items():
                for sentence, gold_l, pred_l, token in err_list:
                    lang_error_counter[gold_l or pred_l][err_type] += 1
                    error_patterns[err_type].append((sentence, gold_l, pred_l, token))

            result = {
                "text": sentence,
                "gold_spans": sample["spans"],
                "predicted_spans": [{"text": t, "lang": l} for t, l in pred_spans],
                "alignment": alignment
            }

            if pred_set == gold_set:
                exact += 1
                result["type"] = "exact"
            elif pred_set & gold_set:
                partial += 1
                result["type"] = "partial"
                mismatches.append(result)
            else:
                result["type"] = "none"
                mismatches.append(result)

            eval_results.append(result)

            if len(gold_spans) == len(pred_spans):
                y_true.extend(gold_langs)
                y_pred.extend(pred_langs)
            all_gold_langs.extend(gold_langs)
            all_pred_langs.extend(pred_langs)
    # --- MODIFICATION END ---


    # Calculate complexity metrics
    complexity_metrics = evaluate_by_complexity(dataset, eval_results)

    # Generate reports
    print("\nGenerating evaluation reports...")
    generate_summary_txt_report(
        total, exact, partial, mismatches, y_true, y_pred,
        all_gold_langs, all_pred_langs, error_patterns, lang_error_counter,
        complexity_metrics
    )
    generate_html_report(
        total, exact, partial, mismatches, y_true, y_pred,
        all_gold_langs, all_pred_langs, error_patterns, lang_error_counter, 
        dataset, complexity_metrics
    )
    
    # Print command-line summary
    print_command_line_summary(
        total, exact, partial, 
        y_true, y_pred, 
        all_gold_langs, all_pred_langs,
        lang_error_counter, 
        complexity_metrics,
        overall_start_time
    )

if __name__ == "__main__":
    main()