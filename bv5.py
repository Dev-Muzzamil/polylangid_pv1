# d.py â€” Enhanced Polyglot Language Detection (Target: F1 > 0.91)

from __future__ import annotations

import os, re, math, string, logging, unicodedata
from functools import lru_cache
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# ------------------------------
# Dependencies (graceful degrade)
# ------------------------------

missing = []
try:
    from transformers.pipelines import pipeline
except Exception:
    pipeline = None
    missing.append('transformers')

try:
    import torch
    _torch_available = True
    _torch_cuda = torch.cuda.is_available()
except Exception:
    _torch_available = False
    _torch_cuda = False
    missing.append('torch')

try:
    import fasttext
except Exception:
    fasttext = None
    missing.append('fasttext')

try:
    import jieba
    _jieba_available = True
except Exception:
    jieba = None
    _jieba_available = False
    missing.append('jieba')

try:
    from janome.tokenizer import Tokenizer as JanomeTokenizer
    _janome = JanomeTokenizer()
    _janome_available = True
except Exception:
    _janome_available = False
    _janome = None
    missing.append('janome')

try:
    from pythainlp.tokenize import word_tokenize as thai_tokenize
    _thai_available = True
except Exception:
    _thai_available = False
    thai_tokenize = None
    missing.append('pythainlp')

# Vietnamese tokenizer (pyvi)
try:
    from pyvi import ViTokenizer as _ViTokenizer
    def vi_tokenize_to_list(text: str) -> List[str]:
        try:
            return [x for x in _ViTokenizer.tokenize(text).split() if x.strip()]
        except Exception:
            return []
    _vi_tokenizer_available = True
except Exception:
    _vi_tokenizer_available = False
    vi_tokenize_to_list = None # type: ignore
    missing.append('pyvi')

# Indonesian stemmer (Sastrawi)
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    _id_stemmer_factory = StemmerFactory()
    _id_stemmer = _id_stemmer_factory.create_stemmer()
    _id_stemmer_available = True
except Exception:
    _id_stemmer_available = False
    _id_stemmer = None
    missing.append('Sastrawi')

logger = logging.getLogger("D1_EnhancedPolyLangID")
logging.basicConfig(level=logging.INFO)

if missing and len(missing) > 2:
    logger.warning(f"Missing dependencies: {missing}")

# ------------------------------
# Config
# ------------------------------

TOP_20_LANGS = {
    'en','zh','hi','es','fr','ar','bn','pt','ru','ur',
    'id','de','ja','tr','ko','it','th','vi','pl','nl'
}

TRANSFORMER_MODEL = os.environ.get("POLYLANGID_XLMR_PATH", os.path.join(os.path.dirname(__file__), "models", "xlmr"))
FASTTEXT_PATH_DEFAULT = os.environ.get("POLYLANGID_FASTTEXT_PATH","lid.176.ftz")
FASTTEXT_FALLBACK_PATH = os.environ.get("POLYLANGID_FASTTEXT_FALLBACK","")
TRANSFORMER_FP16 = True
TRANSFORMER_BATCH_SIZE = 64 if _torch_cuda else 16
FASTTEXT_TOP_K = 5
FASTTEXT_TOP_K_SHORT = 3

# Enhanced smoothing parameters
SWITCH_PENALTY = 0.15
MIN_LANG_SCORE = 1e-6
SHORT_TOKEN_MAX_LEN = 2
SHORT_SWITCH_EXTRA = 0.18
SHORT_NO_PENALTY_SCRIPTS = {"HAN","HIRAGANA","KATAKANA","THAI","HANGUL"}

UNKNOWN_RATIO_FALLBACK = 0.65
UNKNOWN_MIN_PROB = 0.03
UNKNOWN_INJECT_MAXP_THRESHOLD = 0.25
CANDIDATE_KEEP_THRESHOLD = 0.015
UNKNOWN_NEIGHBOR_FILL_THRESHOLD = 0.55

# Primary / canonical scripts for mismatch penalization
LANG_PRIMARY_SCRIPT = {
    'hi':'DEVANAGARI','bn':'BENGALI','ar':'ARABIC','ur':'ARABIC','zh':'HAN','ja':'HAN',
    'ko':'HANGUL','th':'THAI','ru':'CYRILLIC'
}

SCRIPT_MISMATCH_PENALTY = 0.28
SCRIPT_MISMATCH_LEN_THRESHOLD = 3

# ------------------------------
# Enhanced Patterns and Lexicons (Shifted from Word Lists to Patterns)
# ------------------------------

LANGUAGE_PATTERNS = {
    "en": [
        r'\b(the|and|that|have|you|this|with|from|they|been|which|their|will|would|could|should)\b',
        r"\b(I'm|you're|we're|they're|he's|she's|it's|don't|doesn't|didn't|can't|won't)\b",
        r'\b\w+ing\b', r'\b\w+ed\b', r'\b\w+ly\b', r'\b\w+tion\b', r'\b\w+ness\b'
    ],
    "fr": [
        r'\b(le|la|les|un|une|des|et|de|du|dans|pour|avec|qui|que|ce|cette|ces|mais|ou|oÃ¹|sur)\b',
        r'\b\w*ment\b', r'\b\w*tion\b', r'\b\w*ique\b', r'\b\w*Ã©e\b', r'[Ã©Ã¨ÃªÃ«Ã Ã¢Ã¤Ã´Ã¶Ã¹Ã»Ã¼Ã®Ã¯]'
    ],
    "de": [
        r'\b(der|die|das|und|in|von|zu|den|mit|sich|auf|fÃ¼r|ist|im|dem|nicht|ein|eine|als|auch)\b',
        r'\b\w*heit\b', r'\b\w*keit\b', r'\b\w*ung\b', r'\b\w*lich\b', r'\bge\w+t\b', r'sch', r'[Ã¤Ã¶Ã¼ÃŸ]'
    ],
    "es": [
        r'\b(el|la|los|las|un|una|y|de|en|es|con|por|que|no|se|le|lo|me|te|su|sus)\b',
        r'\b\w*ciÃ³n\b', r'\b\w*mente\b', r'\b\w*dad\b', r'\b\w+ado\b', r'\b\w+ido\b'
    ],
    "it": [
        r'\b(il|la|lo|gli|le|di|da|in|con|su|per|tra|fra|a|e|ma|o|se|che)\b',
        r'\b\w*zione\b', r'\b\w*mente\b', r'\b\w*tÃ \b', r'\b\w+ato\b', r'\b\w+ito\b'
    ],
    "pt": [
        r'\b(o|a|os|as|um|uma|e|de|em|Ã©|com|por|que|nÃ£o|se|do|da|dos|das|no|na)\b',
        r'\b\w*Ã§Ã£o\b', r'\b\w*mente\b', r'\b\w*dade\b', r'[Ã£Ãµ]', r'nh', r'lh'
    ],
    "id": [
        r'\b(yang|dan|di|ke|dari|untuk|pada|dengan|adalah|ini|itu|akan|sudah)\b',
        r'^ber\w+', r'^me\w+', r'^pe\w+', r'^ter\w+', r'^se\w+',
        r'\w+kan\b', r'\w+nya\b', r'\w+lah\b', r'^ke\w+an\b'
    ],
    "ru": [r'\b(Ñ |Ð¸|Ð²|Ð½Ðµ|Ñ‡Ñ‚Ð¾|Ð¾Ð½|ÐºÐ°Ðº|Ñ Ñ‚Ð¾)\b', r'\w+Ð¾Ð²\b', r'\w+Ð¸Ð¹\b', r'\w+Ð°Ñ \b'],
    "pl": [r'\b(i|w|na|nie|jest|siÄ™|to|że|jak)\b', r'\w+Ä‡\b', r'[Å„Å›Ä‡ÅºÅ¼]'],
    "nl": [r'\b(de|en|van|ik|te|dat|die|in|een|hij|het|niet|zijn|is|was)\b', r'\w+lijk\b', r'\w+heid\b', r'ij', r'cht'],
    "tr": [
        r'\b(ve|bir|bu|bu|o|ben|sen|iÃ§in|de|ile)\b',
        r'\w+[aeÄ±i]yor\b', r'\w+m[Ä±i]ÅŸ\b', r'\w+lar\b', r'\w+ler\b', r'[ÄŸÄ±ÅŸÃ§Ã¼Ã¶]'
    ],
    "hi": [r'à¤¹à¥ˆ', r'à¤•à¤°', r'à¤¸à¥‡', r'à¤•à¥€', r'à¤®à¥‡à¤‚', r'à¤¨à¤¹à¥€à¤‚', r'à¤¹à¥‹', r'à¤¹à¥ˆà¤‚'],
    "ar": [r'\bØ§Ù„\w+', r'Ù ÙŠ', r'Ù…Ù†', r'Ø¹Ù„Ù‰', r'Ù„Ø§', r'ÙƒØ§Ù†'],
    "ur": [r'Û Û’', r'Ú©Û’', r'Ú©ÛŒ', r'Ú©Ø§', r'Ø³Û’', r'Ù†Û’', r'Û ÙˆÚº'],
    "zh": [r'çš„', r'äº†', r'åœ¨', r'æ˜¯', r'æˆ‘', r'æœ‰', r'å’Œ', r'ä¸ '],
    "ja": [r'ã §ã ™', r'ã ¾ã ™', r'ã —ã Ÿ', r'ã “ã ®', r'ã  ã ®', r'ã ¨', r'ã «', r'ã‚’', r'ã ¯'],
    "ko": [r'ìž…ë‹ˆë‹¤', r'í•©ë‹ˆë‹¤', r'ìŠµë‹ˆë‹¤', r'ì–´ìš”', r'ì„œìš”', r'ì ˜', r'ëŠ”', r'ì €'],
    "th": [r'à¸„à¸£à¸±à¸š', r'à¸„à¹ˆà¸°', r'à¹„à¸¡à¹ˆ', r'à¹ƒà¸™', r'à¹ à¸¥à¸°', r'à¸—à¸µà¹ˆ', r'à¸ à¸±à¸š'],
    "vi": [r'\bcá»§a\b', r'\bvÃ \b', r'\blÃ \b', r'\bcho\b', r'\btrong\b', r'\bÄ‘Ã£\b', r'\w+hÃ³a\b'],
    "bn": [r'à¦ à¦°', r'à¦ à¦¬à¦‚', r'à¦•à¦°à§‡', r'à¦¹à¦¯à¦¼', r'à¦¨à¦¾', r'à¦œà¦¨à§ à¦¯']
}

CHARACTER_PATTERNS = {
    'de': ['Ã¤','Ã¶','Ã¼','ÃŸ'],
    'fr': ['Ã§','Ã©','Ã¨','Ãª','Ã ','Ã¹','Ã´','Ã¢','Ã®','Å“','Ã¯'],
    'es': ['Ã±','Ã­','Ã³','Ã¡','Ã©','Ãº','Ã¼'],
    'pt': ['Ã£','Ãµ','Ã§','Ã ','Ã¡','Ã¢','Ã©','Ãª','Ã­','Ã³','Ã´','Ãº'],
    'it': ['Ã ','Ã¨','Ã©','Ã¬','Ã²','Ã¹'],
    'tr': ['ÄŸ','Ä±','ÅŸ','Ã§','Ã¼','Ã¶'],
    'pl': ['Ä…','Ä‡','Ä™','Å‚','Å„','Ã³','Å›','Åº','Å¼'],
    'nl': ['ij','oe','eu','aa','ee','oo','uu'],
    'vi': ['Äƒ','Ã¢','Ãª','Ã´','Æ¡','Æ°','Ä‘']
}

# Enhanced HAN character hints for zh/ja disambiguation
SIMP_ONLY_CHARS = set("è‰ºæœ¯çˆ±ä¼˜çŽ°ä¹¦åº†é—®è§‚è ”å¹¿äº§ä¼—è®¯ç”µè½¦é—¨é—»åŒ»æ°”"
                      "å›½ä½“å­¦ä¸“å®žå ·ä¼šå…šæ ¥ä¸Žä¸ºä¹¡å¯¼å¯¹å°†å¹´å¾—å¿…")
TRAD_BIAS_CHARS = set("è— è¡“æ„›å„ªç ¾æ›¸è§€è ¯å»£é–€é†«æ°£åœ‹é«”å­¸å°ˆé œå¯§é§…æ™‚å††è¦‹æ›œ")
JP_SPECIFIC_CHARS = set("å††é§…æ™‚æ›œè¦‹åƒ ç•‘ç•‘è¾¼")

PERFECT_SCRIPT_MAP = {
    "BENGALI":"bn", "HIRAGANA":"ja", "KATAKANA":"ja",
    "HANGUL":"ko", "THAI":"th", "DEVANAGARI":"hi"
}

SCRIPT_LANG_MAP = {
    "LATIN": ["en","fr","de","es","it","pt","nl","pl","tr","vi","id"],
    "CYRILLIC": ["ru"],
    "ARABIC": ["ar","ur"],
    "DEVANAGARI": ["hi"],
    "HAN": ["zh","ja"],
    "HIRAGANA": ["ja"],
    "KATAKANA": ["ja"],
    "HANGUL": ["ko"],
    "THAI": ["th"],
    "BENGALI": ["bn"]
}

# Vietnamese helpers
VI_DIACRITICS = set("ÄƒÃ¢ÃªÃ´Æ¡Æ°Ä‚Ã‚ÃŠÃ”Æ Æ¯Ä‘Ä ")
VI_ACCENTED = (set("aÃ Ã¡áº£Ã£áº¡Äƒáº±áº¯áº³áºµáº·Ã¢áº§áº¥áº©áº«áº­eÃ¨Ã©áº»áº½áº¸Ãªá» áº¿á»ƒá»…á»‡iÃ¬Ã­á»‰Ä©á»‹oÃ²Ã³á» Ãµá» Ã´á»“á»‘á»•á»—á»™Æ¡á» á»›á»Ÿá»¡á»£uÃ¹Ãºá»§Å©á»¥Æ°á»«á»©á»­á»¯á»±yá»³Ã½á»·á»¹á»µ")
    | set("AÃ€Ã áº¢Ãƒáº Ä‚áº°áº®áº²áº´áº¶Ã‚áº¦áº¤áº¨áºªáº¬EÃˆÃ‰áººáº¼áº¸ÃŠá»€áº¾á»‚á»„á»†IÃŒÃ á»ˆÄ¨á»ŠOÃ’Ã“á»ŽÃ•á»ŒÃ”á»’á» á»”á»–á»˜Æ á»œá»šá»žá» á»¢UÃ™Ãšá»¦Å¨á»¤Æ¯á»ªá»¨á»¬á»®á»°Yá»²Ã á»¶á»¸á»´")
    | set("Ä‘Ä ")) - set("aeiouyAEIOUY")

VI_SYLLABLE_REGEX = re.compile(r"[bcdfghjklmnpqrstvwxyzÄ‘]*[aÃ Ã¡áº£Ã£áº¡Äƒáº±áº¯áº³áºµáº·Ã¢áº§áº¥áº©áº«áº­eÃ¨Ã©áº»áº½áº¸Ãªá» áº¿á»ƒá»…á»‡iÃ¬Ã­á»‰Ä©á»‹oÃ²Ã³á» Ãµá» Ã´á»“á»‘á»•á»—á»™Æ¡á» á»›á»Ÿá»¡á»£uÃ¹Ãºá»§Å©á»¥Æ°á»«á»©á»­á»¯á»±yá»³Ã½á»·á»¹á»µ]+[bcdfghjklmnpqrstvwxyzÄ‘]*", re.IGNORECASE)

# Whitelist expanded from boundary error analysis in the new report
VI_COMPOUND_WHITELIST = {
    "tÃ¢mtrÃ­","giáº¥cmÆ¡","tÆ°Æ¡nglai","lÃ²ngdÅ©ngcáº£m","giaÄ‘Ã¬nh","nghá»‡thuáº­t","Ä‘áº¡idÆ°Æ¡ng","thiÃªnnhiÃªn","Ã¢mnháº¡c",
    "sá»±imláº·ng","trÃ­tuá»‡","sá»±thanhlá»‹ch","tá»±do","ngÃ´isao","bÃ³ngtá»‘i","cá»­asá»•","hÃ²abÃ¬nh","hyvá» ng","Ã¡nhsÃ¡ng",
    "bá» nvá»¯ng", "nghiÃªncá»©u", "lÃ½tÆ°á»Ÿng", "quÃ½giÃ¡", "trÃ¡chnhiá»‡m", "tÆ°duy", "hiá»‡nÄ‘áº¡ihÃ³a",
    "cÃ´ngnghá»‡sinhhá» c", "xÃ£há»™i", "chá»§nghÄ©aduyváº­t", "tÃ¢mlinh", "tÃ¢mlÃ½", "cÃ¡chmáº¡ng",
    "giÃ¡trá»‹luáº­n", "tÃ­nhtoÃ¡n", "chÃ­nhtrá»‹", "ká»¹nÄƒng", "triáº¿thá» c", "mÃ´itrÆ°á» ng", "disáº£n",
    "kinhnghiá»‡m", "toÃ ncáº§uhÃ³a", "giÃ¡odá»¥c", "vÄƒnhá» c", "vÄƒnminh", "tÆ°Æ¡nglai", "cáº¥utrÃºc",
    "hạnhphÃºc", "hyvọng", "lÃ½tưởng", "kỹnăng", "tÃ­nhtoÃ¡n", "hiệnđại", "hiệnđạihÃ³a", "giÃ¡trị"
}


VI_VOWELS = set("aÃ Ã¡áº£Ã£áº¡Äƒáº±áº¯áº³áºµáº·Ã¢áº§áº¥áº©áº«áº­eÃ¨Ã©áº»áº½áº¸Ãªá» áº¿á»ƒá»…á»‡iÃ¬Ã­á»‰Ä©á»‹oÃ²Ã³á» Ãµá» Ã´á»“á»‘á»•á»—á»™Æ¡á» á»›á»Ÿá»¡á»£uÃ¹Ãºá»§Å©á»¥Æ°á»«á»©á»­á»¯á»±yá»³Ã½á»·á»¹á»µAÃ€Ã áº¢Ãƒáº Ä‚áº°áº®áº²áº´áº¶Ã‚áº¦áº¤áº¨áºªáº¬EÃˆÃ‰áººáº¼áº¸ÃŠá»€áº¾á»‚á»„á»†IÃŒÃ á»ˆÄ¨á»ŠOÃ’Ã“á»ŽÃ•á»ŒÃ”á»’á» á»”á»–á»˜Æ á»œá»šá»žá» á»¢UÃ™Ãšá»¦Å¨á»¤Æ¯á»ªá»¨á»¬á»®á»°Yá»²Ã á»¶á»¸á»´")
VI_ONSETS = ["ngh","ng","gh","kh","th","nh","ph","tr","ch","qu","gi","b","c","d","Ä‘","g","h","k","l","m","n","p","q","r","s","t","v","x"]
VI_CODAS = ["nh","ng","ch","c","m","n","p","t"]

# Enhanced Arabic/Urdu disambiguation
UR_SPECIFIC_CHARS = set("ÛŒÚ¯Ù¾Ú†Ú˜Ú‘ÚºÛ’")
AR_SPECIFIC_CHARS = set("Ø¸Ø¶Øº")

# ------------------------------
# Utilities
# ------------------------------

@lru_cache(maxsize=8192)
def get_script(ch: str) -> str:
    try:
        return unicodedata.name(ch).split(' ')[0]
    except Exception:
        return "UNKNOWN"

def dominant_script(token: str) -> Optional[str]:
    counts = Counter()
    for ch in token:
        if ch.isalpha():
            counts[get_script(ch)] += 1
    return counts.most_common(1)[0][0] if counts else None

def is_short_token(token: str) -> bool:
    return len(token) <= SHORT_TOKEN_MAX_LEN

def english_pattern_match_count(text: str) -> int:
    patterns = [
        r'\b(the|and|that|have|you|this|with|from|they|been|which|their)\b',
        r'\b\w+ing\b', r'\b\w+ed\b', r'\b\w+ly\b'
    ]
    lower = text.lower()
    return sum(1 for p in patterns if re.search(p, lower))

def char_pattern_score(token_lower: str) -> Dict[str, float]:
    scores = {}
    for lang, chars in CHARACTER_PATTERNS.items():
        cnt = sum(1 for c in chars if c in token_lower)
        if cnt:
            scores[lang] = float(cnt)
    return scores

def pattern_hint_scores(token_lower: str) -> Dict[str, float]:
    scores = {}
    for lang, pats in LANGUAGE_PATTERNS.items():
        m = 0
        for pat in pats:
            if re.search(pat, token_lower):
                m += 1
        if m:
            scores[lang] = 1.0 - (0.6 ** m)
    return scores

def script_candidate_score(token: str) -> Dict[str, float]:
    sc = dominant_script(token)
    if not sc:
        return {}

    sc_up = sc.upper()
    if sc_up in PERFECT_SCRIPT_MAP:
        lang = PERFECT_SCRIPT_MAP[sc_up]
        if lang in TOP_20_LANGS:
            return {lang: 1.0}

    if sc_up in SCRIPT_LANG_MAP:
        base_prior = 0.15
        length_factor = min(len(token)/8.0, 1.0)
        adjusted = base_prior * (0.5 + 0.5*length_factor)
        return {l: adjusted for l in SCRIPT_LANG_MAP[sc_up] if l in TOP_20_LANGS}

    return {}

# Enhanced tokenization (keeping best parts from b.py)
@lru_cache(maxsize=4096)
def _char_script(ch: str) -> str:
    if not ch.isalpha():
        return 'OTHER'
    try:
        return unicodedata.name(ch).split(' ')[0]
    except Exception:
        return 'OTHER'

def _flush(buf: List[str], out: List[str]):
    if not buf:
        return
    seg = ''.join(buf)
    if seg.strip():
        out.append(seg)
    buf.clear()

def _segment_by_script(text: str) -> List[str]:
    out, buf = [], []
    prev_script = None

    for ch in text:
        if ch.isspace():
            _flush(buf, out)
            prev_script = None
            continue

        cat = unicodedata.category(ch)
        if cat.startswith('M'):
            buf.append(ch)
            continue

        if cat.startswith('P') or cat.startswith('S'):
            _flush(buf, out)
            if ch.strip():
                out.append(ch)
            prev_script = None
            continue

        sc = _char_script(ch)
        if prev_script is None or sc == prev_script:
            buf.append(ch)
            prev_script = sc
        else:
            _flush(buf, out)
            buf.append(ch)
            prev_script = sc

    _flush(buf, out)
    return out

def _merge_short_fragments(tokens: List[str]) -> List[str]:
    merged = []
    i = 0

    while i < len(tokens):
        tok = tokens[i]
        sc = dominant_script(tok)

        if sc and sc.upper() in ['DEVANAGARI','BENGALI'] and len(tok) <= 2:
            j = i+1
            accum = tok

            while j < len(tokens):
                nxt = tokens[j]
                sc2 = dominant_script(nxt)
                if sc2 and sc2.upper()==sc.upper() and len(nxt)<=3 and len(accum)+len(nxt)<=8:
                    accum += nxt
                    j += 1
                else:
                    break

            merged.append(accum)
            i = j
        else:
            merged.append(tok)
            i += 1

    return merged

# Enhanced Vietnamese compound splitting
def _find_vi_boundaries(w: str) -> List[int]:
    wl = w.lower()
    candidates: List[Tuple[int,int]] = []

    for i in range(1, len(wl)):
        onset = None
        for on in VI_ONSETS:
            j = i + len(on)
            if wl.startswith(on, i) and j < len(wl) and wl[j] in VI_VOWELS:
                onset = on
                break

        if not onset:
            continue

        # Check for valid syllable structure
        if not any(ch in VI_VOWELS for ch in wl[:i]):
            continue

        last2 = wl[i-2:i]
        last1 = wl[i-1:i]
        valid_coda = (last2 in VI_CODAS) or (last1 in VI_CODAS) or (wl[i-1] in VI_VOWELS)

        if not valid_coda:
            continue

        candidates.append((i, len(onset)))

    # Clean up adjacent boundaries
    candidates.sort()
    cleaned: List[int] = []
    prev_i: Optional[int] = None
    prev_len = 0

    for i, on_len in candidates:
        if prev_i is not None and i == prev_i + 1 and prev_len >= 2:
            continue
        cleaned.append(i)
        prev_i, prev_len = i, on_len

    return cleaned

def split_vietnamese_concatenations(tokens: List[str]) -> List[str]:
    out: List[str] = []

    for idx, t in enumerate(tokens):
        if (4 <= len(t) <= 40 and t.isalpha() and
            dominant_script(t) == 'LATIN' and
            any(ch in VI_ACCENTED for ch in t)):

            tl = t.lower()
            if tl in VI_COMPOUND_WHITELIST:
                out.append(t)
                continue

            diac_cnt = sum(1 for ch in t if ch in VI_ACCENTED)
            left = tokens[idx-1] if idx > 0 else None
            right = tokens[idx+1] if idx+1 < len(tokens) else None

            left_vi = bool(left and any(ch in VI_ACCENTED for ch in left))
            right_vi = bool(right and any(ch in VI_ACCENTED for ch in right))

            bounds = _find_vi_boundaries(t)

            if (bounds and
                (left_vi or right_vi) and
                not t.isupper() and
                len(t) >= 14):

                parts: List[str] = []
                prev = 0
                for bnd in bounds:
                    parts.append(t[prev:bnd])
                    prev = bnd
                parts.append(t[prev:])

                if all(p.strip() for p in parts):
                    out.extend(parts)
                    continue

        out.append(t)

    return out


def split_indonesian_concatenations(tokens: List[str]) -> List[str]:
    # This function relies on a comprehensive root list which has been removed.
    # Stubbing it out to avoid errors, but it will no longer function.
    return tokens


def tokenize(text: str) -> List[str]:
    if not text or not text.strip():
        return []

    t = unicodedata.normalize('NFC', text)
    segs = _segment_by_script(t)
    tokens: List[str] = []

    def _has_kana_context(idx: int) -> bool:
        for j in range(max(0, idx-2), min(len(segs), idx+3)):
            if j == idx:
                if any(get_script(c) in ('HIRAGANA','KATAKANA') for c in segs[j]):
                    return True
                continue
            s = segs[j]
            if any(get_script(c) in ('HIRAGANA','KATAKANA') for c in s):
                return True
        return False

    for idx, seg in enumerate(segs):
        if not seg.strip():
            continue

        sc = _char_script(seg[0])

        # Japanese: Janome for kana segments
        if (sc in ('HIRAGANA','KATAKANA') and
            _janome_available and _janome):
            try:
                # FIX: Use tok.surface to get only the token text, not the full POS string.
                jtoks = [tok.surface for tok in _janome.tokenize(seg, stream=True) if tok.surface.strip()]
                tokens.extend(jtoks if len(jtoks) > 1 else [seg])
                continue
            except Exception:
                pass

        # HAN: prefer Janome with kana context else jieba
        if sc == 'HAN':
            used = False
            if _janome_available and _janome and _has_kana_context(idx):
                try:
                    # FIX: Use tok.surface to get only the token text.
                    jtoks = [tok.surface for tok in _janome.tokenize(seg, stream=True) if tok.surface.strip()]
                    if jtoks:
                        tokens.extend(jtoks)
                        used = True
                except Exception:
                    used = False

            if not used and _jieba_available and jieba:
                try:
                    tokens.extend([x for x in jieba.lcut(seg) if x.strip()])
                    continue
                except Exception:
                    pass

            if used:
                continue

        # Thai - improved fallback when pythainlp not available
        if sc == 'THAI':
            if _thai_available and thai_tokenize:
                try:
                    th = [x for x in thai_tokenize(seg) if x.strip()]
                    tokens.extend(th if len(th) > 1 else [seg])
                    continue
                except Exception:
                    pass

            # Fallback: treat as single unit to avoid wrong splits
            # Thai doesn't use spaces between words, so splitting by punctuation/whitespace only
            thai_tokens = re.findall(r'[^\s\.,;:!?\'"()]+|[\s\.,;:!?\'"()]+', seg)
            if thai_tokens:
                tokens.extend([t for t in thai_tokens if t.strip()])
            else:
                tokens.append(seg)
            continue

        # Vietnamese (pyvi)
        if (sc == 'LATIN' and any(ch in VI_DIACRITICS for ch in seg) and
            _vi_tokenizer_available and vi_tokenize_to_list):
            try:
                vi_toks = vi_tokenize_to_list(seg)
                if len(vi_toks) > 1:
                    tokens.extend(vi_toks)
                    continue
            except Exception:
                pass

        # Indonesian stemmer to surface roots (no longer used as root list is removed)
        # Keeping Sastrawi dependency check for now, but stemming logic is commented out
        # if (sc == 'LATIN' and _id_stemmer_available and _id_stemmer and
        #     len(seg) > 5):
        #     pass # Stemming relied on the removed ID_COMPREHENSIVE_ROOTS

        # Indic
        if sc == 'DEVANAGARI':
            tokens.extend(re.findall(r'[\u0900-\u097F]+', seg))
            continue
        if sc == 'BENGALI':
            tokens.extend(re.findall(r'[\u0980-\u09FF]+', seg))
            continue

        # Fallback Vietnamese heuristic
        if sc == 'LATIN' and any(ch in VI_DIACRITICS for ch in seg):
            vi_tokens = re.findall(r"[A-Za-zÄ‚Ã‚Ä ÃŠÃ”Æ Æ¯ÄƒÃ¢Ä‘ÃªÃ´Æ¡Æ°]+", seg)
            if len(vi_tokens) >= 2:
                tokens.extend(vi_tokens)
                continue

        # Default words - but handle Thai specially
        if sc == 'THAI':
            # For Thai, don't split - keep the whole segment as one token
            if seg.strip():
                tokens.append(seg.strip())
        else:
            tokens.extend(re.findall(r"[\w']+", seg))

    # Split very long tokens - but preserve Thai integrity
    final_tokens = []
    for tok in tokens:
        # Check if it's Thai first
        if dominant_script(tok) == 'THAI':
            final_tokens.append(tok)
        elif len(tok) > 20 and not re.search(r'\s', tok):
            split = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?![a-z])|[\u0900-\u097F]+|[\u0980-\u09FF]+|[\u4E00-\u9FFF]+', tok)
            final_tokens.extend(split if split else [tok])
        else:
            final_tokens.append(tok)

    cleaned = [t for t in final_tokens if t.strip()]
    merged = _merge_short_fragments(cleaned)
    merged = split_vietnamese_concatenations(merged)
    merged = split_indonesian_concatenations(merged)

    return merged

# ------------------------------
# Model Manager (from b.py)
# ------------------------------

class ModelManager:
    def __init__(self, enable_transformer: bool=True, fasttext_path: str=FASTTEXT_PATH_DEFAULT):
        self.transformer = None
        self.fasttext = None
        self._ft_cache: Dict[Tuple[str, Optional[str]], Dict[str,float]] = {}

        self.enable_transformer = enable_transformer and (pipeline is not None)
        if self.enable_transformer and pipeline is not None:
            try:
                device = 0 if (_torch_available and _torch_cuda) else -1
                model_path = TRANSFORMER_MODEL
                # If the path is a directory and contains model files, use it as local model
                if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "config.json")):
                    logger.info(f"Loading XLM-R model from local directory: {model_path}")
                    if _torch_available and _torch_cuda and TRANSFORMER_FP16:
                        import torch as _torch_mod
                        self.transformer = pipeline("text-classification", model=model_path, device=device, return_all_scores=True, torch_dtype=_torch_mod.float16)
                    else:
                        self.transformer = pipeline("text-classification", model=model_path, device=device, return_all_scores=True)
                else:
                    logger.info(f"Loading XLM-R model from HuggingFace hub: {model_path}")
                    if _torch_available and _torch_cuda and TRANSFORMER_FP16:
                        import torch as _torch_mod
                        self.transformer = pipeline("text-classification", model=model_path, device=device, return_all_scores=True, torch_dtype=_torch_mod.float16)
                    else:
                        self.transformer = pipeline("text-classification", model=model_path, device=device, return_all_scores=True)
                logger.info("Transformer loaded")
            except Exception as e:
                logger.warning(f"Transformer load failed: {e}")
                self.transformer = None
                self.enable_transformer = False

        if fasttext is not None:
            try:
                if os.path.exists(fasttext_path):
                    self.fasttext = fasttext.load_model(fasttext_path)
                    logger.info(f"fastText model loaded: {fasttext_path}")
                elif FASTTEXT_FALLBACK_PATH and os.path.exists(FASTTEXT_FALLBACK_PATH):
                    self.fasttext = fasttext.load_model(FASTTEXT_FALLBACK_PATH)
                    logger.info(f"fastText fallback loaded: {FASTTEXT_FALLBACK_PATH}")
                else:
                    logger.warning(f"fastText model not found: {fasttext_path}")
            except Exception as e:
                logger.warning(f"fastText load failed: {e}")

    def transformer_probs(self, tokens: List[str]) -> List[Dict[str,float]]:
        if not self.transformer or not tokens:
            return [{} for _ in tokens]

        results: List[Dict[str,float]] = []
        bs = TRANSFORMER_BATCH_SIZE

        for i in range(0, len(tokens), bs):
            batch = tokens[i:i+bs]
            try:
                if _torch_available and _torch_cuda:
                    torch_mod = __import__('torch')
                    with torch_mod.inference_mode():
                        outs = self.transformer(batch)
                else:
                    outs = self.transformer(batch)
            except Exception:
                outs = [[] for _ in batch]

            for out in outs:
                dist: Dict[str,float] = {}
                if isinstance(out, list):
                    total = 0.0
                    for entry in out:
                        if isinstance(entry, dict):
                            lab = entry.get('label','').lower()
                            sc = float(entry.get('score',0.0))
                            if lab in TOP_20_LANGS:
                                dist[lab] = sc
                                total += sc

                    if total > 0:
                        inv = 1.0/total
                        for k in list(dist.keys()):
                            dist[k] *= inv

                results.append(dist)

        return results

    def fasttext_probs_batch(self, tokens: List[str]) -> List[Dict[str,float]]:
        if not self.fasttext or not tokens:
            return [{} for _ in tokens]

        results: List[Dict[str,float]] = []

        for token in tokens:
            key = (token, dominant_script(token))
            if key in self._ft_cache:
                results.append(self._ft_cache[key])
                continue

            try:
                k = FASTTEXT_TOP_K_SHORT if len(token) <= SHORT_TOKEN_MAX_LEN else FASTTEXT_TOP_K
                sc = dominant_script(token)
                if sc and sc.upper() in ['DEVANAGARI','BENGALI','THAI','HAN','HIRAGANA','KATAKANA']:
                    k = min(k+3, 10)

                labels, probs = self.fasttext.predict(token, k=k)
                dist: Dict[str,float] = {}
                total = 0.0

                for lab, pr in zip(labels, probs):
                    lang = lab.replace("__label__","")
                    if lang in TOP_20_LANGS:
                        dist[lang] = float(pr)
                        total += float(pr)

                if total > 0:
                    inv = 1.0/total
                    for k2 in list(dist.keys()):
                        dist[k2] *= inv

                self._ft_cache[key] = dist
                results.append(dist)
            except Exception:
                results.append({})

        return results

# ------------------------------
# Enhanced Core Detector
# ------------------------------

class EnhancedDetector:
    def __init__(self, enable_transformer: bool=True, fasttext_path: str=FASTTEXT_PATH_DEFAULT):
        self.model_mgr = ModelManager(enable_transformer, fasttext_path)
        self._token_cache: Dict[str, Dict[str,float]] = {}

        self.debug_counters = {
            'id_boost': 0,
            'ja_han_force': 0,
            'problematic_word_fix': 0,
            'enhanced_disambiguation': 0,
        }

    def _dynamic_weights(self, token: str) -> Dict[str,float]:
        length = max(len(token), 1)
        script = dominant_script(token)
        sc_up = script.upper() if script else ""

        has_transformer = self.model_mgr.enable_transformer and self.model_mgr.transformer is not None

        if sc_up == "LATIN":
            if has_transformer:
                if length <= 2:
                    return {"transformer": 0.15, "fasttext": 0.38, "pattern": 0.22, "script": 0.15, "char": 0.10}
                elif length <= 4:
                    return {"transformer": 0.30, "fasttext": 0.42, "pattern": 0.18, "script": 0.08, "char": 0.02}
                else:
                    return {"transformer": 0.45, "fasttext": 0.38, "pattern": 0.12, "script": 0.03, "char": 0.02}
            else:
                if length <= 2:
                    return {"transformer": 0.0, "fasttext": 0.30, "pattern": 0.40, "script": 0.20, "char": 0.10}
                elif length <= 4:
                    return {"transformer": 0.0, "fasttext": 0.40, "pattern": 0.35, "script": 0.20, "char": 0.05}
                else:
                    return {"transformer": 0.0, "fasttext": 0.45, "pattern": 0.35, "script": 0.15, "char": 0.05}
        else:
            if has_transformer:
                if length <= 2:
                    return {"transformer": 0.15, "fasttext": 0.25, "pattern": 0.25, "script": 0.25, "char": 0.10}
                elif length <= 4:
                    return {"transformer": 0.25, "fasttext": 0.30, "pattern": 0.20, "script": 0.20, "char": 0.05}
                else:
                    return {"transformer": 0.40, "fasttext": 0.30, "pattern": 0.15, "script": 0.12, "char": 0.03}
            else:
                if length <= 2:
                    return {"transformer": 0.0, "fasttext": 0.35, "pattern": 0.30, "script": 0.25, "char": 0.10}
                elif length <= 4:
                    return {"transformer": 0.0, "fasttext": 0.40, "pattern": 0.30, "script": 0.25, "char": 0.05}
                else:
                    return {"transformer": 0.0, "fasttext": 0.45, "pattern": 0.30, "script": 0.20, "char": 0.05}

    def _fuse(self, token: str, t_probs: Dict[str,float], f_probs: Dict[str,float],
              patt: Dict[str,float], scp: Dict[str,float], ch: Dict[str,float]) -> Dict[str,float]:

        cands = set(t_probs) | set(f_probs) | set(patt) | set(scp) | set(ch)
        cands = [c for c in cands if c in TOP_20_LANGS]

        if not cands:
            return {}

        w = self._dynamic_weights(token)
        fused: Dict[str,float] = {}

        for lang in cands:
            s = 0.0
            s += w["transformer"]*t_probs.get(lang,0.0)
            s += w["fasttext"]*f_probs.get(lang,0.0)
            s += w["pattern"]*patt.get(lang,0.0)
            s += w["script"]*scp.get(lang,0.0)
            s += w["char"]*ch.get(lang,0.0)

            if s > 0:
                fused[lang] = s

        # Model agreement boost
        for lang in ["en","id","zh","ja","hi","ar","vi"]:
            if t_probs.get(lang,0) > 0.4 and f_probs.get(lang,0) > 0.4:
                fused[lang] = min(0.95, fused.get(lang,0)+0.1)

        # Normalize
        total = sum(fused.values())
        if total > 0:
            inv = 1.0/total
            for k in list(fused.keys()):
                fused[k] *= inv

        return {k:v for k,v in fused.items() if v >= CANDIDATE_KEEP_THRESHOLD}

    @lru_cache(maxsize=8192)
    def _pre_fuse_token(self, token: str) -> Dict[str,float]:
        tk = token.strip()
        if not tk or tk.isdigit() or all(c in string.punctuation for c in tk):
            return {}

        lower = tk.lower()
        
        sc = script_candidate_score(tk)
        patt = pattern_hint_scores(lower)
        ch = char_pattern_score(lower)

        f_probs: Dict[str,float] = {}
        if self.model_mgr.fasttext:
            f_probs = self.model_mgr.fasttext_probs_batch([tk])[0]
        
        fused = self._fuse(tk, {}, f_probs, patt, sc, ch)

        # Script-based fallback for strong scripts
        script = dominant_script(tk)
        if script and script.upper() in ("DEVANAGARI","BENGALI","THAI"):
            if not fused or max(fused.values(), default=0.0) < 0.10:
                mapping = {"DEVANAGARI":"hi","BENGALI":"bn","THAI":"th"}
                lang = mapping.get(script.upper())
                if lang:
                    return {lang:1.0}

        return fused

    def _apply_models_and_fuse(self, tokens: List[str], pre: List[Dict[str,float]]) -> List[Dict[str,float]]:
        t_dists = self.model_mgr.transformer_probs(tokens) if self.model_mgr.transformer else [{} for _ in tokens]
        f_dists = self.model_mgr.fasttext_probs_batch(tokens) if self.model_mgr.fasttext else [{} for _ in tokens]

        out = []
        for tok, pre_d, tprob, fprob in zip(tokens, pre, t_dists, f_dists):
            lower = tok.lower()
            fused = self._fuse(tok, tprob, fprob, pattern_hint_scores(lower),
                             script_candidate_score(tok), char_pattern_score(lower))

            # Blend in pre-fused heuristic distribution
            if pre_d:
                alpha = 0.22
                keys = set(fused.keys()) | set(pre_d.keys())
                mixed: Dict[str,float] = {}

                for k in keys:
                    mixed[k] = fused.get(k,0.0)*(1.0 - alpha) + pre_d.get(k,0.0)*alpha

                tot = sum(mixed.values())
                if tot > 0:
                    inv = 1.0/tot
                    for k in list(mixed.keys()):
                        mixed[k] *= inv

                fused = {k:v for k,v in mixed.items() if v >= CANDIDATE_KEEP_THRESHOLD}

            out.append(fused)

        return out

    def _adaptive_unknown_injection(self, dists: List[Dict[str,float]], tokens: List[str]) -> List[Dict[str,float]]:
        n = len(tokens)
        enriched = []

        for i, dist in enumerate(dists):
            if not dist:
                fb = {}
                tl = tokens[i].lower()
                for d in (pattern_hint_scores(tl), script_candidate_score(tokens[i]), char_pattern_score(tl)):
                    for k,v in d.items():
                        fb[k] = max(fb.get(k,0), v)

                if fb:
                    tot = sum(fb.values())
                    fb = {k: v/tot for k,v in fb.items() if v>0}
                    enriched.append(fb)
                else:
                    enriched.append({'unknown':1.0})
                continue

            new = dict(dist)
            maxp = max(new.values())

            # Enhanced engine disagreement detection
            engine_disagreement = self._detect_engine_disagreement(tokens[i])
            
            # Neighbor context
            window = dists[max(0,i-2):min(n,i+3)]
            neighbor_avg = 0.0
            cnt = 0
            for w in window:
                if w:
                    neighbor_avg += max(w.values())
                    cnt += 1
            neighbor_avg = (neighbor_avg/cnt) if cnt else maxp

            sc = dominant_script(tokens[i])
            sc_up = sc.upper() if sc else ""

            # Adaptive threshold based on context and engine disagreement
            base_th = UNKNOWN_INJECT_MAXP_THRESHOLD * (1.0 - 0.7*neighbor_avg)
            
            # Increase threshold if engines disagree
            if engine_disagreement:
                base_th *= 1.3
            
            th = min(0.10, base_th) if sc_up=="LATIN" else max(0.07, min(base_th, 0.25))

            # Short token adjustments
            if len(tokens[i])<=2:
                th = min(th, 0.05 if sc_up=="LATIN" else 0.10)
                # Extra uncertainty for very short tokens with engine disagreement
                if engine_disagreement and len(tokens[i]) == 1:
                    th *= 1.2

            # Strong script confidence bypass
            if sc_up not in ['LATIN'] and maxp >= 0.18:
                enriched.append(new)
                continue

            # Enhanced confidence assessment
            strong_hint = any(v>=0.25 for v in new.values()) or len(new)>=2
            
            # Check for known problematic patterns
            is_problematic = self._is_problematic_token(tokens[i], new)

            # Inject unknown if confidence is low or engines disagree significantly
            if (maxp < th and not strong_hint) or (engine_disagreement and maxp < 0.3) or is_problematic:
                unk_factor = 0.7
                if engine_disagreement:
                    unk_factor = 0.9  # More aggressive unknown injection on disagreement
                if is_problematic:
                    unk_factor = 0.8
                    
                unk = max(UNKNOWN_MIN_PROB, (th - maxp)*unk_factor)
                total_exist = sum(new.values())
                scale = (1.0 - unk)/total_exist if total_exist>0 else 0.0

                for k in list(new.keys()):
                    new[k] *= scale
                new['unknown'] = unk

            enriched.append(new)

        return enriched

    def _detect_engine_disagreement(self, token: str) -> bool:
        """
        Detect if different engines (transformer vs fasttext vs patterns) disagree significantly.
        """
        predictions = {}
        
        # Get transformer prediction if available
        if self.model_mgr.transformer:
            try:
                t_result = self.model_mgr.transformer_probs([token])
                if t_result and t_result[0]:
                    predictions['transformer'] = max(t_result[0], key=t_result[0].get)
            except Exception:
                pass
        
        # Get fasttext prediction if available
        if self.model_mgr.fasttext:
            try:
                f_result = self.model_mgr.fasttext_probs_batch([token])
                if f_result and f_result[0]:
                    predictions['fasttext'] = max(f_result[0], key=f_result[0].get)
            except Exception:
                pass
        
        # Get pattern-based prediction
        patterns = pattern_hint_scores(token.lower())
        if patterns:
            predictions['patterns'] = max(patterns, key=patterns.get)
        
        # Check for disagreement
        if len(predictions) >= 2:
            unique_predictions = set(predictions.values())
            return len(unique_predictions) > 1
        
        return False

    def _is_problematic_token(self, token: str, distribution: Dict[str, float]) -> bool:
        """
        Identify tokens that are known to be problematic for classification.
        """
        token_lower = token.lower()
        
        # Very short tokens are often problematic
        if len(token) <= 2:
            return True
        
        # Tokens with very flat distributions
        if distribution:
            values = list(distribution.values())
            max_val = max(values)
            if max_val < 0.4 and len([v for v in values if v > 0.15]) >= 3:
                return True
        
        # Known problematic patterns
        problematic_patterns = [
            r'^[a-z]{1,3}$',  # Very short latin words
            r'^\d+$',         # Pure numbers
            r'^[^\w\s]+$',    # Pure punctuation
            r'^(a|an|the|is|in|on|at|to|of|for|and|or|but)$',  # Common ambiguous words
        ]
        
        return any(re.match(pattern, token_lower) for pattern in problematic_patterns)

    def _enhanced_disambiguate(self, dists: List[Dict[str,float]], tokens: List[str]) -> List[Dict[str,float]]:
        n = len(tokens)
        if n == 0: return dists

        # --- Pre-computation for Sentence-Level Context ---
        has_kana = any(get_script(c) in ("HIRAGANA", "KATAKANA") for t in tokens for c in t)

        latin_evidence = Counter()
        for tok in tokens:
            tl = tok.lower()
            # Count pattern matches for non-English Latin languages
            for lang, patterns in LANGUAGE_PATTERNS.items():
                if lang not in ['en', 'id'] and lang in SCRIPT_LANG_MAP['LATIN']:
                    for pat in patterns:
                        if re.search(pat, tl):
                            latin_evidence[lang] += 1
            # Give a stronger signal for Indonesian morphology
            if any(re.search(p, tl) for p in LANGUAGE_PATTERNS.get('id', [])):
                latin_evidence['id'] += 2
        
        dominant_other_latin, dominant_count = (latin_evidence.most_common(1)[0]) if latin_evidence else (None, 0)
        
        for i, (tok, dist) in enumerate(zip(tokens, dists)):
            if not dist:
                continue

            tl = tok.lower()
            sc = dominant_script(tok)
            sc_up = sc.upper() if sc else ""

            # --- Soft English Prior with Context Awareness ---
            if sc_up == 'LATIN' and 'en' in dist:
                # Calculate soft prior based on context
                en_prior = self._calculate_soft_english_prior(tok, tokens, dominant_other_latin, dominant_count)
                dist['en'] *= en_prior

            # --- Rebalanced Chinese vs. Japanese using Kana Sentence Prior ---
            if sc_up == 'HAN' and ('zh' in dist or 'ja' in dist):
                if has_kana:
                    # Boost Japanese if Kana is present anywhere in the sentence
                    dist['ja'] = dist.get('ja', 0) + 0.3
                    dist['zh'] = dist.get('zh', 0) * 0.7
                
                # Token-level rules after applying the prior
                if any(ch in SIMP_ONLY_CHARS for ch in tok):
                    dist['zh'] = dist.get('zh', 0) + 0.50
                    dist['ja'] = dist.get('ja', 0) * 0.5
                elif any(ch in JP_SPECIFIC_CHARS for ch in tok):
                    dist['ja'] = dist.get('ja', 0) + 0.50
                    dist['zh'] = dist.get('zh', 0) * 0.5

            # --- Urdu vs Arabic based on characters ---
            if any(l in dist for l in ("ar","ur")):
                if any(c in UR_SPECIFIC_CHARS for c in tok):
                    dist['ur'] = dist.get('ur', 0) + 0.8
                    dist['ar'] = dist.get('ar', 0) * 0.1
                elif any(c in AR_SPECIFIC_CHARS for c in tok):
                    dist['ar'] = dist.get('ar', 0) + 0.4
                    dist['ur'] = dist.get('ur', 0) * 0.4

            # --- Boost based on morphological and pattern matching ---
            pattern_scores = Counter()
            for lang, patterns in LANGUAGE_PATTERNS.items():
                if lang in dist:
                    for pat in patterns:
                        if re.search(pat, tl):
                            pattern_scores[lang] += 1
            
            if pattern_scores:
                best_pattern_lang, best_pattern_count = pattern_scores.most_common(1)[0]
                if best_pattern_count > 0 and len(dist) > 1:
                    # Give a boost to the language with the most pattern matches for this token
                    dist[best_pattern_lang] = dist.get(best_pattern_lang, 0) + 0.25 * best_pattern_count
                    # Suppress other languages in the same script group
                    for other_lang in dist:
                        if other_lang != best_pattern_lang and SCRIPT_LANG_MAP.get(dominant_script(tok), [best_pattern_lang])[0] == SCRIPT_LANG_MAP.get(dominant_script(tok), [other_lang])[0]:
                            dist[other_lang] *= 0.7

            # Normalize at the end of each token's logic
            total = sum(dist.values())
            if total > 0:
                inv = 1.0 / total
                for k in list(dist.keys()):
                    dist[k] *= inv
        
        return dists

    def _are_related(self, a: str, b: str) -> bool:
        groups = [
            {"en","de","nl"}, {"es","pt","it","fr"}, {"hi","ur"}, {"zh","ja"}, {"id"}
        ]
        return any(a in g and b in g for g in groups)

    def _enhanced_dp(self, dists: List[Dict[str,float]], tokens: List[str]) -> List[str]:
        n = len(dists)
        if n == 0: return []

        if n == 1:
            d0 = dists[0]
            if not d0: return ["unknown"]
            return [max(d0.items(), key=lambda x: x[1])[0]]

        langs = set()
        for d in dists:
            langs.update(d.keys())
        langs.add("unknown")
        langs = list(langs)

        L = len(langs)
        idx = {l:i for i,l in enumerate(langs)}

        dp = [[float('-inf')]*L for _ in range(n)]
        par = [[-1]*L for _ in range(n)]

        # Initialize
        if dists[0]:
            for lang,score in dists[0].items():
                dp[0][idx[lang]] = math.log(max(score, MIN_LANG_SCORE))
        else:
            dp[0][idx['unknown']] = math.log(MIN_LANG_SCORE)

        for i in range(1, n):
            cur = dists[i]
            cur_tok = tokens[i]
            prev_tok = tokens[i-1]

            for ci, cl in enumerate(langs):
                cs = cur.get(cl, MIN_LANG_SCORE)
                clog = math.log(max(cs, MIN_LANG_SCORE))

                for pj, pl in enumerate(langs):
                    if dp[i-1][pj] == float('-inf'): continue

                    trans = self._enhanced_transition(pl, cl, prev_tok, cur_tok)

                    # Script mismatch penalty
                    cur_script = dominant_script(cur_tok)
                    cur_script_up = cur_script.upper() if cur_script else ''
                    primary = LANG_PRIMARY_SCRIPT.get(cl)
                    mismatch = False

                    if primary and cur_script_up and cur_script_up != primary and len(cur_tok) > SCRIPT_MISMATCH_LEN_THRESHOLD:
                        if not (primary=='HAN' and cur_script_up in ('HAN','HIRAGANA','KATAKANA')):
                            mismatch = True

                    # Emission bonus/penalty for ID/EN morphology
                    cur_tl = cur_tok.lower()
                    if any(re.search(p, cur_tl) for p in LANGUAGE_PATTERNS.get('id',[])):
                        if cl == 'id':
                            clog += 0.30 # Bonus for ID
                        elif cl == 'en':
                            clog -= 0.40 # Penalty for EN

                    emission_adj = clog - (SCRIPT_MISMATCH_PENALTY if mismatch else 0.0)
                    score = dp[i-1][pj] + emission_adj - trans

                    if score > dp[i][ci]:
                        dp[i][ci] = score
                        par[i][ci] = pj

        # Backtrack
        best = max(range(L), key=lambda j: dp[n-1][j])
        path = []
        cur = best

        for i in range(n-1, -1, -1):
            path.append(langs[cur])
            cur = par[i][cur] if par[i][cur] != -1 else cur

        path.reverse()
        return [p if p is not None else 'unknown' for p in path]

    def _enhanced_transition(self, pl: str, cl: str, prev_tok: str, cur_tok: str) -> float:
        """Transition penalty from previous language pl to current language cl.
        Higher return value means stronger penalty for switching from pl to cl.
        """
        if pl == cl:
            return 0.0

        trans = SWITCH_PENALTY

        # Extra penalty for very short-token switches (except for scripts where short tokens are common)
        cur_sc = dominant_script(cur_tok)
        cur_sc_up = cur_sc.upper() if cur_sc else ''
        if len(cur_tok) <= SHORT_TOKEN_MAX_LEN and cur_sc_up not in SHORT_NO_PENALTY_SCRIPTS:
            trans += SHORT_SWITCH_EXTRA

        # Implausible transitions get additional costs
        implausible_transitions = {
            ('hi','id'): 0.9, ('id','hi'): 0.9,
            ('ar','id'): 0.7, ('th','en'): 0.6,
            ('en','hi'): 0.45, ('hi','en'): 0.35,
            ('id','en'): 0.35, ('en','id'): 0.35,
            ('fr','en'): 0.30, ('en','fr'): 0.20,
            # Enhanced ZH/JA transition penalties
            ('zh','ja'): 0.25, ('ja','zh'): 0.25,
            # Enhanced UR/AR transition penalties for short tokens
            ('ur','ar'): 0.20, ('ar','ur'): 0.20,
            # Cross-script penalties
            ('zh','en'): 0.40, ('en','zh'): 0.40,
            ('ja','en'): 0.40, ('en','ja'): 0.40,
            ('ar','en'): 0.35, ('en','ar'): 0.35,
            ('ur','en'): 0.35, ('en','ur'): 0.35,
            ('th','ja'): 0.50, ('ja','th'): 0.50,
        }

        # Dynamic penalties based on token characteristics
        penalty_adjustment = 0.0
        
        # Extra penalty for ZH/JA confusion on short tokens
        if (pl, cl) in [('zh','ja'), ('ja','zh')] and len(cur_tok) <= 3:
            penalty_adjustment += 0.15
            
        # Extra penalty for UR/AR confusion on short tokens  
        if (pl, cl) in [('ur','ar'), ('ar','ur')] and len(cur_tok) <= 4:
            penalty_adjustment += 0.15
            
        # Reduce penalty for script-consistent transitions
        prev_script = dominant_script(prev_tok)
        cur_script = dominant_script(cur_tok)
        if prev_script and cur_script and prev_script == cur_script:
            if prev_script.upper() in ['HAN', 'ARABIC', 'THAI']:
                penalty_adjustment -= 0.05

        base_penalty = implausible_transitions.get((pl, cl), 0.0)
        trans += base_penalty + penalty_adjustment

        # Small discount for related languages (easier switch)
        try:
            if self._are_related(pl, cl):
                trans = max(0.0, trans - 0.08)
        except Exception:
            pass

        return trans

    def _sentence_guess(self, tokens: List[str], fused: List[Dict[str,float]]) -> Optional[str]:
        # Enhanced sentence-level language detection
        votes = Counter()
        for d in fused:
            if d:
                best = max(d.items(), key=lambda x: x[1])[0]
                votes[best] += 1

        if votes:
            top, cnt = votes.most_common(1)[0]
            if cnt / max(1, len(fused)) >= 0.25:
                return top

        # Full sentence models as fallback
        text = " ".join(tokens).strip()

        if self.model_mgr.transformer:
            try:
                outs = self.model_mgr.transformer(text)
                if outs and isinstance(outs, list):
                    for out in outs:
                        if isinstance(out, list):
                            for e in out:
                                if isinstance(e, dict):
                                    lab = e.get('label','').lower()
                                    if lab in TOP_20_LANGS:
                                        return lab
            except Exception:
                pass

        return None

    def _fill_unknowns(self, tokens: List[str], chosen: List[str], fused: List[Dict[str,float]]) -> List[str]:
        if not chosen: return chosen
        res = chosen[:]
        n = len(res)

        # Local neighbor fill
        for i, c in enumerate(res):
            if c == 'unknown':
                left = res[i-1] if i > 0 else None
                right = res[i+1] if i+1 < len(res) else None

                if left and right and left == right and left in TOP_20_LANGS:
                    res[i] = left

        # Script-based fill
        for i, c in enumerate(res):
            if c == 'unknown':
                sc = dominant_script(tokens[i])
                if sc and sc.upper() in PERFECT_SCRIPT_MAP:
                    lang = PERFECT_SCRIPT_MAP[sc.upper()]
                    if lang in TOP_20_LANGS:
                        res[i] = lang

        # Majority backfill for high unknown ratio
        unk_ratio = sum(1 for c in res if c == 'unknown') / len(res)
        if unk_ratio > 0.4:
            sentence_guess = self._sentence_guess(tokens, fused)
            if sentence_guess and sentence_guess in TOP_20_LANGS:
                for i, c in enumerate(res):
                    if c == 'unknown':
                        token_dist = fused[i] if i < len(fused) else {}
                        if not token_dist or max(token_dist.values(), default=0) < 0.08:
                            res[i] = sentence_guess

        return res

    def _latin_consolidation(self, tokens: List[str], langs: List[str]) -> List[str]:
        res = langs[:]
        counts = Counter(lang for lang in res if lang != 'unknown')
        if not counts: return res

        # Find dominant language overall
        dom_lang, dom_count = counts.most_common(1)[0]
        latin_set = {'en','id','fr','es','pt','it','de','nl','pl','tr','vi'}
        n = len(tokens)

        # More conservative consolidation: only if one language is overwhelmingly dominant
        if dom_lang in latin_set and dom_count >= max(8, int(0.85 * n)):
            for i in range(n):
                sc = dominant_script(tokens[i])
                # Only change 'unknown' or other Latin languages for short, ambiguous tokens
                if (not sc) or sc.upper() == 'LATIN':
                    if (res[i] == 'unknown' or res[i] in latin_set) and len(tokens[i]) <= 4:
                        res[i] = dom_lang
        
        return res


    def detect_languages(self, text: str) -> List[Tuple[str,str]]:
        if not text or not text.strip():
            return []

        # Step 1: MaskLID sentence wrapper to get candidate languages
        candidate_languages = self._masklid_sentence_wrapper(text)

        # Step 2: Tokenize with enhanced splitting
        tokens = tokenize(unicodedata.normalize('NFC', text))
        if not tokens:
            return [(text.strip(), "unknown")]

        # Step 3: Apply Latin glue-splitter for boundary error reduction
        tokens = self._latin_glue_splitter(tokens)

        # Pre-fuse with enhanced heuristics
        pre = [self._pre_fuse_token(t) for t in tokens]

        # Apply models and fuse
        fused = self._apply_models_and_fuse(tokens, pre)

        # Step 4: Apply MaskLID constraints - filter distributions to only candidate languages
        for i, dist in enumerate(fused):
            if dist and candidate_languages:
                # Keep only candidate languages in the distribution
                filtered_dist = {}
                for lang in dist:
                    if lang in candidate_languages or lang == 'unknown':
                        filtered_dist[lang] = dist[lang]
                
                # Renormalize if we have valid candidates
                if filtered_dist and any(k != 'unknown' for k in filtered_dist):
                    total = sum(filtered_dist.values())
                    if total > 0:
                        fused[i] = {k: v/total for k, v in filtered_dist.items()}
                    else:
                        fused[i] = filtered_dist

        # Heuristic fallback for low-confidence tokens
        for i, dist in enumerate(fused):
            if not dist or max(dist.values(), default=0.0) < 0.12:
                fb = {}
                tl = tokens[i].lower()
                for d in (pattern_hint_scores(tl), script_candidate_score(tokens[i]), char_pattern_score(tl)):
                    for k, v in d.items():
                        # Apply candidate language constraint here too
                        if not candidate_languages or k in candidate_languages:
                            fb[k] = max(fb.get(k, 0), v)
                if fb:
                    tot = sum(fb.values())
                    fused[i] = {k: v/tot for k, v in fb.items()}

        # Unknown injection (conservative)
        fused = self._adaptive_unknown_injection(fused, tokens)

        # Enhanced disambiguation with sentence-level context
        fused = self._enhanced_disambiguate(fused, tokens)

        # Enhanced DP smoothing
        chosen = self._enhanced_dp(fused, tokens)

        # Post-processing
        unk_ratio = sum(1 for c in chosen if c == 'unknown') / len(chosen)
        if unk_ratio >= UNKNOWN_RATIO_FALLBACK:
            guess = self._sentence_guess(tokens, fused)
            if guess and guess in TOP_20_LANGS:
                # Prefer candidate languages for guessing
                if not candidate_languages or guess in candidate_languages:
                    new = []
                    for i, c in enumerate(chosen):
                        if c != 'unknown':
                            new.append(c)
                            continue
                        maxp = max(fused[i].values()) if fused[i] else 0.0
                        new.append(guess if maxp < 0.08 else c)
                    chosen = new

        # Fill remaining unknowns
        chosen = self._fill_unknowns(tokens, chosen, fused)

        # Conservative Latin consolidation
        chosen = self._latin_consolidation(tokens, chosen)

        # Merge adjacent spans
        merged: List[Tuple[str,str]] = []
        cur_lang, buf = None, []

        for tok, lang in zip(tokens, chosen):
            if cur_lang is None:
                cur_lang, buf = lang, [tok]
            elif lang == cur_lang:
                buf.append(tok)
            else:
                merged.append((" ".join(buf), cur_lang))
                cur_lang, buf = lang, [tok]

        if buf and cur_lang is not None:
            merged.append((" ".join(buf), cur_lang))

        return [(seg.strip(), lang) for seg, lang in merged if seg.strip()]

    def _masklid_sentence_wrapper(self, text: str) -> set:
        """
        MaskLID sentence wrapper - uses iterative masking to extract language set.
        This constrains the token-level predictions to only languages present in the sentence.
        """
        if not text or not text.strip():
            return set()
        
        # Step 1: Get initial sentence-level language predictions using different strategies
        candidate_langs = set()
        
        # Strategy 1: Use fastText on the whole sentence (if available)
        if self.model_mgr.fasttext:
            try:
                full_pred = self.model_mgr.fasttext.predict(text.replace('\n', ' '), k=5)
                if len(full_pred) >= 2 and hasattr(full_pred[0], '__iter__'):
                    labels, scores = full_pred
                    for label, score in zip(labels, scores):
                        if score > 0.1:  # Confidence threshold
                            lang = label.replace('__label__', '')
                            if lang in TOP_20_LANGS:
                                candidate_langs.add(lang)
            except Exception:
                pass
        
        # Strategy 2: Use transformer on the whole sentence (if available)
        if self.model_mgr.transformer:
            try:
                result = self.model_mgr.transformer(text, top_k=None)
                if result and isinstance(result, list):
                    for item in result:
                        if item.get('score', 0) > 0.1:
                            lang = item.get('label', '')
                            if lang in TOP_20_LANGS:
                                candidate_langs.add(lang)
            except Exception:
                pass
        
        # Strategy 3: Script-based language detection
        scripts_found = set()
        for char in text:
            script = get_script(char)
            if script:
                scripts_found.add(script.upper())
        
        # Map scripts to languages
        for script in scripts_found:
            if script in PERFECT_SCRIPT_MAP:
                candidate_langs.add(PERFECT_SCRIPT_MAP[script])
            elif script in SCRIPT_LANG_MAP:
                candidate_langs.update(SCRIPT_LANG_MAP[script][:3])  # Top 3 for each script
        
        # Strategy 4: Pattern-based detection for strong patterns
        text_lower = text.lower()
        for lang, patterns in LANGUAGE_PATTERNS.items():
            if lang in TOP_20_LANGS:
                for pattern in patterns[:5]:  # Check top 5 patterns
                    if re.search(pattern, text_lower):
                        candidate_langs.add(lang)
                        break
        
        # Ensure we have at least a few candidates
        if len(candidate_langs) < 2:
            # Fallback to common languages based on script
            if any(get_script(c) == 'LATIN' for c in text):
                candidate_langs.update(['en', 'fr', 'es', 'de', 'it', 'pt'])
            candidate_langs.update(['en'])  # Always include English as fallback
        
        # Limit to top 10 to prevent explosion
        if len(candidate_langs) > 10:
            # Prioritize by frequency in training data or use heuristics
            prioritized = []
            common_order = ['en', 'zh', 'es', 'hi', 'fr', 'ar', 'pt', 'ru', 'ja', 'de']
            for lang in common_order:
                if lang in candidate_langs:
                    prioritized.append(lang)
                if len(prioritized) >= 8:
                    break
            # Add remaining candidates
            for lang in candidate_langs:
                if lang not in prioritized and len(prioritized) < 10:
                    prioritized.append(lang)
            candidate_langs = set(prioritized)
        
        return candidate_langs

    def _latin_glue_splitter(self, tokens: List[str]) -> List[str]:
        """
        Latin glue-splitter for long tokens to fix boundary errors.
        Identifies and splits concatenated words in Latin scripts.
        """
        result = []
        
        for token in tokens:
            if len(token) <= 6:  # Only split longer tokens
                result.append(token)
                continue
                
            script = dominant_script(token)
            if not script or script.upper() != 'LATIN':
                result.append(token)
                continue
            
            # Try to split concatenated words
            splits = self._try_split_concatenated(token)
            if len(splits) > 1:
                result.extend(splits)
            else:
                result.append(token)
        
        return result
    
    def _try_split_concatenated(self, token: str) -> List[str]:
        """
        Attempt to split a concatenated Latin token into constituent words.
        Uses morphological patterns and common affixes.
        """
        if len(token) <= 6:
            return [token]
        
        token_lower = token.lower()
        
        # Common prefixes and suffixes for different languages
        prefixes = {
            'un', 're', 'pre', 'dis', 'mis', 'over', 'under', 'anti', 'auto', 'co', 'counter',
            'de', 'ex', 'extra', 'hyper', 'inter', 'intra', 'macro', 'micro', 'multi', 
            'non', 'post', 'pro', 'pseudo', 'semi', 'sub', 'super', 'trans', 'ultra'
        }
        
        suffixes = {
            'ing', 'ed', 'er', 'est', 'ly', 'tion', 'sion', 'ness', 'ment', 'able', 'ible',
            'ous', 'ful', 'less', 'ism', 'ist', 'ity', 'age', 'ance', 'ence', 'ship',
            'ado', 'ando', 'endo', 'ido', 'ada', 'ida', 'mente', 'ción', 'sión',
            'ment', 'tion', 'able', 'ible', 'eux', 'euse', 'ique', 'aire', 'oire',
            'ung', 'keit', 'heit', 'lich', 'isch', 'bar', 'sam',
            'ata', 'uto', 'ito', 'oso', 'osa', 'ivo', 'iva', 'evo', 'eva'
        }
        
        # Try splitting at natural boundaries
        best_splits = [token]
        max_score = 0
        
        # Try different split positions
        for i in range(3, len(token) - 2):
            left = token_lower[:i]
            right = token_lower[i:]
            
            # Score this split based on morphological patterns
            score = 0
            
            # Bonus for common word patterns
            if len(left) >= 3 and len(right) >= 3:
                score += 1
                
            # Check if left part ends with common suffixes
            for suffix in suffixes:
                if left.endswith(suffix) and len(left) > len(suffix) + 2:
                    score += 2
                    break
            
            # Check if right part starts with common prefixes
            for prefix in prefixes:
                if right.startswith(prefix) and len(right) > len(prefix) + 2:
                    score += 2
                    break
            
            # Bonus for reasonable word lengths
            if 3 <= len(left) <= 12 and 3 <= len(right) <= 12:
                score += 1
            
            # Penalty for very uneven splits
            ratio = min(len(left), len(right)) / max(len(left), len(right))
            if ratio > 0.3:
                score += 1
            
            if score > max_score:
                max_score = score
                best_splits = [token[:i], token[i:]]
        
        # Only split if we found a good boundary
        if max_score >= 3:
            # Recursively try to split the parts
            final_splits = []
            for part in best_splits:
                sub_splits = self._try_split_concatenated(part)
                final_splits.extend(sub_splits)
            return final_splits
        
        return [token]

    def _calculate_soft_english_prior(self, token: str, tokens: List[str], dominant_other_latin: Optional[str], dominant_count: int) -> float:
        """
        Calculate soft prior for English based on context.
        Returns a factor between 0.3 and 1.0 to multiply English probability.
        """
        token_lower = token.lower()
        
        # Base prior is neutral
        prior = 1.0
        
        # Factor 1: Strong English patterns get boosted
        strong_english_patterns = [
            r'\b(the|and|that|have|for|not|with|you|this|but|his|from|they)\b',
            r'\b(was|were|been|has|had|will|would|could|should|might)\b',
            r'\b(ing|ed|ly|tion|sion)$',
            r'^(un|re|pre|dis|mis|over|under)',
        ]
        
        if any(re.search(p, token_lower) for p in strong_english_patterns):
            prior *= 1.2  # Boost for strong English indicators
            return min(prior, 1.0)
        
        # Factor 2: Check for non-English characteristics
        if dominant_other_latin and dominant_count >= 2:
            # Strong evidence for another Latin language
            
            # Check if token matches the dominant language patterns
            dominant_patterns = LANGUAGE_PATTERNS.get(dominant_other_latin, [])
            if any(re.search(p, token_lower) for p in dominant_patterns[:5]):
                prior *= 0.3  # Strong penalty if it matches another language
            else:
                # Check for general non-English characteristics
                non_english_indicators = [
                    (r'ñ|ç|ã|õ|ê|â|ô|á|é|í|ó|ú|à|è|ì|ò|ù', 0.4),  # Romance diacritics
                    (r'ü|ä|ö|ß', 0.4),  # German umlauts
                    (r'ą|ę|ł|ń|ó|ś|ź|ż', 0.4),  # Polish diacritics
                    (r'ş|ğ|ı|ö|ü|ç', 0.4),  # Turkish characters
                    (r'(zione|zione|ment|eur|euse)$', 0.5),  # Romance suffixes
                    (r'(heit|keit|ung|lich)$', 0.5),  # German suffixes
                    (r'(ość|acz|arz|nik)$', 0.5),  # Slavic suffixes
                ]
                
                for pattern, penalty in non_english_indicators:
                    if re.search(pattern, token_lower):
                        prior *= penalty
                        break
                else:
                    # Moderate penalty for ambiguous tokens in non-English context
                    prior *= 0.6
        
        # Factor 3: Length-based adjustment
        if len(token) <= 2:
            prior *= 0.8  # Short tokens are less reliable for English
        elif len(token) >= 8:
            prior *= 1.1  # Longer tokens are more reliable
        
        # Factor 4: Context from surrounding tokens
        non_english_context = 0
        for context_token in tokens:
            if context_token != token:
                context_lower = context_token.lower()
                if any(c in 'ñçãõêâôáéíóúàèìòùüäößąęłńóśźżşğıöüç' for c in context_lower):
                    non_english_context += 1
        
        if non_english_context >= 3:
            prior *= 0.5  # Strong non-English context
        elif non_english_context >= 1:
            prior *= 0.7  # Some non-English context
        
        # Ensure prior stays within bounds [0.3, 1.0]
        return max(0.3, min(1.0, prior))

# ------------------------------
# Global API
# ------------------------------

_global_detector: Optional[EnhancedDetector] = None

def get_detector(enable_transformer: bool=True, fasttext_path: str=FASTTEXT_PATH_DEFAULT) -> EnhancedDetector:
    global _global_detector
    if _global_detector is None:
        _global_detector = EnhancedDetector(enable_transformer, fasttext_path)
    return _global_detector

def detect_languages(text: str) -> List[Tuple[str,str]]:
    det = get_detector()
    return det.detect_languages(text)

# ------------------------------
# Batch processing support
# ------------------------------

def batch_detect_languages(texts: List[str], max_workers: int=4) -> List[List[Tuple[str,str]]]:
    if not texts: return []

    det = get_detector()
    results: List[Optional[List[Tuple[str,str]]]] = [None] * len(texts)

    def process(idx: int):
        return idx, det.detect_languages(texts[idx])

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(process, i) for i in range(len(texts))]
        for fut in as_completed(futs):
            idx, res = fut.result()
            results[idx] = res

    return results # type: ignore

if __name__ == "__main__":
    # Example usage
    test_text = "Hello world! Bonjour le monde! Hola mundo! ã “ã‚“ã «ã ¡ã ¯ä¸–ç•Œï¼ "
    result = detect_languages(test_text)
    for segment, lang in result:
        print(f"{lang}: {segment}")