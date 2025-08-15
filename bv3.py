# d.py — Enhanced Polyglot Language Detection (Target: F1 > 0.80)

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

TRANSFORMER_MODEL = "papluca/xlm-roberta-base-language-detection"
FASTTEXT_PATH_DEFAULT = os.environ.get("POLYLANGID_FASTTEXT_PATH","lid.176.ftz")
FASTTEXT_FALLBACK_PATH = os.environ.get("POLYLANGID_FASTTEXT_FALLBACK","")
TRANSFORMER_FP16 = True
TRANSFORMER_BATCH_SIZE = 64 if _torch_cuda else 16
FASTTEXT_TOP_K = 5
FASTTEXT_TOP_K_SHORT = 3

# Enhanced smoothing parameters
SWITCH_PENALTY = 0.15  # Reduced even further for better flow
MIN_LANG_SCORE = 1e-6
SHORT_TOKEN_MAX_LEN = 2
SHORT_SWITCH_EXTRA = 0.18  # Reduced from 0.20
SHORT_NO_PENALTY_SCRIPTS = {"HAN","HIRAGANA","KATAKANA","THAI","HANGUL"}

UNKNOWN_RATIO_FALLBACK = 0.65  # Reduced from 0.70
UNKNOWN_MIN_PROB = 0.03  # Reduced from 0.04
UNKNOWN_INJECT_MAXP_THRESHOLD = 0.25  # Reduced from 0.30
CANDIDATE_KEEP_THRESHOLD = 0.015  # Reduced from 0.020
UNKNOWN_NEIGHBOR_FILL_THRESHOLD = 0.55

# Primary / canonical scripts for mismatch penalization
LANG_PRIMARY_SCRIPT = {
    'hi':'DEVANAGARI','bn':'BENGALI','ar':'ARABIC','ur':'ARABIC','zh':'HAN','ja':'HAN',
    'ko':'HANGUL','th':'THAI','ru':'CYRILLIC'
}

SCRIPT_MISMATCH_PENALTY = 0.28
SCRIPT_MISMATCH_LEN_THRESHOLD = 3

# ------------------------------
# Enhanced Patterns and Lexicons
# ------------------------------

LANGUAGE_PATTERNS = {
    "en": [
        r'\b(the|and|that|have|you|this|with|from|they|been|which|their|will|would|could|should|must|can|are|is|was|were|be|do|does|did|has|had|of|to|for|on|as|by|it|an|or|if|about|there|what|when|who|where|why|how)\b',
        r"\b(?:I'm|you're|we're|they're|he's|she's|it's|don't|doesn't|didn't|can't|won't|isn't|aren't)\b",
        r'\b\w+ing\b', r'\b\w+ed\b', r'\b\w+ly\b', r'\b\w+tion\b',
        r'\b(very|just|now|then|here|some|all|any|many|much|most|more|less|good|bad|big|small|new|old|young|long|short|high|low|fast|slow)\b',
        r'\b(because|through|after|before|while|during|until|since|although|though|however|therefore|moreover|furthermore)\b'
    ],
    "fr": [
        r'\b(le|la|les|un|une|des|et|de|du|dans|pour|avec|qui|que|ce|cette|ces|mais|ou|où|sur|par|est|sont|était|sera|avoir|être)\b',
        r'\b(très|plus|moins|bien|mal|tout|tous|toute|toutes|même|aussi|encore|déjà|jamais|toujours|souvent|parfois)\b',
        r'\b\w*ment\b', r'\b\w*tion\b', r'\b\w*sion\b', r'\w*[éèêëàâäôöùûüîï]\w*'
    ],
    "de": [
        r'\b(der|die|das|und|in|von|zu|den|mit|sich|auf|für|ist|im|dem|nicht|ein|eine|als|auch|es|an|haben|sein|werden)\b',
        r'\b(sehr|mehr|weniger|gut|schlecht|groß|klein|neu|alt|jung|lang|kurz|hoch|niedrig|schnell|langsam)\b',
        r'\b\w*heit\b', r'\b\w*keit\b', r'\b\w*ung\b', r'\w*[äöüß]\w*'
    ],
    "es": [r'\b(el|la|los|las|un|una|y|de|en|es|con|por|que|no|se|le|lo|me|te|su|sus|ser|estar|tener)\b'],
    "it": [r'\b(il|la|lo|gli|le|di|da|in|con|su|per|tra|fra|a|e|ma|o|se|che|chi|cui|dove|quando|come|essere|avere)\b'],
    "pt": [
        r'\b(o|a|os|as|um|uma|e|de|em|é|com|por|que|não|se|do|da|dos|das|no|na|ser|estar|ter)\b',
        r'\b\w*ção\b', r'\b\w*são\b', r'\b\w*oso\b', r'\b\w*osa\b', r'\b\w*ioso\b', r'\b\w*iosa\b',
        r'\b(também|então|já|muito|ainda|mais|menos|sempre|nunca|hoje|ontem|amanhã)\b',
        r'\b(para|pela|pelo|pelos|pelas|numa|nuns|numas|desta|deste|destes|destas)\b',
        r'[ãõ]', r'ção', r'são', r'oso$', r'osa$', r'ioso$', r'iosa$'
    ],
    "ru": [r'\b(это|что|который|быть|иметь|мочь|сказать|говорить|знать|стать|видеть|хотеть|такой|весь|один)\b'],
    "pl": [r'\b(i|jest|są|był|była|było|byli|były|a|ale|lub|bo|gdy|že|jak|co|gdzie|ja|ty|on|ona|my|wy|oni|one|być|mieć)\b'],
    "nl": [r'\b(en|de|het|een|van|in|op|voor|met|aan|bij|door|over|onder|naar|uit|tegen|tussen|zonder|zijn|hebben|worden)\b'],
    "tr": [
        r'\b(ve|bir|bu|şu|o|ben|sen|biz|siz|onlar|var|yok|ile|için|gibi|kadar|sonra|önce|olmak|etmek)\b',
        r'\b(çok|daha|az|iyi|kötü|büyük|küçük|yeni|eski|genç|uzun|kısa|yüksek|alçak|hızlı|yavaş)\b',
        r'\w*[ğıişçüö]\w*', r'uluslararası', r'teknoloji'
    ],
    "hi": [r'है', r'था', r'कर', r'से', r'रहा', r'की', r'के', r'को', r'में', r'पर', r'और', r'होना', r'करना'],
    "ar": [r'\bال\w{3,}', r'في', r'من', r'إلى', r'على', r'مع', r'عن', r'كان', r'يكون', r'هذا', r'ذلك'],
    "ur": [r'ہوں', r'ہے', r'تھا', r'رہا', r'چاہیے', r'کرना', r'میں', r'سے', r'اور', r'کے', r'کی', r'کا', r'ہونا'],
    "zh": [r'的', r'了', r'在', r'是', r'我', r'有', r'和', r'就', r'不', r'人', r'都', r'一', r'个', r'会', r'说'],
    "ja": [r'です', r'ます', r'した', r'ある', r'この', r'その', r'から', r'まで', r'である', r'という'],
    "ko": [r'입니다', r'어요', r'습니까', r'네요', r'하다', r'있다', r'없다', r'이다', r'것이', r'그리고'],
    "th": [r'ครับ', r'ค่ะ', r'จ้า', r'นะ', r'ใน', r'และ', r'หรือ', r'แต่', r'เป็น', r'มี'],
    "vi": [r'của', r'và', r'có', r'là', r'trong', r'với', r'cho', r'từ', r'được', r'này'],
    "id": [
        r'yang', r'dan', r'di', r'ke', r'dari', r'untuk', r'pada', r'dengan', r'adalah', r'ini', r'itu',
        r'akan', r'sudah', r'belum', r'sedang', r'bisa', r'dapat', r'harus', r'mau', r'ingin',
        r'ber\w+', r'me\w+', r'pe\w+', r'ter\w+', r'se\w+', r'ke\w+an',
        r'\w*kan\b', r'\w*nya\b', r'\w*lah\b', r'\w*kah\b', r'inter\w+', r'multi\w+', r'teknologi'
    ],
    "bn": [r'এর', r'এবং', r'যে', r'হয়', r'করে', r'থেকে', r'সাথে', r'জন্য', r'হতে', r'করা']
}

CHARACTER_PATTERNS = {
    'de': ['ä','ö','ü','ß'],
    'fr': ['ç','é','è','ê','à','ù','ô','â','î','œ','ï'],
    'es': ['ñ','í','ó','á','é','ú','ü'],
    'pt': ['ã','õ','ç','à','á','â','é','ê','í','ó','ô','ú'],
    'it': ['à','è','é','ì','ò','ù'],
    'tr': ['ğ','ı','ş','ç','ü','ö'],
    'pl': ['ą','ć','ę','ł','ń','ó','ś','ź','ż'],
    'nl': ['ij','oe','eu','aa','ee','oo','uu'],
    'vi': ['ă','â','ê','ô','ơ','ư','đ']
}

# Enhanced HAN character hints for zh/ja disambiguation
SIMP_ONLY_CHARS = set("艺术爱优现书庆问观联广产众讯电车门闻医气")
TRAD_BIAS_CHARS = set("藝術愛優現書觀聯廣門醫氣國體學專靜寧駅時円見曜")
JP_SPECIFIC_CHARS = set("円駅時曜見")  # More specifically Japanese

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

# Enhanced strong word lists
STRONG_EN_WORDS = {
    "hello","world","the","this","is","and","of","to","in","it","you","that","he","was","for","on","are","as","with","his","they","i","at","be","have","from","or","one","had","by","but","not","what","all","were","we","when","your","can","said","there","use","an","each","which","she","do","how","their","if","will","up","other","about","out","many","then","them","these","so","some","her","would","make","like","him","into","time","has","look","two","more","write","go","see","number","no","way","could","people","my","than","first","water","been","call","who","oil","its","now","find","long","down","day","did","get","come","made","may","part",
    # Additional distinctive English words
    "because","through","just","any","good","new","first","last","long","great","little","own","other","old","right","big","high","different","small","large","next","early","young","important","few","public","bad","same","able"
}

# Portuguese specific indicators
STRONG_PT_WORDS = {
    "também", "então", "já", "muito", "ainda", "mais", "menos", "sempre", "nunca", "hoje", "ontem", "amanhã",
    "para", "pela", "pelo", "pelos", "pelas", "numa", "nuns", "numas", "desta", "deste", "destes", "destas",
    "coração", "viagem", "família", "sabedoria", "natureza", "liberdade", "esperança"
}

# Spanish specific indicators  
STRONG_ES_WORDS = {
    "también", "entonces", "ya", "mucho", "aún", "más", "menos", "siempre", "nunca", "hoy", "ayer", "mañana",
    "para", "por", "pero", "como", "cuando", "donde", "porque", "hasta", "desde", "hacia",
    "corazón", "viaje", "familia", "sabiduría", "naturaleza", "libertad", "esperanza"
}

# Enhanced Indonesian detection
ID_COMPREHENSIVE_ROOTS = {
    "alam","harapan","esensi","jiwa","keluarga","cahaya","bunga","lautan",
    "bayangan","keberanian","keheningan","buku","keanggunan","gunung",
    "kebijaksanaan","mimpi","sungai","perdamaian","kebebasan","kehidupan",
    "pikiran","perasaan","perjalanan","strategi","kebenaran","keadilan",
    "kemerdekaan","kesehatan","kekuatan","kematian","kesempatan","kemajuan",
    "kemunduran","keterampilan","kecantikan","kebersihan","kesabaran",
    "kejujuran","kemurahan","kekayaan","kemiskinan","kesulitan","kemudahan",
    "kebugaran","kecerdasan","kemampuan","kebodohan","kemalasan","kegembiraan",
    "kesedihan","kekhawatiran","kebingungan","kebosanan","kecemasan",
    "kegelisahan","kegagalan","kesuksesan","kemenangan","kekalahan",
    "keterbatasan","keterikatan","keterasingan","keterpaksaan","keterampilan",
    "keterbukaan","bencana","kucing","natureza","dunia","masyarakat","masa","depan","masadepan",
    "seni","musik","cinta","strategi"  # Added from error analysis
}

ID_TRIGGERS = ID_COMPREHENSIVE_ROOTS | {
    "ini","adalah","teks","indonesia","bahasa","yang","dan","dengan","untuk",
    "pada","dari","ke","di","akan","sudah","belum","sedang","bisa","dapat",
    "harus","mau","ingin","mereka"
}

# Enhanced problematic word handling (from error analysis)
PROBLEMATIC_WORDS = {
    # German words often misclassified
    "eleganz": "de", "katze": "de", "wesen": "de", "mut": "de", "berg": "de",
    "natur": "de", "freiheit": "de", "musik": "de", "kunst": "de", "nationalismus": "de",
    "schatten": "de", "fenster": "de", "stern": "de", "familie": "de", "weisheit": "de",
    
    # Turkish words often misclassified  
    "pencere": "tr", "cesaret": "tr", "okyanus": "tr", "ruh": "tr",
    "sen": "tr", "bayangan": "tr", "seni": "tr", "kedi": "tr", "gölge": "tr",
    "aile": "tr", "bilgelik": "tr", "yıldız": "tr", "doğa": "tr", "özgürlük": "tr",
    
    # French words often misclassified as EN
    "silence": "fr", "art": "fr", "nature": "fr", "essence": "fr",
    "amour": "fr", "sagesse": "fr", "esprit": "fr", "chat": "fr", "multiculturel": "fr",
    "étoile": "fr", "famille": "fr", "liberté": "fr", "ombre": "fr", "fenêtre": "fr",
    
    # Portuguese words often misclassified as ES or EN
    "alma": "pt", "rio": "pt", "luz": "pt", "paz": "pt", "harmonioso": "pt",
    "viagem": "pt", "coração": "pt", "família": "pt", "sabedoria": "pt", "estrela": "pt",
    "sombra": "pt", "janela": "pt", "liberdade": "pt", "natureza": "pt",
    "futuro": "pt", "moral": "pt", "valores": "pt", "patrimônio": "pt", "multicultural": "pt",
    "nacional": "pt", "pensamento": "pt", "metodologia": "pt", "internacional": "pt",
    
    # Spanish words
    "arte": "es", "coraje": "es", "viaje": "es", "colectivo": "es",
    "historia": "es", "familia": "es", "sabiduría": "es", "estrella": "es",
    "sombra": "es", "ventana": "es", "libertad": "es", "naturaleza": "es",
    
    # Polish words
    "ocean": "pl", "sen": "pl", "kot": "pl", "strategia": "pl",
    "rodzina": "pl", "mądrość": "pl", "gwiazda": "pl", "cień": "pl",
    "okno": "pl", "wolność": "pl", "przyroda": "pl",
    
    # Specific Urdu words often misclassified as Arabic
    "رجحان": "ur", "اختراع": "ur", "طرز": "ur", "اقدار": "ur", "روایت": "ur", 
    "مستقبل": "ur", "معلومات": "ur", "مواد": "ur", "انسانیت": "ur", "سبز": "ur",
    
    # Indonesian words frequently misclassified
    "cahaya": "id", "harapan": "id", "esensi": "id", "alam": "id",
    "lautan": "id", "kucing": "id", "strategi": "id", "internasional": "id",
    "keluarga": "id", "kebijaksanaan": "id", "bintang": "id", "bayangan": "id",
    "jendela": "id", "kebebasan": "id", "teknologi": "id", "ekonomi": "id",
    "eksistensi": "id", "aksiologi": "id", "filosofi": "id", "material": "id"
}

# Vietnamese helpers
VI_DIACRITICS = set("ăâêôơưĂÂÊÔƠƯđĐ")
VI_ACCENTED = (set("aàáảãạăằắẳẵặâầấẩẫậeèéẻẽẸêềếểễệiìíỉĩịoòóỏõọôồốổỗộơờớởỡợuùúủũụưừứửữựyỳýỷỹỵ")
    | set("AÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬEÈÉẺẼẸÊỀẾỂỄỆIÌÍỈĨỊOÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢUÙÚỦŨỤƯỪỨỬỮỰYỲÝỶỸỴ")
    | set("đĐ")) - set("aeiouyAEIOUY")

VI_SYLLABLE_REGEX = re.compile(r"[bcdfghjklmnpqrstvwxyzđ]*[aàáảãạăằắẳẵặâầấẩẫậeèéẻẽẸêềếểễệiìíỉĩịoòóỏõọôồốổỗộơờớởỡợuùúủũụưừứửữựyỳýỷỹỵ]+[bcdfghjklmnpqrstvwxyzđ]*", re.IGNORECASE)

VI_COMPOUND_WHITELIST = {
    "tâmtrí","giấcmơ","tươnglai","lòngdũngcảm","giađình","nghệthuật","đạidương","thiênnhiên","âmnhạc",
    "sựimlặng","trítuệ","sựthanhlịch","tựdo","ngôisao","bóngtối","cửasổ","hòabình","hyvọng","ánhsáng"
}

VI_VOWELS = set("aàáảãạăằắẳẵặâầấẩẫậeèéẻẽẸêềếểễệiìíỉĩịoòóỏõọôồốổỗộơờớởỡợuùúủũụưừứửữựyỳýỷỹỵAÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬEÈÉẺẼẸÊỀẾỂỄỆIÌÍỈĨỊOÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢUÙÚỦŨỤƯỪỨỬỮỰYỲÝỶỸỴ")
VI_ONSETS = ["ngh","ng","gh","kh","th","nh","ph","tr","ch","qu","gi","b","c","d","đ","g","h","k","l","m","n","p","q","r","s","t","v","x"]
VI_CODAS = ["nh","ng","ch","c","m","n","p","t"]

# Enhanced Arabic/Urdu disambiguation
UR_SPECIFIC_CHARS = set("یگپچژڑںے")
UR_WORDS = {"ہے","میں","نے","کا","کی","کے","سے","اور","کرنا","ہونا"}
AR_SPECIFIC_CHARS = set("ظضغ")

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
            any(ch in VI_DIACRITICS for ch in t)):
            
            tl = t.lower()
            if tl in VI_COMPOUND_WHITELIST:
                out.append(t)
                continue
                
            diac_cnt = sum(1 for ch in t if ch in VI_DIACRITICS)
            left = tokens[idx-1] if idx > 0 else None
            right = tokens[idx+1] if idx+1 < len(tokens) else None
            
            left_vi = bool(left and any(ch in VI_DIACRITICS for ch in left))
            right_vi = bool(right and any(ch in VI_DIACRITICS for ch in right))
            
            bounds = _find_vi_boundaries(t)
            min_len_ok = 10 if diac_cnt >= 2 else 8
            need_context = (diac_cnt >= 2)
            
            # Be more conservative about splitting Vietnamese
            if (bounds and 
                (left_vi and right_vi) and  # Only split if surrounded by Vietnamese
                not t.isupper() and
                len(t) >= 12):  # And only for longer words
                
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
    out = []
    
    for idx, t in enumerate(tokens):
        if (6 <= len(t) <= 30 and t.isalpha() and 
            dominant_script(t) == 'LATIN' and 
            not any(ch in VI_DIACRITICS for ch in t)):
            
            left = tokens[idx-1] if idx > 0 else None
            right = tokens[idx+1] if idx+1 < len(tokens) else None
            
            left_id = left and left.lower() in ID_COMPREHENSIVE_ROOTS
            right_id = right and right.lower() in ID_COMPREHENSIVE_ROOTS
            
            splits = []
            i = 0
            
            while i < len(t):
                found = False
                for j in range(len(t), i+3, -1):
                    part = t[i:j].lower()
                    if part in ID_COMPREHENSIVE_ROOTS:
                        splits.append(t[i:j])
                        i = j
                        found = True
                        break
                        
                if not found:
                    splits = []
                    break
            
            if (splits and len(splits) >= 2 and 
                sum(len(s) for s in splits) == len(t) and 
                (left_id or right_id or len(t) >= 12)):
                out.extend(splits)
                continue
        
        out.append(t)
    
    return out

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
                jtoks = [str(tok) for tok in _janome.tokenize(seg) if str(tok).strip()]
                tokens.extend(jtoks if len(jtoks) > 1 else [seg])
                continue
            except Exception: 
                pass

        # HAN: prefer Janome with kana context else jieba
        if sc == 'HAN':
            used = False
            if _janome_available and _janome and _has_kana_context(idx):
                try:
                    jtoks = [str(tok) for tok in _janome.tokenize(seg) if str(tok).strip()]
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

        # Indonesian stemmer to surface roots
        if (sc == 'LATIN' and _id_stemmer_available and _id_stemmer and 
            len(seg) > 5):
            try:
                stemmed = _id_stemmer.stem(seg)
                if stemmed != seg and stemmed in ID_COMPREHENSIVE_ROOTS:
                    tokens.append(stemmed)
                    continue
            except Exception: 
                pass

        # Indic
        if sc == 'DEVANAGARI':
            tokens.extend(re.findall(r'[\u0900-\u097F]+', seg))
            continue
        if sc == 'BENGALI':
            tokens.extend(re.findall(r'[\u0980-\u09FF]+', seg))
            continue

        # Fallback Vietnamese heuristic
        if sc == 'LATIN' and any(ch in VI_DIACRITICS for ch in seg):
            vi_tokens = re.findall(r"[A-Za-zĂÂĐÊÔƠƯăâđêôơư]+", seg)
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
                if _torch_available and _torch_cuda and TRANSFORMER_FP16:
                    import torch as _torch_mod
                    self.transformer = pipeline("text-classification", model=TRANSFORMER_MODEL, device=device, return_all_scores=True, torch_dtype=_torch_mod.float16)
                else:
                    self.transformer = pipeline("text-classification", model=TRANSFORMER_MODEL, device=device, return_all_scores=True)
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
        tl = token.lower()
        
        # Check if transformer is available
        has_transformer = self.model_mgr.enable_transformer and self.model_mgr.transformer is not None
        
        if sc_up == "LATIN":
            strong_en = (re.fullmatch(r"[a-zA-Z']+", token) and tl in STRONG_EN_WORDS)
            strong_id = (tl in {"yang","dan","di","ke","dari","untuk","pada","dengan","adalah","ini","itu"} or 
                        re.search(r'(kan|nya|lah|kah)$', tl))
            
            if has_transformer:
                if strong_en or strong_id:
                    return {"transformer":0.70,"fasttext":0.25,"pattern":0.03,"script":0.01,"char":0.01}
                elif length<=2:
                    return {"transformer":0.20,"fasttext":0.35,"pattern":0.20,"script":0.15,"char":0.10}
                elif length<=4:
                    return {"transformer":0.35,"fasttext":0.40,"pattern":0.15,"script":0.08,"char":0.02}
                else:
                    return {"transformer":0.50,"fasttext":0.35,"pattern":0.10,"script":0.03,"char":0.02}
            else:
                # No transformer - differentiate English vs Indonesian better
                if strong_en:
                    return {"transformer":0.0,"fasttext":0.40,"pattern":0.45,"script":0.10,"char":0.05}
                elif strong_id:
                    return {"transformer":0.0,"fasttext":0.50,"pattern":0.35,"script":0.10,"char":0.05}
                elif length<=2:
                    return {"transformer":0.0,"fasttext":0.25,"pattern":0.40,"script":0.25,"char":0.10}
                elif length<=4:
                    return {"transformer":0.0,"fasttext":0.35,"pattern":0.40,"script":0.20,"char":0.05}
                else:
                    return {"transformer":0.0,"fasttext":0.40,"pattern":0.40,"script":0.15,"char":0.05}
        else:
            if has_transformer:
                if length<=2:
                    return {"transformer":0.15,"fasttext":0.25,"pattern":0.25,"script":0.25,"char":0.10}
                elif length<=4:
                    return {"transformer":0.25,"fasttext":0.30,"pattern":0.20,"script":0.20,"char":0.05}
                else:
                    return {"transformer":0.40,"fasttext":0.30,"pattern":0.15,"script":0.12,"char":0.03}
            else:
                # No transformer - boost pattern weights for non-Latin scripts too
                if length<=2:
                    return {"transformer":0.0,"fasttext":0.35,"pattern":0.30,"script":0.25,"char":0.10}
                elif length<=4:
                    return {"transformer":0.0,"fasttext":0.40,"pattern":0.30,"script":0.25,"char":0.05}
                else:
                    return {"transformer":0.0,"fasttext":0.45,"pattern":0.30,"script":0.20,"char":0.05}

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
        
        # Check problematic words first
        if lower in PROBLEMATIC_WORDS:
            self.debug_counters['problematic_word_fix'] += 1
            return {PROBLEMATIC_WORDS[lower]: 1.0}
        
        sc = script_candidate_score(tk)
        patt = pattern_hint_scores(lower)
        ch = char_pattern_score(lower)
        
        f_probs: Dict[str,float] = {}
        if self.model_mgr.fasttext:
            f_probs = self.model_mgr.fasttext_probs_batch([tk])[0]
        
        script = dominant_script(tk)
        sc_up = script.upper() if script else ""
        
        if sc_up == "LATIN":
            # Enhanced Indonesian detection
            ind_prefix = re.match(r'^(ber|me|mem|men|meng|meny|pe|per|pel|se|ke)[a-z]{2,}', lower)
            ind_suffix = re.search(r'(kan|lah|nya|kah|wan|man|i|an)$', lower)
            ind_ng = ('ng' in lower) or ('ny' in lower)
            trigger = lower in ID_TRIGGERS or lower in ID_COMPREHENSIVE_ROOTS
            composite_suffix = ((lower.endswith('kan') and 'ng' in lower) or 
                              (lower.startswith('ke') and lower.endswith('an') and len(lower)>=6))
            
            strong_morph = trigger or ind_prefix or composite_suffix or (ind_suffix and ind_ng)
            english_like = (english_pattern_match_count(lower) >= 2 or 
                          lower.endswith(('tion','ment','ance')) or 
                          lower in STRONG_EN_WORDS)
            
            if strong_morph and not english_like:
                if lower in ID_COMPREHENSIVE_ROOTS:
                    self.debug_counters['id_boost'] += 1
                    return {'id': 1.0}
                
                if f_probs.get('id', 0.0) > 0.50:
                    self.debug_counters['id_boost'] += 1
                    return {'id': 0.90, 'en': 0.10}
            
            # Strong English gate
            if (f_probs.get('en', 0.0) > 0.70 and lower.isascii() and 
                (english_pattern_match_count(lower) >= 1 or lower in STRONG_EN_WORDS) and 
                not strong_morph):
                return {'en': 1.0}
        
        fused = self._fuse(tk, {}, f_probs, patt, sc, ch)
        
        # Indonesian stem boost
        try:
            if _id_stemmer_available and _id_stemmer:
                stem = _id_stemmer.stem(lower)
                if stem and stem != lower and stem in ID_COMPREHENSIVE_ROOTS:
                    fused['id'] = max(fused.get('id', 0.0), 0.85)
                    if 'en' in fused and fused['en'] < 0.80:
                        fused['en'] *= 0.6
                    
                    totb = sum(fused.values())
                    if totb > 0:
                        for k in list(fused.keys()):
                            fused[k] /= totb
        except Exception:
            pass
        
        # Script-based fallback for strong scripts
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
            
            th = UNKNOWN_INJECT_MAXP_THRESHOLD * (1.0 - 0.7*neighbor_avg)
            th = min(0.10, th) if sc_up=="LATIN" else max(0.07, min(th, 0.25))
            
            if len(tokens[i])<=2:
                th = min(th, 0.05 if sc_up=="LATIN" else 0.10)
            
            if sc_up not in ['LATIN'] and maxp >= 0.18:
                enriched.append(new)
                continue
            
            strong_hint = any(v>=0.25 for v in new.values()) or len(new)>=2
            
            if maxp < th and not strong_hint:
                unk = max(UNKNOWN_MIN_PROB, (th - maxp)*0.7)
                total_exist = sum(new.values())
                scale = (1.0 - unk)/total_exist if total_exist>0 else 0.0
                
                for k in list(new.keys()):
                    new[k] *= scale
                new['unknown'] = unk
            
            enriched.append(new)
        
        return enriched

    def _enhanced_disambiguate(self, dists: List[Dict[str,float]], tokens: List[str]) -> List[Dict[str,float]]:
        n = len(tokens)
        has_kana = any(any(ch for ch in t if get_script(ch) in ("HIRAGANA","KATAKANA")) for t in tokens)
        
        # Enhanced sentence-level evidence
        pt_evidence = sum(1 for t in tokens if re.search(r'(ção|ções|são|viagem|coração|luz|ã|õ|nh|lh|ão|ões|futuro|natureza|patrimônio)', t.lower()))
        es_evidence = sum(1 for t in tokens if re.search(r'(ción|ciones|ñ|montaña|[áéíóúü]|también|entonces)', t.lower()))
        it_evidence = sum(1 for t in tokens if re.search(r'(zione|zioni|ggia|ggio|famiglia|ità)', t.lower()))
        fr_evidence = sum(1 for t in tokens if re.search(r'(tion|sion|étoile|nature|[çéèêàùôâî]|multiculturel)', t.lower()))
        de_evidence = sum(1 for t in tokens if re.search(r'[äöüß]|freiheit|natur|keit|heit|eleganz|katze|wesen|nationalismus', t.lower()))
        nl_evidence = sum(1 for t in tokens if ('ij' in t.lower() or re.search(r'(heid|lijk)', t.lower()) or t.lower() in ['het','een','van','schaduw','vrijheid']))
        
        # Indonesian context count
        id_morphology_count = 0
        for token in tokens:
            tl = token.lower()
            if (re.match(r'^(ber|me|mem|men|meng|pe|ke|se)', tl) or 
                re.search(r'(kan|nya|lah)$', tl) or
                tl in {'yang','dan','dengan','untuk','adalah','ini','itu'} or 
                tl in ID_COMPREHENSIVE_ROOTS):
                id_morphology_count += 1
        
        for i,(tok,dist) in enumerate(zip(tokens, dists)):
            if not dist: 
                continue
            
            tl = tok.lower()
            sc = dominant_script(tok)
            sc_up = sc.upper() if sc else ""
            
            self.debug_counters['enhanced_disambiguation'] += 1
            
            # Script hard filters
            if sc_up in ("ARABIC","CYRILLIC","DEVANAGARI","BENGALI","HANGUL","THAI"):
                allowed_map = {
                    "ARABIC": {"ar","ur"},
                    "CYRILLIC": {"ru"},
                    "DEVANAGARI": {"hi"},
                    "BENGALI": {"bn"},
                    "HANGUL": {"ko"},
                    "THAI": {"th"}
                }
                
                allowed = allowed_map.get(sc_up)
                if allowed:
                    for k in list(dist.keys()):
                        if k not in allowed:
                            dist.pop(k, None)
                    
                    tot = sum(dist.values())
                    if tot > 0:
                        inv = 1.0/tot
                        for k in list(dist.keys()):
                            dist[k] *= inv
            
            # Suppress implausible scripts on pure Latin
            if sc_up == 'LATIN':
                for l in ('ar','ur','zh','ja','ko','th','hi','bn','ru'):
                    dist.pop(l, None)
            
            # Enhanced sentence-level group priors
            if sc_up == 'LATIN' and len(dist) >= 2:
                # Strong word indicators - more aggressive English boost
                if tl in STRONG_EN_WORDS and 'en' in dist:
                    dist['en'] *= 2.5  # Increased from 2.0
                    for l in ('es', 'pt', 'fr', 'de', 'it', 'nl'):
                        if l in dist:
                            dist[l] *= 0.3  # More aggressive suppression
                elif tl in STRONG_PT_WORDS and 'pt' in dist:
                    dist['pt'] *= 2.5  # Increased from 2.0
                    for l in ('es', 'en', 'fr', 'it'):
                        if l in dist:
                            dist[l] *= 0.3  # More aggressive suppression
                elif tl in STRONG_ES_WORDS and 'es' in dist:
                    dist['es'] *= 2.5  # Increased from 2.0
                    for l in ('pt', 'en', 'fr', 'it'):
                        if l in dist:
                            dist[l] *= 0.3  # More aggressive suppression
                
                # Romance languages - enhanced with more aggressive Portuguese boost
                if any(l in dist for l in ('es','pt','it','fr')):
                    if pt_evidence >= 1 and 'pt' in dist:  # Lowered threshold
                        dist['pt'] *= 1.6  # Increased boost
                        for r in ('es','it','fr'):
                            if r in dist: 
                                dist[r] *= 0.6  # More aggressive suppression
                    elif es_evidence >= 2 and 'es' in dist:
                        dist['es'] *= 1.4
                        for r in ('pt','it','fr'):
                            if r in dist: 
                                dist[r] *= 0.7
                    elif it_evidence >= 2 and 'it' in dist:
                        dist['it'] *= 1.3
                        for r in ('es','pt','fr'):
                            if r in dist: 
                                dist[r] *= 0.75
                    elif fr_evidence >= 2 and 'fr' in dist:
                        dist['fr'] *= 1.3
                        for r in ('es','pt','it'):
                            if r in dist: 
                                dist[r] *= 0.75
                
                # Germanic languages
                if any(l in dist for l in ('de','nl')):
                    if de_evidence >= 2 and 'de' in dist:
                        dist['de'] *= 1.35
                        if 'nl' in dist: 
                            dist['nl'] *= 0.75
                    elif nl_evidence >= 2 and 'nl' in dist:
                        dist['nl'] *= 1.35
                        if 'de' in dist: 
                            dist['de'] *= 0.75
            
            # Enhanced Arabic/Urdu disambiguation
            if any(l in dist for l in ("ar","ur")):
                ur_boost = 0.0
                for neigh in tokens[max(0,i-2):min(n,i+3)]:
                    if any(ch in UR_SPECIFIC_CHARS for ch in neigh):
                        ur_boost += 0.25
                    elif any(word in neigh for word in UR_WORDS):
                        ur_boost += 0.20
                    elif any(ch in AR_SPECIFIC_CHARS for ch in neigh):
                        ur_boost -= 0.15
                
                if sc_up == 'LATIN' and 'ur' in dist:
                    dist['ur'] *= 0.3
                
                if ur_boost > 0 and 'ur' in dist:
                    dist['ur'] = min(1.0, dist['ur'] + ur_boost)
                    if 'ar' in dist:
                        dist['ar'] = max(0.0, dist['ar'] - ur_boost*0.6)
                elif ur_boost < 0 and 'ar' in dist:
                    dist['ar'] = min(1.0, dist['ar'] - ur_boost*0.5)
                    if 'ur' in dist:
                        dist['ur'] = max(0.0, dist['ur'] + ur_boost*0.3)
            
            # Enhanced Hindi vs Bengali
            if any(l in dist for l in ("hi","bn")):
                neighbor_scripts = [dominant_script(t) for t in tokens[max(0,i-2):min(n,i+3)] if t.strip()]
                dev = sum(1 for s in neighbor_scripts if s and s.upper()=="DEVANAGARI")
                beng = sum(1 for s in neighbor_scripts if s and s.upper()=="BENGALI")
                
                if dev > beng and "hi" in dist:
                    dist["hi"] = dist.get("hi",0.0)+0.25
                    if "bn" in dist: 
                        dist["bn"] = max(0.0, dist["bn"]-0.12)
                elif beng > dev and "bn" in dist:
                    dist["bn"] = dist.get("bn",0.0)+0.25
                    if "hi" in dist: 
                        dist["hi"] = max(0.0, dist["hi"]-0.12)
            
            # Enhanced Chinese vs Japanese
            if any(l in dist for l in ("zh","ja")):
                neighbor_scripts = [dominant_script(t) for t in tokens[max(0,i-2):min(n,i+3)] if t.strip()]
                local_kana = any(s and s.upper() in ("HIRAGANA","KATAKANA") for s in neighbor_scripts)
                han_count = sum(1 for s in neighbor_scripts if s and s.upper()=="HAN")
                
                # Enhanced character-level hints
                if any(ch in SIMP_ONLY_CHARS for ch in tok):
                    if 'zh' in dist: 
                        dist['zh'] = dist.get('zh',0.0) + 0.30
                    if 'ja' in dist: 
                        dist['ja'] = max(0.0, dist['ja'] - 0.15)
                elif any(ch in JP_SPECIFIC_CHARS for ch in tok):
                    if 'ja' in dist: 
                        dist['ja'] = dist.get('ja',0.0) + 0.35
                    if 'zh' in dist: 
                        dist['zh'] = max(0.0, dist['zh'] - 0.20)
                elif any(ch in TRAD_BIAS_CHARS for ch in tok):
                    if (local_kana or has_kana) and 'ja' in dist:
                        dist['ja'] = dist.get('ja',0.0) + 0.30
                        if 'zh' in dist: 
                            dist['zh'] = max(0.0, dist['zh'] - 0.15)
                    else:
                        if 'ja' in dist and 'zh' in dist and dist['zh'] - dist['ja'] < 0.20:
                            dist['ja'] += 0.12
                
                # Kana context handling
                if (local_kana or has_kana) and sc_up == 'HAN' and 'ja' not in dist:
                    dist['ja'] = 0.12
                
                if (local_kana or has_kana) and 'ja' in dist:
                    if not ('zh' in dist and dist['zh'] > dist['ja'] + 0.35):
                        dist['ja'] = dist.get('ja',0.0)+0.40
                        if 'zh' in dist: 
                            dist['zh'] = max(0.0, dist['zh']-0.25)
                elif han_count >= 2 and 'zh' in dist:
                    dist['zh'] = dist.get('zh',0.0)+0.18
                    if 'ja' in dist: 
                        dist['ja'] = max(0.0, dist['ja']-0.08)
                
                # Single character with kana context
                if len(tok) == 1 and has_kana and 'ja' in dist:
                    if not ('zh' in dist and dist['zh'] > 0.60):
                        dist['ja'] = max(dist['ja'], 0.75)
                        if 'zh' in dist and dist['zh'] < 0.80:
                            dist['zh'] *= 0.5
                        self.debug_counters['ja_han_force'] += 1
            
            # Enhanced Vietnamese diacritics handling
            if sc_up=='LATIN' and any(ch in VI_DIACRITICS for ch in tok):
                cur = dist.get('vi',0.0)
                floor = 0.45 if len(tok)>2 else 0.35
                dist['vi'] = max(cur, floor)
                
                # Suppress competitors more aggressively
                if 'en' in dist: 
                    dist['en'] *= 0.5
                for r in ('pt','es','fr','de','it','nl'):
                    if r in dist and dist[r] < 0.7:
                        dist[r] *= 0.6
                
                # Multi-syllable bonus
                if len(tl) >= 6 and re.search(r'[ăâêôơư]', tl) and not re.search(r'\s', tok):
                    syls = VI_SYLLABLE_REGEX.findall(tl)
                    if len(syls) >= 2:
                        dist['vi'] = max(dist['vi'], 0.55)
            
            # Enhanced Indonesian morphology with sentence context
            if sc_up=='LATIN' and ('en' in dist or 'id' in dist):
                morph_boost = 0.0
                
                # Base morphology
                if (tl.endswith(("kan","lah","nya","kah")) or 
                    tl in {"yang","dan","dengan","untuk","pada","adalah","ini","itu","mereka"} or 
                    re.match(r'ke[bcdfghjklmnpqrstvwxyz].+', tl)):
                    morph_boost = 0.25
                
                if re.match(r'^(ber|me|men|mem|meng|meny|pe|per|pel)', tl):
                    morph_boost = 0.30
                
                # Enhanced context: check if surrounded by Indonesian words
                context_boost = 0.0
                if id_morphology_count >= 2:
                    context_boost = 0.15 * min(id_morphology_count / len(tokens), 0.5)
                
                # Specific Indonesian words often misclassified
                if tl in ID_COMPREHENSIVE_ROOTS:
                    morph_boost = max(morph_boost, 0.35)
                
                # Apply boosts
                total_boost = morph_boost + context_boost
                if total_boost > 0 and 'id' in dist:
                    dist['id'] = min(1.0, dist.get('id', 0.0) + total_boost)
                    # Reduce English confidence proportionally
                    if 'en' in dist:
                        dist['en'] = max(0.0, dist['en'] - total_boost * 0.7)
                    self.debug_counters['id_boost'] += 1
                    
                if tl in ID_COMPREHENSIVE_ROOTS:
                    morph_boost = 0.40
                
                # Sentence context multiplier
                if id_morphology_count >= 3:
                    morph_boost *= 1.3
                elif id_morphology_count >= 2:
                    morph_boost *= 1.15
                
                if morph_boost > 0:
                    if 'id' in dist: 
                        dist['id'] = min(1.0, dist.get('id',0.0) + morph_boost)
                    if 'en' in dist: 
                        dist['en'] = max(0.0, dist['en'] - morph_boost*0.8)
            
            # Enhanced Portuguese vs Spanish suffix disambiguation
            if sc_up=='LATIN':
                # Stronger EN suppression on accented/bigrams
                accented = any(ord(c) > 127 for c in tok)
                bigram_hits = any(pat in tl for pat in ('ção','ções','cão','ción','ciones','ä','ö','ü','ï','ñ','ç','é','è','ê','ò','ô','ã','õ','ij'))
                
                if (accented or bigram_hits) and 'en' in dist and tl not in STRONG_EN_WORDS:
                    if dist['en'] < 0.9: 
                        dist['en'] *= 0.25  # More aggressive
                
                # Suffix-based disambiguation
                if re.search(r'(ção|ções)$', tl):
                    dist['pt'] = dist.get('pt',0.0)+0.35
                    if 'es' in dist: 
                        dist['es'] = max(0.0, dist['es']-0.20)
                
                if re.search(r'(ción|ciones)$', tl):
                    dist['es'] = dist.get('es',0.0)+0.35
                    if 'pt' in dist: 
                        dist['pt'] = max(0.0, dist['pt']-0.20)
                
                if tl.endswith('zione'):
                    dist['it'] = dist.get('it',0.0)+0.25
                    for r in ('es','pt'):
                        if r in dist: 
                            dist[r] *= 0.8
                
                if (tl.endswith('heid') or tl.endswith('lijk')) and 'de' in dist and 'nl' in dist:
                    dist['nl'] = dist.get('nl',0.0)+0.25
                    dist['de'] *= 0.8
            
            # Normalize
            total = sum(dist.values())
            if total > 0:
                inv = 1.0/total
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
            # Fallback initialize to unknown to avoid all -inf
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
                    
                    # Indonesian morphology emission bonuses
                    cur_tl = cur_tok.lower()
                    if cl == 'id':
                        if (re.search(r'(kan|lah|nya|kah)$', cur_tl) or 
                            re.match(r'^(ber|me|men|mem|meng|pe|ke|se)', cur_tl) or 
                            cur_tl in ID_COMPREHENSIVE_ROOTS):
                            clog += 0.15
                    
                    if cl in ('hi','en'):
                        if (re.search(r'(kan|lah|nya|kah)$', cur_tl) or 
                            re.match(r'^(ber|me|men|mem|meng|pe|ke|se)', cur_tl)):
                            clog -= 0.15
                    
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
            ('id','en'): 0.15, ('en','id'): 0.15,
        }

        prev_sc = dominant_script(prev_tok)
        prev_sc_up = prev_sc.upper() if prev_sc else ''
        cur_tl = cur_tok.lower()

        if (pl, cl) in implausible_transitions:
            w = implausible_transitions[(pl, cl)]
            if (pl, cl) == ('en','hi') and cur_sc_up != 'DEVANAGARI':
                trans += w
            elif (pl, cl) in [('hi','id'), ('id','hi')] and prev_sc_up == cur_sc_up == 'LATIN':
                trans += w
            elif (pl, cl) in [('id','en'), ('en','id')]:
                if (pl, cl) == ('en','id') and (re.match(r'^(ber|me|men|mem|meng)', cur_tl) or cur_tl in ID_COMPREHENSIVE_ROOTS):
                    trans += w * 0.2  # allow easier switch to ID if morphology suggests it
                else:
                    trans += w
            else:
                trans += w * 0.7

        # Small discount for related languages (easier switch)
        try:
            if self._are_related(pl, cl):
                trans = max(0.0, trans - 0.08)
        except Exception:
            pass

        return trans
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
        for lang,score in dists[0].items():
            dp[0][idx[lang]] = math.log(max(score, MIN_LANG_SCORE))
        
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
                    
                    # Indonesian morphology emission bonuses
                    cur_tl = cur_tok.lower()
                    if cl == 'id':
                        if (re.search(r'(kan|lah|nya|kah)$', cur_tl) or 
                            re.match(r'^(ber|me|men|mem|meng|pe|ke|se)', cur_tl) or 
                            cur_tl in ID_COMPREHENSIVE_ROOTS):
                            clog += 0.15
                    
                    if cl in ('hi','en'):
                        if (re.search(r'(kan|lah|nya|kah)$', cur_tl) or 
                            re.match(r'^(ber|me|men|mem|meng|pe|ke|se)', cur_tl)):
                            clog -= 0.15
                    
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

    def _sentence_guess(self, tokens: List[str], fused: List[Dict[str,float]]) -> Optional[str]:
        # Enhanced sentence-level language detection
        votes = Counter()
        for d in fused:
            if d:
                best = max(d.items(), key=lambda x: x[1])[0]
                votes[best] += 1
        
        if votes:
            top, cnt = votes.most_common(1)[0]
            if cnt / max(1, len(fused)) >= 0.25:  # Lowered threshold
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
        counts = Counter(res)
        
        # Find dominant Latin script language
        latin_set = {'en','id','fr','es','pt','it','de','nl','pl','tr','vi'}
        dom_lang, dom_count = None, 0
        
        for l, c in counts.items():
            if l in latin_set and c > dom_count:
                dom_lang, dom_count = l, c
        
        n = len(tokens)
        # More conservative threshold than b.py
        if dom_lang and dom_count >= max(7, int(0.8 * n)):
            for i in range(n):
                sc = dominant_script(tokens[i])
                if (not sc) or sc.upper() == 'LATIN':
                    if res[i] == 'unknown' or res[i] not in latin_set:
                        # Only consolidate longer tokens to avoid over-merging
                        if len(tokens[i]) > 3:
                            res[i] = dom_lang
        
        # Preserve strong English words
        for i, t in enumerate(tokens):
            if t.lower() in STRONG_EN_WORDS:
                res[i] = 'en'
        
        return res

    def detect_languages(self, text: str) -> List[Tuple[str,str]]:
        if not text or not text.strip():
            return []
        
        tokens = tokenize(unicodedata.normalize('NFC', text))
        if not tokens:
            return [(text.strip(), "unknown")]
        
        # Pre-fuse with enhanced heuristics
        pre = [self._pre_fuse_token(t) for t in tokens]
        
        # Apply models and fuse
        fused = self._apply_models_and_fuse(tokens, pre)
        
        # Heuristic fallback for low-confidence tokens
        for i, dist in enumerate(fused):
            if not dist or max(dist.values(), default=0.0) < 0.12:
                fb = {}
                tl = tokens[i].lower()
                for d in (pattern_hint_scores(tl), script_candidate_score(tokens[i]), char_pattern_score(tl)):
                    for k, v in d.items():
                        fb[k] = max(fb.get(k, 0), v)
                if fb:
                    tot = sum(fb.values())
                    fused[i] = {k: v/tot for k, v in fb.items()}
        
        # Unknown injection (conservative)
        fused = self._adaptive_unknown_injection(fused, tokens)
        
        # Enhanced disambiguation
        fused = self._enhanced_disambiguate(fused, tokens)
        
        # Enhanced DP smoothing
        chosen = self._enhanced_dp(fused, tokens)
        
        # Post-processing
        unk_ratio = sum(1 for c in chosen if c == 'unknown') / len(chosen)
        if unk_ratio >= UNKNOWN_RATIO_FALLBACK:
            guess = self._sentence_guess(tokens, fused)
            if guess and guess in TOP_20_LANGS:
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
        
        # Merge adjacent spans - but be more careful with Latin scripts
        merged: List[Tuple[str,str]] = []
        cur_lang, buf = None, []
        
        for tok, lang in zip(tokens, chosen):
            tok_script = dominant_script(tok)
            prev_script = dominant_script(buf[-1]) if buf else None
            
            if cur_lang is None:
                cur_lang, buf = lang, [tok]
            elif lang == cur_lang:
                # For Latin scripts, be more conservative about merging
                if tok_script and tok_script.upper() == 'LATIN':
                    # Only merge if languages are very similar or if tokens are very short
                    should_merge = False
                    if lang in ['es', 'pt'] and len(tok) <= 3:  # Short function words
                        should_merge = True
                    elif lang in ['en'] and (tok.lower() in STRONG_EN_WORDS or len(tok) <= 2):
                        should_merge = True
                    elif lang == cur_lang and len(tok) <= 2:  # Very short tokens
                        should_merge = True
                    elif not prev_script or prev_script == tok_script:
                        # Same script, but only merge if confidence is high or tokens are related
                        should_merge = True
                    
                    if should_merge:
                        buf.append(tok)
                    else:
                        merged.append((" ".join(buf), cur_lang))
                        cur_lang, buf = lang, [tok]
                else:
                    # Non-Latin scripts - use original logic
                    if not tok_script or not prev_script or tok_script == prev_script:
                        buf.append(tok)
                    else:
                        merged.append((" ".join(buf), cur_lang))
                        cur_lang, buf = lang, [tok]
            else:
                merged.append((" ".join(buf), cur_lang))
                cur_lang, buf = lang, [tok]
        
        if buf and cur_lang is not None:
            merged.append((" ".join(buf), cur_lang))
        
        return [(seg.strip(), lang) for seg, lang in merged if seg.strip()]

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
    test_text = "Hello world! Bonjour le monde! Hola mundo! こんにちは世界！"
    result = detect_languages(test_text)
    for segment, lang in result:
        print(f"{lang}: {segment}")
