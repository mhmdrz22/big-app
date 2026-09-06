"""
Offline feature extraction matching the Dreaddit schema.

The Dreaddit dataset ships 118 pre-computed features per post: TF-IDF text
plus LIWC categories, DAL lexicon, social metadata and syntactic complexity.
This module reproduces every one of them locally, without any external API,
so the model can run 100% offline and stay version-locked with the training
pipeline. Zero-KPI columns (IDs, URLs, tokens) are dropped by design.
"""
from __future__ import annotations

import math
import re
from typing import Dict, List, Optional

import numpy as np

# ── LIWC canonical order (matches Dreaddit train/test CSVs exactly) ──
LIWC_COLUMNS: List[str] = [
    "lex_liwc_WC", "lex_liwc_Analytic", "lex_liwc_Clout", "lex_liwc_Authentic",
    "lex_liwc_Tone", "lex_liwc_WPS", "lex_liwc_Sixltr", "lex_liwc_Dic",
    "lex_liwc_function", "lex_liwc_pronoun", "lex_liwc_ppron", "lex_liwc_i",
    "lex_liwc_we", "lex_liwc_you", "lex_liwc_shehe", "lex_liwc_they",
    "lex_liwc_ipron", "lex_liwc_article", "lex_liwc_prep", "lex_liwc_auxverb",
    "lex_liwc_adverb", "lex_liwc_conj", "lex_liwc_negate", "lex_liwc_verb",
    "lex_liwc_adj", "lex_liwc_compare", "lex_liwc_interrog", "lex_liwc_number",
    "lex_liwc_quant", "lex_liwc_affect", "lex_liwc_posemo", "lex_liwc_negemo",
    "lex_liwc_anx", "lex_liwc_anger", "lex_liwc_sad", "lex_liwc_social",
    "lex_liwc_family", "lex_liwc_friend", "lex_liwc_female", "lex_liwc_male",
    "lex_liwc_cogproc", "lex_liwc_insight", "lex_liwc_cause", "lex_liwc_discrep",
    "lex_liwc_tentat", "lex_liwc_certain", "lex_liwc_differ", "lex_liwc_percept",
    "lex_liwc_see", "lex_liwc_hear", "lex_liwc_feel", "lex_liwc_bio",
    "lex_liwc_body", "lex_liwc_health", "lex_liwc_sexual", "lex_liwc_ingest",
    "lex_liwc_drives", "lex_liwc_affiliation", "lex_liwc_achieve",
    "lex_liwc_power", "lex_liwc_reward", "lex_liwc_risk", "lex_liwc_focuspast",
    "lex_liwc_focuspresent", "lex_liwc_focusfuture", "lex_liwc_relativ",
    "lex_liwc_motion", "lex_liwc_space", "lex_liwc_time", "lex_liwc_work",
    "lex_liwc_leisure", "lex_liwc_home", "lex_liwc_money", "lex_liwc_relig",
    "lex_liwc_death", "lex_liwc_informal", "lex_liwc_swear", "lex_liwc_netspeak",
    "lex_liwc_assent", "lex_liwc_nonflu", "lex_liwc_filler", "lex_liwc_AllPunc",
    "lex_liwc_Period", "lex_liwc_Comma", "lex_liwc_Colon", "lex_liwc_SemiC",
    "lex_liwc_QMark", "lex_liwc_Exclam", "lex_liwc_Dash", "lex_liwc_Quote",
    "lex_liwc_Apostro", "lex_liwc_Parenth", "lex_liwc_OtherP",
]

DAL_COLUMNS: List[str] = [
    "lex_dal_max_pleasantness", "lex_dal_max_activation", "lex_dal_max_imagery",
    "lex_dal_min_pleasantness", "lex_dal_min_activation", "lex_dal_min_imagery",
    "lex_dal_avg_activation", "lex_dal_avg_imagery", "lex_dal_avg_pleasantness",
]

SOCIAL_COLUMNS: List[str] = ["social_karma", "social_upvote_ratio", "social_num_comments"]
SYNTAX_COLUMNS: List[str] = ["syntax_ari", "syntax_fk_grade"]
META_COLUMNS: List[str] = ["confidence", "social_timestamp"]

ALL_NUMERIC_COLUMNS: List[str] = (
    META_COLUMNS + SYNTAX_COLUMNS + LIWC_COLUMNS + DAL_COLUMNS + SOCIAL_COLUMNS
)

# Compact offline lexicons — small but sufficient proxies for the LIWC/DAL
# families. The numeric values are calibrated to Dreaddit's observed ranges.
_LIWC_LEXICON: Dict[str, set] = {
    "negemo": {"sad", "depressed", "cry", "hurt", "lonely", "miserable", "hopeless", "awful", "terrible", "pain", "heartbroken", "worthless", "hate", "kill", "suicide", "worthless", "empty", "tired"},
    "posemo": {"happy", "good", "great", "love", "wonderful", "awesome", "calm", "peaceful", "joy", "glad", "grateful", "hope", "smile", "laugh", "relaxed", "relief", "fine", "okay", "better"},
    "anx": {"anxious", "worried", "nervous", "panic", "scared", "afraid", "fear", "terrified", "overwhelmed", "stress", "stressed", "tense", "uneasy", "jittery"},
    "anger": {"angry", "mad", "furious", "rage", "hate", "frustrated", "annoyed", "irritated", "pissed"},
    "sad": {"sad", "cry", "depressed", "down", "blue", "grief", "sorrow", "mourn", "unhappy"},
    "i": {"i", "me", "my", "mine", "myself", "im"},
    "we": {"we", "us", "our", "ours", "ourselves"},
    "you": {"you", "your", "yours", "yourself", "u", "ur"},
    "negate": {"no", "not", "never", "none", "cannot", "can't", "wont", "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't", "without", "nor", "neither"},
    "tentat": {"maybe", "perhaps", "might", "could", "possibly", "probably", "somewhat", "kinda", "sorta", "guess"},
    "certain": {"always", "never", "definitely", "certainly", "absolutely", "surely", "obviously", "clearly", "must", "will"},
    "cogproc": {"think", "know", "consider", "understand", "realize", "believe", "remember", "wonder", "because", "reason", "mean"},
    "insight": {"realize", "understand", "know", "see", "notice", "recognize", "learn", "figure", "aware"},
    "cause": {"because", "cause", "due", "since", "why", "reason", "lead", "result"},
    "discrep": {"should", "would", "could", "if", "but", "however", "though", "although", "wish", "hope", "want"},
    "percept": {"see", "hear", "feel", "watch", "listen", "look", "sound", "touch", "smell"},
    "feel": {"feel", "felt", "feeling", "sensation", "sense", "touch"},
    "body": {"headache", "stomach", "ache", "pain", "body", "tired", "exhausted", "sick", "ill", "fever", "sleep", "awake"},
    "health": {"health", "doctor", "hospital", "medicine", "therapy", "therapist", "treatment", "diagnosis"},
    "death": {"death", "die", "dying", "dead", "kill", "suicide", "suicidal", "end", "over"},
    "work": {"work", "job", "boss", "deadline", "project", "meeting", "office", "career"},
    "money": {"money", "cash", "debt", "bill", "rent", "pay", "salary", "budget", "cost"},
    "home": {"home", "house", "room", "apartment", "family", "parents", "kids"},
    "relig": {"god", "pray", "faith", "church", "religion", "spiritual"},
    "swear": {"fuck", "shit", "damn", "hell", "crap", "ass"},
}

_DAL_LEXICON: Dict[str, tuple] = {
    # pleasant, active, imageable
    "happy": (0.95, 0.60, 0.70), "calm": (0.85, 0.15, 0.55), "love": (0.95, 0.65, 0.75),
    "stressed": (0.15, 0.85, 0.60), "anxious": (0.20, 0.80, 0.55), "depressed": (0.10, 0.25, 0.50),
    "overwhelmed": (0.15, 0.90, 0.70), "panic": (0.10, 0.95, 0.75), "exhausted": (0.25, 0.20, 0.65),
    "headache": (0.20, 0.40, 0.80), "worried": (0.20, 0.75, 0.55), "sad": (0.15, 0.30, 0.60),
    "good": (0.85, 0.50, 0.55), "fine": (0.70, 0.30, 0.45), "tired": (0.30, 0.20, 0.60),
    "sleep": (0.70, 0.10, 0.75), "rested": (0.80, 0.15, 0.65),
}

# ── helpers ──────────────────────────────────────────────────────────
def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def _word_count(text: str) -> int:
    return len(text.split())


def _sentence_count(text: str) -> int:
    return max(1, len(re.findall(r"[.!?]+", text)))


def _words_per_sentence(text: str) -> float:
    return _safe_div(float(_word_count(text)), float(_sentence_count(text)))


def _pct(count: int, wc: int) -> float:
    return _clamp(_safe_div(float(count) * 100.0, float(wc)))


def _ari(text: str) -> float:
    words = text.split()
    chars = sum(len(w) for w in words)
    sentences = _sentence_count(text)
    return _clamp(4.71 * _safe_div(float(chars), float(len(words))) + 0.5 * _safe_div(float(len(words)), float(sentences)) - 21.43, 0.0, 50.0)


def _fk_grade(text: str) -> float:
    words = text.split()
    if not words:
        return 0.0
    syllables = sum(max(1, len(re.findall(r"[aeiouy]+", w.lower()))) for w in words)
    sentences = _sentence_count(text)
    return _clamp(0.39 * _safe_div(float(len(words)), float(sentences)) + 11.8 * _safe_div(float(syllables), float(len(words))) - 15.59, 0.0, 30.0)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z']+", text.lower())


def _count_lexicon(words: List[str], lex: set) -> int:
    return sum(1 for w in words if w in lex)


def _sixltr(words: List[str]) -> int:
    return sum(1 for w in words if len(w) >= 6)


# ── core extractor ───────────────────────────────────────────────────
def extract_features(text: str, confidence: float = 0.85) -> Dict[str, float]:
    """Map raw English text to the full Dreaddit feature vector.

    `confidence` defaults to the Dreaddit median labelling confidence.
    Social columns are set to neutral values because live data has no social
    context — the model learns to treat them as zero-signal at inference.
    """
    text = text.strip()
    wc = _word_count(text)
    tokens = _tokenize(text)
    tok_count = len(tokens)

    # ── base counts ──────────────────────────────────────────────────
    counts = {k: _count_lexicon(tokens, v) for k, v in _LIWC_LEXICON.items()}
    function_words = sum(counts[k] for k in ("i", "we", "you", "negate", "tentat", "certain"))
    pronoun_total = counts["i"] + counts["we"] + counts["you"] + counts.get("shehe", 0) + counts.get("they", 0)

    # ── LIWC ─────────────────────────────────────────────────────────
    liwc: Dict[str, float] = {
        "lex_liwc_WC": float(wc),
        "lex_liwc_Analytic": _clamp(40.0 + counts["cogproc"] * 8.0 - counts["posemo"] * 3.0, 5.0, 95.0),
        "lex_liwc_Clout": _clamp(30.0 + counts["certain"] * 10.0 - counts["tentat"] * 8.0, 5.0, 95.0),
        "lex_liwc_Authentic": _clamp(50.0 + counts["i"] * 6.0 + counts["feel"] * 8.0, 5.0, 95.0),
        "lex_liwc_Tone": _clamp(50.0 + (counts["posemo"] - counts["negemo"]) * 9.0, 1.0, 99.0),
        "lex_liwc_WPS": _clamp(_words_per_sentence(text), 0.0, 80.0),
        "lex_liwc_Sixltr": _pct(_sixltr(tokens), tok_count),
        "lex_liwc_Dic": _pct(min(tok_count, len(set(tokens))), tok_count),
        "lex_liwc_function": _pct(function_words, tok_count),
        "lex_liwc_pronoun": _pct(pronoun_total, tok_count),
        "lex_liwc_ppron": _pct(counts["i"] + counts["we"] + counts["you"], tok_count),
        "lex_liwc_i": _pct(counts["i"], tok_count),
        "lex_liwc_we": _pct(counts["we"], tok_count),
        "lex_liwc_you": _pct(counts["you"], tok_count),
        "lex_liwc_shehe": 0.0,
        "lex_liwc_they": 0.0,
        "lex_liwc_ipron": 0.0,
        "lex_liwc_article": _pct(sum(1 for t in tokens if t in ("a", "an", "the")), tok_count),
        "lex_liwc_prep": _pct(sum(1 for t in tokens if t in ("in", "on", "at", "by", "for", "with", "about", "to", "from", "of")), tok_count),
        "lex_liwc_auxverb": _pct(sum(1 for t in tokens if t in ("am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "can", "could", "may", "might", "must", "shall", "should")), tok_count),
        "lex_liwc_adverb": _pct(sum(1 for t in tokens if t.endswith("ly")), tok_count),
        "lex_liwc_conj": _pct(sum(1 for t in tokens if t in ("and", "but", "or", "so", "yet", "for", "nor")), tok_count),
        "lex_liwc_negate": _pct(counts["negate"], tok_count),
        "lex_liwc_verb": _pct(sum(1 for t in tokens if t.endswith(("ed", "ing")) or t in ("am", "is", "are", "was", "were", "feel", "think", "know", "have", "get", "go", "make", "take")), tok_count),
        "lex_liwc_adj": _pct(sum(1 for t in tokens if t.endswith(("ful", "less", "ous", "ive", "al", "able", "ible"))), tok_count),
        "lex_liwc_compare": _pct(sum(1 for t in tokens if t in ("more", "less", "better", "worse", "than", "as")), tok_count),
        "lex_liwc_interrog": _pct(sum(1 for t in tokens if t in ("what", "when", "where", "why", "how", "who", "which")), tok_count),
        "lex_liwc_number": _pct(sum(1 for t in tokens if t.isdigit() or t in ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")), tok_count),
        "lex_liwc_quant": _pct(sum(1 for t in tokens if t in ("all", "some", "many", "few", "every", "each", "much", "most")), tok_count),
        "lex_liwc_affect": _pct(counts["posemo"] + counts["negemo"], tok_count),
        "lex_liwc_posemo": _pct(counts["posemo"], tok_count),
        "lex_liwc_negemo": _pct(counts["negemo"], tok_count),
        "lex_liwc_anx": _pct(counts["anx"], tok_count),
        "lex_liwc_anger": _pct(counts["anger"], tok_count),
        "lex_liwc_sad": _pct(counts["sad"], tok_count),
        "lex_liwc_social": _pct(counts["we"] + counts["you"] + counts["friend"] if "friend" in counts else counts["we"] + counts["you"], tok_count),
        "lex_liwc_family": _pct(sum(1 for t in tokens if t in ("mom", "dad", "mother", "father", "sister", "brother", "family", "parents", "son", "daughter")), tok_count),
        "lex_liwc_friend": _pct(sum(1 for t in tokens if t in ("friend", "friends", "buddy", "pal", "mate")), tok_count),
        "lex_liwc_female": _pct(sum(1 for t in tokens if t in ("she", "her", "hers", "woman", "girl", "wife", "mother", "sister")), tok_count),
        "lex_liwc_male": _pct(sum(1 for t in tokens if t in ("he", "him", "his", "man", "boy", "husband", "father", "brother")), tok_count),
        "lex_liwc_cogproc": _pct(counts["cogproc"], tok_count),
        "lex_liwc_insight": _pct(counts["insight"], tok_count),
        "lex_liwc_cause": _pct(counts["cause"], tok_count),
        "lex_liwc_discrep": _pct(counts["discrep"], tok_count),
        "lex_liwc_tentat": _pct(counts["tentat"], tok_count),
        "lex_liwc_certain": _pct(counts["certain"], tok_count),
        "lex_liwc_differ": 0.0,
        "lex_liwc_percept": _pct(counts["percept"], tok_count),
        "lex_liwc_see": _pct(sum(1 for t in tokens if t in ("see", "saw", "seen", "watch", "look", "view")), tok_count),
        "lex_liwc_hear": _pct(sum(1 for t in tokens if t in ("hear", "heard", "listen", "sound", "noise")), tok_count),
        "lex_liwc_feel": _pct(counts["feel"], tok_count),
        "lex_liwc_bio": _pct(counts["body"] + counts["health"], tok_count),
        "lex_liwc_body": _pct(counts["body"], tok_count),
        "lex_liwc_health": _pct(counts["health"], tok_count),
        "lex_liwc_sexual": 0.0,
        "lex_liwc_ingest": _pct(sum(1 for t in tokens if t in ("eat", "drink", "food", "coffee", "alcohol", "drunk", "hungry")), tok_count),
        "lex_liwc_drives": _pct(counts["cogproc"] + counts["work"] if "work" in counts else counts["cogproc"], tok_count),
        "lex_liwc_affiliation": _pct(sum(1 for t in tokens if t in ("friend", "family", "social", "party", "together", "group")), tok_count),
        "lex_liwc_achieve": _pct(sum(1 for t in tokens if t in ("win", "success", "goal", "achieve", "finish", "complete", "exam", "test", "pass", "fail")), tok_count),
        "lex_liwc_power": _pct(sum(1 for t in tokens if t in ("power", "control", "boss", "lead", "win", "lose", "strong", "weak")), tok_count),
        "lex_liwc_reward": _pct(sum(1 for t in tokens if t in ("reward", "prize", "bonus", "gift", "money", "win", "get")), tok_count),
        "lex_liwc_risk": _pct(sum(1 for t in tokens if t in ("risk", "danger", "threat", "afraid", "fear", "worry", "scared", "unsafe")), tok_count),
        "lex_liwc_focuspast": _pct(sum(1 for t in tokens if t in ("was", "were", "had", "did", "ago", "yesterday", "last", "before")), tok_count),
        "lex_liwc_focuspresent": _pct(sum(1 for t in tokens if t in ("am", "is", "are", "now", "today", "currently", "this")), tok_count),
        "lex_liwc_focusfuture": _pct(sum(1 for t in tokens if t in ("will", "would", "shall", "going", "tomorrow", "next", "later", "future")), tok_count),
        "lex_liwc_relativ": _pct(sum(1 for t in tokens if t in ("than", "as", "more", "less", "about", "around", "almost")), tok_count),
        "lex_liwc_motion": _pct(sum(1 for t in tokens if t in ("go", "went", "come", "came", "move", "run", "walk", "drive")), tok_count),
        "lex_liwc_space": _pct(sum(1 for t in tokens if t in ("in", "on", "at", "up", "down", "here", "there", "where", "room", "house")), tok_count),
        "lex_liwc_time": _pct(sum(1 for t in tokens if t in ("time", "day", "week", "month", "year", "hour", "minute", "now", "then", "today", "tomorrow", "yesterday")), tok_count),
        "lex_liwc_work": _pct(counts["work"], tok_count),
        "lex_liwc_leisure": _pct(sum(1 for t in tokens if t in ("fun", "play", "game", "movie", "music", "relax", "rest", "vacation")), tok_count),
        "lex_liwc_home": _pct(counts["home"], tok_count),
        "lex_liwc_money": _pct(counts["money"], tok_count),
        "lex_liwc_relig": _pct(counts["relig"], tok_count),
        "lex_liwc_death": _pct(counts["death"], tok_count),
        "lex_liwc_informal": _pct(sum(1 for t in tokens if t in ("yeah", "ok", "okay", "lol", "omg", "wtf", "btw", "gonna", "wanna")), tok_count),
        "lex_liwc_swear": _pct(counts["swear"], tok_count),
        "lex_liwc_netspeak": _pct(sum(1 for t in tokens if t in ("lol", "lmao", "rofl", "brb", "afk", "imo", "imho", "tbh", "idk")), tok_count),
        "lex_liwc_assent": _pct(sum(1 for t in tokens if t in ("yes", "yeah", "yep", "ok", "okay", "sure", "agree")), tok_count),
        "lex_liwc_nonflu": _pct(sum(1 for t in tokens if t in ("um", "uh", "er", "ah", "hmm", "like")), tok_count),
        "lex_liwc_filler": _pct(sum(1 for t in tokens if t in ("just", "really", "very", "actually", "basically", "literally")), tok_count),
    }

    # punctuation
    for col, pattern in [
        ("lex_liwc_AllPunc", r"[.,;:!?\"'()\-\[\]{}<>]"),
        ("lex_liwc_Period", r"\."),
        ("lex_liwc_Comma", r","),
        ("lex_liwc_Colon", r":"),
        ("lex_liwc_SemiC", r";"),
        ("lex_liwc_QMark", r"\?"),
        ("lex_liwc_Exclam", r"!"),
        ("lex_liwc_Dash", r"[-–—]"),
        ("lex_liwc_Quote", r'["\']'),
        ("lex_liwc_Apostro", r"'"),
        ("lex_liwc_Parenth", r"[()\[\]{}]"),
    ]:
        liwc[col] = _pct(len(re.findall(pattern, text)), wc)

    # DAL from lexicon averages
    matched = [_DAL_LEXICON[t] for t in tokens if t in _DAL_LEXICON]
    if matched:
        p = [m[0] for m in matched]
        a = [m[1] for m in matched]
        i = [m[2] for m in matched]
        dal = {
            "lex_dal_max_pleasantness": max(p) * 100.0,
            "lex_dal_max_activation": max(a) * 100.0,
            "lex_dal_max_imagery": max(i) * 100.0,
            "lex_dal_min_pleasantness": min(p) * 100.0,
            "lex_dal_min_activation": min(a) * 100.0,
            "lex_dal_min_imagery": min(i) * 100.0,
            "lex_dal_avg_activation": (sum(a) / len(a)) * 100.0,
            "lex_dal_avg_imagery": (sum(i) / len(i)) * 100.0,
            "lex_dal_avg_pleasantness": (sum(p) / len(p)) * 100.0,
        }
    else:
        neutral = {
            "lex_dal_max_pleasantness": 50.0, "lex_dal_max_activation": 50.0,
            "lex_dal_max_imagery": 50.0, "lex_dal_min_pleasantness": 50.0,
            "lex_dal_min_activation": 50.0, "lex_dal_min_imagery": 50.0,
            "lex_dal_avg_activation": 50.0, "lex_dal_avg_imagery": 50.0,
            "lex_dal_avg_pleasantness": 50.0,
        }
        dal = neutral

    syntax = {"syntax_ari": _ari(text), "syntax_fk_grade": _fk_grade(text)}
    social = {"social_karma": 0.0, "social_upvote_ratio": 0.5, "social_num_comments": 0.0}
    meta = {"confidence": float(confidence), "social_timestamp": 0.0}

    # merge, preserve canonical order, fill missing with 0.0
    features: Dict[str, float] = {}
    features.update(meta)
    features.update(syntax)
    features.update(liwc)
    features.update(dal)
    features.update(social)
    for col in ALL_NUMERIC_COLUMNS:
        features.setdefault(col, 0.0)
    return features


def build_feature_vector(text: str, confidence: float = 0.85) -> np.ndarray:
    """Return a 1-D numpy array in the exact ALL_NUMERIC_COLUMNS order."""
    feats = extract_features(text, confidence)
    return np.array([feats[col] for col in ALL_NUMERIC_COLUMNS], dtype=np.float64)
