"""Label-free pairwise evidence. Missing comparisons are NaN, never perfect matches."""
import math
import re
import unicodedata
from functools import lru_cache
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from ber.data.normalize import fold_latin_accents

VERSION = 'pair-features-v1'


@lru_cache(maxsize=100_000)
def tokens(text):
    # Unlike Python regex \w, keep combining marks attached to Indic letters.
    pieces, word = [], []
    for char in text:
        if unicodedata.category(char)[0] in 'LMN' or char == '_':
            word.append(char)
        elif word:
            pieces.append(''.join(word))
            word = []
    if word:
        pieces.append(''.join(word))
    return set(pieces)


def jaccard(a, b):
    return len(a & b) / len(a | b) if a and b else math.nan


@lru_cache(maxsize=100_000)
def numbers(text):
    # Preserve leading zeros. These are numeric tokens, NOT inferred house/postcodes.
    return {''.join(str(unicodedata.decimal(c)) for c in s) for s in re.findall(r'\d+', text)}


@lru_cache(maxsize=100_000)
def trigrams(text):
    return {text[i:i+3] for i in range(max(0, len(text)-2))}


@lru_cache(maxsize=100_000)
def accent_folded(text):
    return fold_latin_accents(text)


def compare_field(a, b, prefix):
    present = bool(a and b)
    ta, tb = tokens(a), tokens(b)
    na, nb = numbers(a), numbers(b)
    values = {
        'query_missing': not bool(a), 'target_missing': not bool(b),
        'both_missing': not bool(a) and not bool(b),
        'query_length': len(a), 'target_length': len(b),
        'query_token_count': len(ta), 'target_token_count': len(tb),
        'exact': float(a == b) if present else math.nan,
        'accent_folded_exact': float(accent_folded(a) == accent_folded(b)) if present else math.nan,
        'length_ratio': min(len(a),len(b))/max(len(a),len(b)) if present else math.nan,
        'edit_similarity': Levenshtein.normalized_similarity(a,b) if present else math.nan,
        'fuzzy_ratio': fuzz.ratio(a,b)/100 if present else math.nan,
        'token_sort_ratio': fuzz.token_sort_ratio(a,b)/100 if present else math.nan,
        'token_set_ratio': fuzz.token_set_ratio(a,b)/100 if present else math.nan,
        'token_jaccard': jaccard(ta,tb),
        'token_overlap_min': len(ta & tb)/min(len(ta),len(tb)) if ta and tb else math.nan,
        'char_trigram_jaccard': jaccard(trigrams(a),trigrams(b)),
        'query_has_numbers': bool(na), 'target_has_numbers': bool(nb),
        'number_sets_equal': float(na == nb) if na and nb else math.nan,
        'number_jaccard': jaccard(na,nb),
        'numbers_disjoint': float(na.isdisjoint(nb)) if na and nb else math.nan,
    }
    return {f'{prefix}_{k}':v for k,v in values.items()}


def pair_features(query, target):
    result = {}
    for field in ('name','address'):
        result.update(compare_field(query[f'{field}_normalized'], target[f'{field}_normalized'], field))
        a, b = set(query[f'{field}_scripts']) - {'Other'}, set(target[f'{field}_scripts']) - {'Other'}
        result[f'{field}_script_overlap'] = float(bool(a & b)) if a and b else math.nan
    a,b = query['country'],target['country']
    result['country_equal'] = float(a==b) if a and b else math.nan
    return result
