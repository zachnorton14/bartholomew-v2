"""Mask mathematical / scientific notation out of the anachronism (delta_peak) score.

The vintage model saw pre-1930 math mostly as OCR garbage, so clean notation is
out-of-distribution and spikes its per-word surprise — flagging STEM questions as
"anachronistic" when only the notation is unusual, not the prose register. We drop
notation-bearing words before scoring and judge the PROSE register alone.

Design: an ALLOWLIST, not a denylist. We keep only clean prose words and mask
everything else. A denylist of symbols can always miss a rare one (√, ∂, ∮, ⅗, a
glued fragment like `a)/(1`) and lose a valuable STEM question; keeping only real
words cannot miss anything by construction.

A "prose word" is, after stripping outer punctuation, a run of letters with optional
internal hyphen/apostrophe (well-known, Boyle's), or an ordinal (20th, 3rd). Anything
containing a digit, symbol, Greek letter, prime, or gluing (H2O, x^2, 50°, a′,
(a′−a)/(1+aa′), 1776) is masked. Numbers and symbols aren't register tells, so
masking them costs no register signal.

    from synth.notation_mask import is_notation, mask_wordbits
    is_notation("(a′−a)/(1+aa′)")  -> True
    is_notation("perpendicular")    -> False
"""
import re

_OUTER = ".,;:!?()[]{}\"'“”‘’…—-«»"                  # punctuation stripped from word edges
_PROSE = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*\Z")  # letters, internal hyphen/apostrophe only
_ORDINAL = re.compile(r"\d+(?:st|nd|rd|th)\Z", re.I)     # 1st, 20th — genuine prose


def is_notation(word):
    """True unless the word is clean prose (so all notation is masked, none missed)."""
    core = word.strip().strip(_OUTER)
    if not core:
        return False                      # pure punctuation (—, parens) — prose, keep
    if _ORDINAL.fullmatch(core):
        return False                      # ordinal — keep
    return not bool(_PROSE.fullmatch(core))


def has_notation(text):
    """True if any word in the text carries notation (the question is at-risk)."""
    return any(is_notation(w) for w in text.split())


def notation_count(text):
    """How many words carry notation (density — higher = more at-risk)."""
    return sum(is_notation(w) for w in text.split())


def mask_wordbits(wb):
    """Given score_word_bits output {'words','bits','bytes'}, return a copy with
    notation words removed so delta_peak/mean see prose only."""
    keep = [i for i, w in enumerate(wb["words"]) if not is_notation(w)]
    return {
        "words": [wb["words"][i] for i in keep],
        "bits": [wb["bits"][i] for i in keep],
        "bytes": [wb["bytes"][i] for i in keep],
    }
