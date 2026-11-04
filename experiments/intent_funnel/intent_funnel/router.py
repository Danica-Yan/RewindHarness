"""Tier-0 (rules) and Tier-1 (classifier) routers with a shared decision rule.

Decision rule (same for every probabilistic model, so they are comparable):
  include every intent with p >= t_include       (multi-label: union of tool sets)
  if max p < gate (or nothing included)          -> FALLBACK (full prompt, all tools)
The gate is the knob swept for the risk-coverage curve.
"""
import re

import numpy as np

from .taxonomy import INTENTS, expand_groups

KEYWORDS = {
    "cancel": r"\bcancel",
    "modify_items": r"\b(change|modify|switch|swap|update)\b.*\b(item|size|color|colour|option|version|model)\b",
    "modify_order_address": r"\b(order|shipping|deliver\w*)\b.*\baddress\b|\baddress\b.*\b(order|shipping)\b",
    "modify_payment": r"\b(payment method|pay with|paid with|change.*payment|gift card|paypal|credit card)\b",
    "modify_user_address": r"\b(my|account|default|profile)\b.*\baddress\b",
    "return": r"\b(return|refund|send (it|them) back)\b",
    "exchange": r"\bexchang|\bswap\b.*\bfor\b|\breplace\b",
    "info_query": r"\b(where is|track|status|how (long|much)|when will|what is|check)\b",
    "transfer": r"\b(human|real person|agent|representative|manager|supervisor)\b",
}


class KeywordRouter:
    name = "rules"

    def __init__(self):
        self.pats = {k: re.compile(v, re.I) for k, v in KEYWORDS.items()}

    def route(self, text):
        hits = [k for k, p in self.pats.items() if p.search(text)]
        return hits or ["continue"]  # no hit -> keep current route (never falls back)


class RulesAsProba:
    """Expose the keyword router through predict_proba (1.0 for every hit) so it uses the same evaluation."""
    name = "rules"

    def __init__(self):
        self.kr = KeywordRouter()

    @property
    def classes(self):
        return list(INTENTS)

    def predict_proba(self, texts):
        P = np.zeros((len(texts), len(INTENTS)))
        for i, t in enumerate(texts):
            for c in self.kr.route(t):
                P[i, INTENTS.index(c)] = 1.0
        return P


GROUP_EXPANSION = True  # module switch so ablations can turn it off


def decide(probs, classes, t_include=0.25, gate=0.5, groups=None):
    """probs: (n, C) -> list of routes (list[str] or None for fallback).
    groups: expand ambiguous sibling intents (taxonomy.ROUTE_GROUPS); defaults to GROUP_EXPANSION."""
    groups = GROUP_EXPANSION if groups is None else groups
    routes = []
    for p in probs:
        if p.max() < gate:
            routes.append(None)
            continue
        sel = [classes[j] for j in np.where(p >= t_include)[0]]
        routes.append((expand_groups(sel) if groups else sel) or None)
    return routes


class TfidfLRRouter:
    name = "tfidf_lr"

    def __init__(self, C=4.0):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.pipeline import FeatureUnion, make_pipeline
        from sklearn.preprocessing import MultiLabelBinarizer
        self.mlb = MultiLabelBinarizer(classes=INTENTS)
        feats = FeatureUnion([
            ("w", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True)),
        ])
        self.pipe = make_pipeline(feats, OneVsRestClassifier(
            LogisticRegression(C=C, max_iter=2000, class_weight="balanced"), n_jobs=2))

    def fit(self, texts, labels):
        self.pipe.fit(texts, self.mlb.fit_transform(labels))
        return self

    def predict_proba(self, texts):
        return self.pipe.predict_proba(texts)

    @property
    def classes(self):
        return list(self.mlb.classes_)


_SENT = re.compile(r"(?<=[.!?;])\s+|\s+(?:and also|also|then|and then)\s+", re.I)


def sentences(text, min_words=3):
    parts = [s.strip() for s in _SENT.split(text) if s and s.strip()]
    merged = []
    for s in parts:  # glue tiny fragments to the previous sentence
        if merged and len(s.split()) < min_words:
            merged[-1] += " " + s
        else:
            merged.append(s)
    return merged or [text]


class SentenceUnion:
    """Wrap any model with predict_proba: classify each sentence, aggregate per class.
    in-scope intents: max over sentences (any sentence can add a route);
    out_of_scope: min over sentences (the turn is OOS only if every sentence is)."""

    def __init__(self, base, oos="out_of_scope"):
        self.base, self.name = base, base.name + "+sent"
        self.oos_idx = base.classes.index(oos)

    @property
    def classes(self):
        return self.base.classes

    def predict_proba(self, texts):
        out = []
        for t in texts:
            p = self.base.predict_proba(sentences(t))
            agg = p.max(0)
            agg[self.oos_idx] = p[:, self.oos_idx].min()
            out.append(agg)
        return np.vstack(out)
