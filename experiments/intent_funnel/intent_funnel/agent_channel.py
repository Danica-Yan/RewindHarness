"""Agent-proposal channel: read the agent's message right before each user turn and predict which write
tools the next steps will need ("I can swap them for cheaper items — shall I proceed?").

Why: on held-out runs most remaining misses were writes the *user* never asks for in words; the agent
proposes them and the user agrees. A user-text router cannot see those.

Training labels (from replays, train split only): agent_before[j] gets the intents of writes executed after
user turn j .. j+HORIZON (the user confirms at turn j, or within the next couple of turns).
The channel only ever ADDS write intents (it never removes tools and never triggers fallback):
    p_combined[c] = max(p_user[c], p_agent[c])   for write intents c
"""
import numpy as np

from .taxonomy import ACTION_TO_INTENT, WRITE_INTENTS

HORIZON = 2


def agent_examples(replays, horizon=HORIZON):
    X, Y = [], []
    for rp in replays:
        ab = rp.get("agent_before") or [""] * len(rp["user_turns"])
        for j, a in enumerate(ab):
            labs = sorted({ACTION_TO_INTENT[w["tool"]] for w in rp["writes"]
                           if j <= w["after_user_turn"] <= j + horizon} & set(WRITE_INTENTS))
            X.append(a)
            Y.append(labs)
    return X, Y


class AgentChannel:
    def __init__(self, C=2.0):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import MultiLabelBinarizer
        self.mlb = MultiLabelBinarizer(classes=WRITE_INTENTS)
        self.pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                                  OneVsRestClassifier(LogisticRegression(C=C, max_iter=2000, class_weight="balanced")))

    def fit(self, replays):
        X, Y = agent_examples(replays)
        self.pipe.fit(X, self.mlb.fit_transform(Y))
        return self

    def predict_proba(self, texts):
        P = self.pipe.predict_proba(texts)
        P[[i for i, t in enumerate(texts) if not t]] = 0.0  # no agent message yet (first turn)
        return P


def combine(P_user, user_classes, P_agent, weight=1.0):
    """max-combine the agent channel into the user router's write-intent columns."""
    P = P_user.copy()
    for k, c in enumerate(WRITE_INTENTS):
        j = user_classes.index(c)
        P[:, j] = np.maximum(P[:, j], weight * P_agent[:, k])
    return P
