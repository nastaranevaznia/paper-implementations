"""Fog layer: local complaint categorization (paper Sec. 3.2, Eq. 1-2).

Each healthcare center runs a lightweight classifier on the raw complaint
text, keeps the text locally (it is deleted right after the label is
extracted) and forwards *only* the category label + non-sensitive metadata
(center ID, timestamp/week, department type) to the cloud.

The default classifier below is a TF-IDF + LogisticRegression surrogate for
the paper's lightweight edge LLM (e.g. Mistral-7B / RoBERTa class).  To plug
a real HuggingFace model, replace ``FogClassifier`` internals with, e.g.::

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    model = AutoModelForSequenceClassification.from_pretrained(..., num_labels=4)
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# ------------------------------------------------------------- Eq. 1 and 2
def eq1_softmax(logits):
    """P(y=k | X) = exp(z_k) / sum_j exp(z_j)."""
    z = np.asarray(logits, dtype=float)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def eq2_argmax(probs):
    """y_hat = argmax_k P(y=k | X)."""
    return int(np.argmax(probs))


# ------------------------------------------------------------ preprocessing
def preprocess(text):
    return " ".join(str(text).lower().split())


# ------------------------------------------------------ lightweight model
class FogClassifier:
    """Lightweight edge classifier (stand-in for the fog LLM)."""

    def __init__(self):
        self.vec = TfidfVectorizer(max_features=4000, ngram_range=(1, 2))
        self.clf = LogisticRegression(max_iter=2000)

    def train(self, texts, labels):
        X = self.vec.fit_transform([preprocess(t) for t in texts])
        self.clf.fit(X, labels)
        return self

    def predict_proba(self, text):
        X = self.vec.transform([preprocess(text)])
        # Explicit Eq. 1 over the model's raw decision scores.
        return eq1_softmax(self.clf.decision_function(X))[0]

    def predict(self, text):
        # Explicit Eq. 2.
        return eq2_argmax(self.predict_proba(text))


# ----------------------------------------------------------------- fog node
class FogNode:
    """One healthcare center's fog node."""

    def __init__(self, center_id, classifier):
        self.center_id = center_id
        self.classifier = classifier

    def process(self, complaint_text, week, department):
        """Classify locally; return only (label + metadata) to the cloud."""
        probs = self.classifier.predict_proba(complaint_text)
        label = self.classifier.predict(complaint_text)
        record = {
            "label": label,
            "center_id": self.center_id,
            "week": week,
            "department": department,
        }
        # Privacy: the raw text never leaves the healthcare center.
        del complaint_text
        return record, probs
