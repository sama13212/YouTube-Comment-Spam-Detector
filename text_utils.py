import html
import re

import numpy as np
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

URL_RE = re.compile(r"(?:https?://|www\.)[^\s\"'<>]+")
TAG_RE = re.compile(r"<[^>]*>")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")


def clean_text(text):
    """HTML unescape -> lowercase -> URLs to a token -> strip HTML tags

    -> drop numbers/punctuation -> drop stopwords."""
    text = html.unescape(str(text)).lower()
    text = URL_RE.sub(" url ", text)    
    text = TAG_RE.sub(" ", text)
    text = NON_ALPHA_RE.sub(" ", text)
    return " ".join(w for w in text.split() if w not in ENGLISH_STOP_WORDS)

#### Linear reg form scratch
class ScratchLogReg(BaseEstimator, ClassifierMixin):
    def __init__(self, lr=1.0, n_iter=500, l2=1e-3):
        self.lr = lr
        self.n_iter = n_iter
        self.l2 = l2

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        t = (y == self.classes_[1]).astype(float)
        n, d = X.shape
        self.w_, self.b_, self.loss_ = np.zeros(d), 0.0, []
        for _ in range(self.n_iter):
            p = expit(X @ self.w_ + self.b_) # Prediction
            self.w_ -= self.lr * (X.T @ (p - t) / n + self.l2 * self.w_)
            self.b_ -= self.lr * np.mean(p - t)
            #ـ Binary Cross Entropy Loss.
            eps = 1e-12
            self.loss_.append(-np.mean(t * np.log(p + eps) + (1 - t) * np.log(1 - p + eps)))
        return self

    def decision_function(self, X):
        return X @ self.w_ + self.b_
    # بترجع ال اجتمالات نفسها قبل ال sigmoid 
    #z = Xw+b

    def predict_proba(self, X):
        p1 = expit(self.decision_function(X))
        return np.column_stack([1 - p1, p1])
    # احتماليه كل class 

    def predict(self, X):
        return self.classes_[(self.decision_function(X) > 0).astype(int)]
    # تحويل ال prob ل label 