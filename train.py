# # YouTube Spam Collection – Part C (Text classification)
# LR + Linear SVM, tuned on the **validation set only**, test set used **once** at the end.
# Bonus: Voting / Stacking ensembles and Logistic Regression from scratch (NumPy).
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix)
from sklearn.model_selection import GridSearchCV, PredefinedSplit, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from text_utils import ScratchLogReg, clean_text
MODELS_DIR, REPORTS_DIR = Path("models"), Path("reports")
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)
LABELS = ["ham", "spam"]

Youtube01 = pd.read_csv("youtube+spam+collection/Youtube01-Psy.csv")
Youtube02 = pd.read_csv("youtube+spam+collection/Youtube02-KatyPerry.csv")
Youtube03 = pd.read_csv("youtube+spam+collection/Youtube03-LMFAO.csv")
Youtube04 = pd.read_csv("youtube+spam+collection/Youtube04-Eminem.csv")
Youtube05 = pd.read_csv("youtube+spam+collection/Youtube05-Shakira.csv")

df = pd.concat(
    [Youtube01, Youtube02, Youtube03, Youtube04, Youtube05],
    ignore_index=True
)

print("raw shape:", df.shape)
print("missing values:\n", df[["CONTENT", "CLASS"]].isna().sum())

# ## 2. Basic cleaning of the dataframe
before = len(df)
df = df.dropna(subset=["CONTENT", "CLASS"]).drop_duplicates(subset="CONTENT").reset_index(drop=True)
print(df["CLASS"].value_counts().rename({0: "ham", 1: "spam"}))

# ## 3. Split 70 / 15 / 15
# X_tmp: 85% (Training + Validation)
X, y = df["CONTENT"], df["CLASS"]
X_tmp, X_test, y_tmp, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=13
)

X_train, X_val, y_train, y_val = train_test_split(
    X_tmp, y_tmp, test_size=0.17647,
    stratify=y_tmp, random_state=13
)

# GridSearchCV that scores on the fixed validation set only (train rows = -1, val rows = 0)
X_tv = pd.concat([X_train, X_val])
y_tv = pd.concat([y_train, y_val])
val_split = PredefinedSplit([-1] * len(X_train) + [0] * len(X_val))


def tune(name, pipe, grid):
    gs = GridSearchCV(pipe, grid, cv=val_split, scoring="f1", refit=False, n_jobs=-1)
    gs.fit(X_tv, y_tv)
    print(f"{name}: best val F1 = {gs.best_score_:.4f}  params = {gs.best_params_}")
    best = pipe.set_params(**gs.best_params_).fit(X_train, y_train)  # refit on TRAIN only
    return best, gs

# ## 4. Models + hyper-parameter tuning (validation set)

def tfidf():
    return TfidfVectorizer(preprocessor=clean_text, sublinear_tf=True)

tfidf_grid = {"tfidf__ngram_range": [(1, 1), (1, 2)], "tfidf__min_df": [1, 2]}
lr_pipe, lr_gs = tune(
    "LogisticRegression",
    Pipeline([("tfidf", tfidf()), ("clf", LogisticRegression(max_iter=2000, random_state=13))]),
    {**tfidf_grid, "clf__C": [0.1, 1, 10, 100]})

svm_pipe, svm_gs = tune(
    "LinearSVC",
    Pipeline([("tfidf", tfidf()), ("clf", LinearSVC(random_state=13))]),
    {**tfidf_grid, "clf__C": [0.05, 0.1, 0.5, 1, 5]})

vocab_size = len(lr_pipe.named_steps["tfidf"].vocabulary_)
print("TF-IDF vocabulary size (LR pipeline):", vocab_size)

#  5. Bonus A – ensembles (same TF-IDF config as the tuned LR, base models fit on train only)

tf_params = {
    "ngram_range": lr_pipe.named_steps["tfidf"].ngram_range,
    "min_df": lr_pipe.named_steps["tfidf"].min_df
}


def base_models():
    return [
        ("lr", 
         LogisticRegression(C=lr_pipe.named_steps["clf"].C,
                             max_iter=2000, 
                             random_state=13)),
        ("svm", 
         CalibratedClassifierCV(LinearSVC(C=svm_pipe.named_steps["clf"].C, 
                                          random_state=13), 
                                          cv=3)),
        ("nb", 
         MultinomialNB(alpha=0.1)),
    ]


vote_pipe = Pipeline([("tfidf", TfidfVectorizer(preprocessor=clean_text, sublinear_tf=True, **tf_params)),
                      ("clf", VotingClassifier(base_models(), voting="soft"))]).fit(X_train, y_train)
stack_pipe = Pipeline([("tfidf", TfidfVectorizer(preprocessor=clean_text, sublinear_tf=True, **tf_params)),
                       ("clf", StackingClassifier(base_models(), cv=5,
                                                  final_estimator=LogisticRegression(random_state=13)))]
                      ).fit(X_train, y_train)

# validation scores 
for n, m in [("LR", lr_pipe), ("SVM", svm_pipe), ("Voting", vote_pipe), ("Stacking", stack_pipe)]:
    print(f"{n:9s} val acc = {accuracy_score(y_val, m.predict(X_val)):.4f}")


# ## 6. Bonus B – Logistic Regression from scratch (NumPy) vs sklearn
# Tuned on validation too.

scratch_pipe, scratch_gs = tune(
    "ScratchLogReg",
    Pipeline([("tfidf", TfidfVectorizer(preprocessor=clean_text, sublinear_tf=True, **tf_params)),
              ("clf", ScratchLogReg(n_iter=500))]),
    {"clf__lr": [0.5, 1.0, 5.0, 10.0], "clf__l2": [1e-4, 1e-3, 1e-2]})

plt.figure(figsize=(5, 3))
plt.plot(scratch_pipe.named_steps["clf"].loss_)
plt.xlabel("iteration"); plt.ylabel("log-loss"); plt.title("Scratch LR training loss")
plt.tight_layout(); plt.savefig(REPORTS_DIR / "scratch_loss.png", dpi=120); plt.close()

# ## 7. FINAL evaluation on the test set (run once)

all_models = {"Logistic Regression": lr_pipe, "Linear SVM": svm_pipe,
              "Voting (soft)": vote_pipe, "Stacking": stack_pipe,
              "LR from scratch (NumPy)": scratch_pipe}
results = {"vocab_size": vocab_size,
           "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test),
           "models": {}}

for name, model in all_models.items():
    pred = model.predict(X_test)
    rep = classification_report(y_test, pred, target_names=LABELS, output_dict=True)
    cm = confusion_matrix(y_test, pred)
    results["models"][name] = {"accuracy": accuracy_score(y_test, pred),
                               "report": rep, "confusion_matrix": cm.tolist()}
    print(f"\n=== {name} ===  acc = {accuracy_score(y_test, pred):.4f}")
    print(classification_report(y_test, pred, target_names=LABELS))
    print(cm)
    ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(cmap="Blues")
    plt.title(name); plt.tight_layout()
    plt.savefig(REPORTS_DIR / f"cm_{name.split(' (')[0].replace(' ', '_')}.png", dpi=120)
    plt.close()

# sklearn LR vs scratch LR side by side (bonus requirement)
print("\nsklearn LR acc:", results["models"]["Logistic Regression"]["accuracy"],
      "| scratch LR acc:", results["models"]["LR from scratch (NumPy)"]["accuracy"])

# ## 8. Save models + metrics for the GUI
for name, model in all_models.items():
    joblib.dump(model, MODELS_DIR / (name.split(" (")[0].replace(" ", "_").lower() + ".joblib"))
(MODELS_DIR / "results.json").write_text(json.dumps(results, indent=2))
print("saved:", sorted(p.name for p in MODELS_DIR.iterdir()))
