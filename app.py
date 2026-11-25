import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

import text_utils

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"


st.write("Models directory:", MODELS_DIR)
st.write("Directory exists:", MODELS_DIR.exists())
st.write("Files:", list(MODELS_DIR.glob("*")) if MODELS_DIR.exists() else [])


LABELS = {0: "Not spam (ham)", 1: "SPAM"}
FILES = {"Logistic Regression": "logistic_regression.joblib", "Linear SVM": "linear_svm.joblib",
         "Voting (soft)": "voting.joblib", "Stacking": "stacking.joblib",
         "LR from scratch (NumPy)": "lr_from_scratch.joblib"}

st.set_page_config(page_title="YouTube Spam Detector", page_icon="💬")
st.title("💬 YouTube Comment Spam Detector")


@st.cache_resource
def load(name):
    return joblib.load(MODELS_DIR / FILES[name])


if not (MODELS_DIR / "results.json").exists():
    st.error("No trained models found. Run `python train.py` first.")
    st.stop()
results = json.loads((MODELS_DIR / "results.json").read_text())

tab_pred, tab_res = st.tabs(["Predict", "Test results"])

with tab_pred:
    name = st.selectbox("Model", list(FILES))
    text = st.text_area("Type a YouTube comment", height=120,
                        placeholder="e.g. Check out my channel and subscribe!!!")
    if st.button("Predict", type="primary"):
        if not text.strip():
            st.warning("Please type a comment first.")
        else:
            model = load(name)
            pred = int(model.predict([text])[0])
            (st.error if pred else st.success)(f"Prediction: **{LABELS[pred]}**")
            if hasattr(model, "predict_proba"):
                st.progress(float(model.predict_proba([text])[0][1]),
                            text="spam probability")
            else:
                st.caption(f"SVM decision score: {model.decision_function([text])[0]:+.3f} "
                           "(> 0 means spam)")
            cleaned = text_utils.clean_text(text)
            st.caption(f"After cleaning: `{cleaned or '(empty)'}`")

with tab_res:
    st.write(f"Vocabulary size: **{results['vocab_size']}** | train/val/test = "
             f"{results['n_train']}/{results['n_val']}/{results['n_test']}")
    rows = []
    for m, r in results["models"].items():
        rep = r["report"]
        rows.append({"Model": m, "Accuracy": r["accuracy"],
                     "Spam precision": rep["spam"]["precision"], "Spam recall": rep["spam"]["recall"],
                     "Spam F1": rep["spam"]["f1-score"], "Ham F1": rep["ham"]["f1-score"]})
    st.dataframe(pd.DataFrame(rows).set_index("Model").round(4))
    pick = st.selectbox("Confusion matrix for", list(results["models"]), key="cm")
    st.dataframe(pd.DataFrame(results["models"][pick]["confusion_matrix"],
                              index=["true ham", "true spam"], columns=["pred ham", "pred spam"]))
