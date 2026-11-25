# YouTube Spam Collection – Part C (text classification)

Data: https://archive.ics.uci.edu/dataset/380/youtube+spam+collection
Unzip it so the 5 CSVs are in `youtube+spam+collection/` (or pass the folder as an argument).

```bash
pip install -r requirements.txt
python train.py [path/to/csv/folder]   # trains, tunes on validation, evaluates once on test, saves models/
streamlit run app.py                   # GUI (loads saved models, no retraining)
```
`train.py` uses `# %%` cells: open in VS Code / Jupyter, or `jupytext --to ipynb train.py`.


Design decisions (mention in the report / discussion)

Duplicates. The raw data has 1956 comments. Exact full-row duplicates are only ~3, but drop_duplicates(subset="CONTENT") removes 196 rows (-> 1760: 919 ham / 841 spam), because spam is often copy-pasted by different accounts (same text, different author/ID). We remove text duplicates before splitting so the same comment cannot appear in both train and test (data leakage), which would inflate the test scores. Trade-off: we lose some real samples (mostly spam) and the class balance shifts slightly. To check the effect, remove subset="CONTENT" and compare the results.



Naive Bayes inside the ensemble. MultinomialNB is not one of the required models and is never evaluated on its own; it is only a third base model inside the Voting/Stacking ensembles (bonus). Reason: LR and Linear SVM are both linear models and learn very similar decision boundaries, so combining only them adds little. NB is a different (generative, probabilistic) model, so its mistakes differ, which is what lets an ensemble improve. A third voter also avoids ties in voting, and NB trains in seconds. Limitation: alpha=0.1 was set by hand and not tuned on the validation set. TODO: either tune alpha on the validation set, or remove NB and say so.