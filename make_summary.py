"""Builds results_summary.md from outputs/results.json (written by replicate.py)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
R = json.loads((ROOT / "outputs" / "results.json").read_text())
LOC, PAP = R["local_smell_dataset"], R["paper_dataset"]
PAPER = {
    "XGBoost": dict(accuracy=0.98, precision=0.95, recall=0.82, f1=0.88, mcc=0.87, roc_auc=0.91),
    "Random Forest": dict(accuracy=0.97, precision=0.89, recall=0.82, f1=0.85, mcc=0.84, roc_auc=0.90),
}
METRICS = [("accuracy", "Accuracy"), ("precision", "Precision (positive class)"),
           ("recall", "Recall (positive class)"), ("f1", "F1 (positive class)"), ("mcc", "MCC"),
           ("roc_auc", "ROC-AUC (probability scores)"), ("roc_auc_hard", "ROC-AUC (hard 0/1 labels)")]
NB_RF = "Random Forest | notebook config (100 trees, no depth limit), 80/20"
f = lambda v: "–" if v is None else f"{v:.4f}"
f2 = lambda v: "–" if v is None else f"{v:.2f}"


def per_class_table(m, names):
    rep = m["report"]
    rows = ["| Class | Precision | Recall | F1-score | Support |", "|---|---:|---:|---:|---:|"]
    for k in names + ["macro avg", "weighted avg"]:
        r = rep[k]
        b = "**" if k == "weighted avg" else ""
        rows.append(f"| {b}{k}{b} | {b}{r['precision']:.4f}{b} | {b}{r['recall']:.4f}{b} | "
                    f"{b}{r['f1-score']:.4f}{b} | {int(r['support']):,} |")
    return "\n".join(rows)


def report_text(m, names):
    rep = m["report"]
    lines = [f"{'':>16}{'precision':>11}{'recall':>9}{'f1-score':>10}{'support':>9}", ""]
    for k in names:
        r = rep[k]
        lines.append(f"{k:>16}{r['precision']:>11.4f}{r['recall']:>9.4f}{r['f1-score']:>10.4f}{int(r['support']):>9}")
    n = int(rep["weighted avg"]["support"])
    lines += ["", f"{'accuracy':>16}{'':>11}{'':>9}{rep['accuracy']:>10.4f}{n:>9}"]
    for k in ["macro avg", "weighted avg"]:
        r = rep[k]
        lines.append(f"{k:>16}{r['precision']:>11.4f}{r['recall']:>9.4f}{r['f1-score']:>10.4f}{n:>9}")
    return "```\n" + "\n".join(lines) + "\n```"


def model_section(ds, folder, names, name, slug, m):
    cm = m["cm"]
    return f"""#### {name}

| Accuracy | MCC | ROC-AUC (proba) | ROC-AUC (hard labels) |
|---:|---:|---:|---:|
| {m['accuracy']:.4f} | {m['mcc']:.4f} | {m['roc_auc']:.4f} | {m['roc_auc_hard']:.4f} |

Per-class and averaged precision / recall / F1:

{per_class_table(m, names)}

Confusion matrix (rows = actual, columns = predicted):

| | Pred. {names[0]} | Pred. {names[1]} |
|---|---:|---:|
| **Actual {names[0]}** | {cm[0][0]:,} (TN) | {cm[0][1]:,} (FP) |
| **Actual {names[1]}** | {cm[1][0]:,} (FN) | {cm[1][1]:,} (TP) |

![{name} confusion matrix](outputs/{folder}/confusion_matrix_{slug}.png)

Classification report:

{report_text(m, names)}
"""


def dataset_block(ds, folder, names, extra=None):
    out = []
    items = list(ds["main"].items()) + (extra or [])
    for name, m in items:
        slug = {"XGBoost": "xgboost", "Random Forest": "random_forest"}.get(
            name, "random_forest_notebook_config")
        out.append(model_section(ds, folder, names, name, slug, m))
    return "\n".join(out)


def comparison_rows():
    rows = ["| Model | Metric | Paper's reported (Table 15) | Reproduced: paper dataset | Reproduced: provided CSV (code smells) |",
            "|---|---|---:|---:|---:|"]
    for model in ["XGBoost", "Random Forest"]:
        for k, label in METRICS:
            rows.append(f"| {model} (Table 4 config) | {label} | {f2(PAPER[model].get(k))} | "
                        f"{f(PAP['main'][model][k])} | {f(LOC['main'][model][k])} |")
    nb = PAP["sensitivity"][NB_RF]
    for k, label in METRICS:
        rows.append(f"| Random Forest (authors' notebook config) | {label} | {f2(PAPER['Random Forest'].get(k))} | "
                    f"{f(nb[k])} | – |")
    return "\n".join(rows)


def sensitivity_rows():
    rows = ["| Dataset | Run | Accuracy | Precision | Recall | F1 | MCC | ROC-AUC (proba) | ROC-AUC (hard) |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for dsname, ds in [("Paper", PAP), ("Provided CSV", LOC)]:
        for name, m in ds["main"].items():
            rows.append(f"| {dsname} | **{name}, main run (80/20, exact-dup removal)** | " +
                        " | ".join(f(m.get(k)) for k, _ in METRICS) + " |")
        for name, m in ds["sensitivity"].items():
            rows.append(f"| {dsname} | {name.replace(' | ', ', ')} | " + " | ".join(f(m.get(k)) for k, _ in METRICS) + " |")
    return "\n".join(rows)


px, prf = PAP["main"]["XGBoost"], PAP["main"]["Random Forest"]
lx, lrf = LOC["main"]["XGBoost"], LOC["main"]["Random Forest"]
nb = PAP["sensitivity"][NB_RF]
tsx = PAP["sensitivity"]["XGBoost | paper protocol: TimeSeriesSplit(5) mean"]
tsr = PAP["sensitivity"]["Random Forest | paper protocol: TimeSeriesSplit(5) mean"]
ddx = PAP["sensitivity"][[k for k in PAP["sensitivity"] if k.startswith("XGBoost | feature-level")][0]]
lddx = LOC["sensitivity"][[k for k in LOC["sensitivity"] if k.startswith("XGBoost | feature-level")][0]]
ps, ls = PAP["split"], LOC["split"]
pinfo, linfo = PAP["info"], LOC["info"]

md = f"""# Results Summary: Replicating Diouf et al. (2026)

**Paper:** M. Diouf, E. Toe, M. Grichi, H. Nakouri, F. Jaafar, *"Predicting Software Security Bugs Using Machine Learning and Quality Metrics: An Empirical Study"*, Computers, Materials & Continua 87(3), 2026. doi:10.32604/cmc.2026.077139 (`TSP_CMC_77139.pdf`).

**Models reproduced:** XGBoost (the paper's best model) and Random Forest (its second-best).
**Protocol:** 80/20 stratified train/test split, `random_state=42`, with SMOTE applied to the training data only.
**Code:** `replicate.py` runs the experiments and `make_summary.py` writes this report.

---

## 0. Read this first: the CSV in this folder is not the paper's dataset

| | Paper (Sec. 3.2.3) | `multi-smell-dataset-v1_2.csv` (provided) |
|---|---|---|
| Task | File-level **security-bug** prediction | Class/method-level **code-smell** detection |
| Rows | 338,442 files (33,294 buggy, 9.84%) | 107,554 rows (9,918 with any smell, 9.22%) |
| Features | 25 SciTools *Understand* metrics (CountStmtExe, SumCyclomatic, …) + `extension`, `Ecosystem` | 14 Halstead/complexity metrics (Volume, Effort, Difficulty, Cyclomatic Complexity, …) + `Project` |
| Labels | `buggy` (0/1) | `Long method`, `God class`, `Feature envy`, `Data class` (0/1 each) |
| Ecosystems / language | 7 ecosystems, multi-language | 89 Apache Java projects |

None of the paper's 25 metrics, nor its `buggy` label, appear in the provided file. Because the paper links to its replication package (github.com/MdioufDataScientist/PredictSecBugs), I **ran the pipeline twice**:

* **A. The provided CSV**, as the task instructs. The binary target `smelly` is 1 if any of the four smell flags is 1. This is the closest analogue to the paper's binary `buggy` label, and the positive rate is similar (9.22% vs 9.84%). These results show the paper's method applied to a different problem, so they **cannot** be compared to Table 15 like-for-like.
* **B. The authors' public dataset** (`paper_dataset/result_total_bon.csv`, downloaded from the repository above, together with the authors' notebook `ML.ipynb`). This is the real replication, and the comparison in Section 6 is based mainly on it.

---

## 1. The paper in brief

* **Problem:** can static, file-level software-quality metrics predict which source files contain security bugs (RQ3)? RQ1 covers feature importance and RQ2 covers thresholds, such as the "3x rule".
* **Data:** 7,685 security bugs from OSV across GHSA, PyPI, OSS-Fuzz, npm, Packagist, Maven and NuGet. Files modified in a security-fixing commit are labelled `buggy = 1`; all other files in the same repository snapshot are labelled 0. There are 25 Understand metrics per file, plus metadata (extension, ecosystem, commit date/SHA, before/after source).
* **Preprocessing (paper Sec. 4.3 and the authors' notebook):** remove duplicate files; check data types and missing values (none found); one-hot encode categoricals; mean-impute and StandardScale numerics; apply SMOTE to training folds only (it beat random oversampling and ADASYN); keep temporal order.
* **Models (11):** KNN (k=3), Decision Tree (depth 3), Random Forest (10 trees, depth 3, max_features=1), MLP, AdaBoost (50), XGBoost (defaults), Naive Bayes, QDA, Linear SVM, RBF SVM and Logistic Regression.
* **Evaluation:** 5-fold time-series cross-validation; Accuracy, Precision, Recall, F1, MCC and ROC-AUC. The best model is XGBoost: Acc 0.98, P 0.95, R 0.82, F1 0.88, MCC 0.87, AUC 0.91.

---

## 2. Dataset inspection

Full dumps are in `outputs/<dataset>/data_inspection.txt`.

### A. Provided CSV: `multi-smell-dataset-v1_2.csv`
* **Shape:** {linfo['raw_shape'][0]:,} rows x {linfo['raw_shape'][1] - 1} columns (618 MB, mostly the raw `Code` text column). Exact duplicate rows across all columns, including the code text: **{linfo['exact_duplicates_removed']:,}**.
* **Columns and types:**
  * Identifiers / text: `File`, `Project`, `Class` and `Code` (strings).
  * Numeric features (14):
    * int: Logical Lines, Distinct/Total Operators, Distinct/Total Operands, Vocabulary, Length, Cyclomatic Complexity
    * float: Calculated Length, Volume, Difficulty, Effort, Time Required, Bugs (Halstead's estimate, not a label)
  * Labels (4 x int 0/1): Long method, God class, Feature envy, Data class.
* **Missing values:** none in any column.
* **Label distribution:** Long method 1,566 (1.46%); God class 4,333 (4.03%); Feature envy 1,996 (1.86%); Data class 3,284 (3.05%).
* **Target used:** `smelly` = any smell. 97,636 clean (90.78%) and 9,918 smelly (9.22%), so imbalanced at about 1:10.
* **Other notes:** the data is heavily right-skewed; for example Effort has a median of 587 and a maximum of 2.8e9. 31.4% of rows repeat another row's feature vector and label, mostly tiny or empty classes.

### B. Paper dataset: `result_total_bon.csv`
* **Shape:** {pinfo['raw_shape'][0]:,} rows x {pinfo['raw_shape'][1]} columns, which matches the paper's 338,442 (off by one). Three `Unnamed: 0*` index columns were dropped.
* **Columns and types:**
  * 26 float metrics. The paper says "25", but the authors' notebook uses these 26, including SumEssential and AvgEssential.
  * Categoricals: `extension` (11 values: .py, .c, .h, .php, .js, .cpp, .java, .cc, .ts, .hpp, .hh) and `Ecosystem` (7).
  * Metadata: `Name`, `commit_date`, `commit_sha`, `file_path`, `Commit_id`, `Source` (before/after) and `key`.
* **Missing values:** none.
* **Target `buggy`:** 305,149 non-buggy (90.16%) and 33,294 buggy (9.84%). This matches the paper exactly.
* **Labelling structure:** `buggy = 1` occurs **only** on `Source = before` rows. The post-fix version of a buggy file is always labelled 0.
* **Duplicates:** **{pinfo['exact_duplicates_removed']:,}** rows are exact duplicates on every non-index column and were removed (paper step 1), leaving {pinfo['raw_shape'][0] - pinfo['exact_duplicates_removed']:,} rows. A further 63.2% of the remaining rows share an identical feature vector and label with another row. In 96% of these duplicate groups, every row has the same `file_path`: the same unchanged file was captured in several commit snapshots. Only 77 rows have all-zero metrics.

---

## 3. Preprocessing replicated, with assumptions

| Step | Paper / notebook | What I did | Assumption? |
|---|---|---|---|
| Duplicate removal | "Removed duplicate files" | Drop rows that are identical on every column (paper data: excluding the 3 index columns; CSV: including `Code`) | **Yes.** The paper says it removed "identical files across commits that remained unchanged", but its reported N (338,442) equals the raw file, so in practice nothing was removed. I removed only exact copies for the main run. The paper's wording, one row per unchanged file (n = 117,742), is tested as a sensitivity run. |
| Missing values | None found; notebook mean/mode-imputes | Same imputers kept inside the pipeline (no effect, since nothing is missing) | – |
| Features | 25 metrics; notebook also uses `extension`, `Ecosystem`, `commit_date` | Paper data: 26 metrics + `extension` + `Ecosystem`. CSV: 14 metrics + `Project` (the analogue of `Ecosystem`). `File`, `Class` and `Code` are dropped as identifiers or raw text. | **Yes.** `commit_date` is excluded because the paper says models predict "solely from quality metrics", and a timestamp would leak project identity under a random split. |
| Encoding | One-hot (notebook) | `OneHotEncoder(handle_unknown="ignore")` | – |
| Scaling | StandardScaler (notebook) | `StandardScaler` on numeric columns | – |
| Feature selection | RQ1 ranks features, but RQ3 trains on all metrics (notebook) | All features used, with no selection | **Yes,** following the notebook |
| Imbalance | SMOTE on training data only | `SMOTE(random_state=42)` inside an `imblearn` Pipeline, so it runs only at `fit` time | – |
| Split | 5-fold time-series CV | **80/20 stratified random split, `random_state=42`** (as the task requires). The paper's time-series CV is repeated as a sensitivity run. | Required by the task |
| Target (CSV only) | – | `smelly` = OR of the 4 smell flags | **Yes,** binary to mirror `buggy` |

The full pipeline saved in each `.pkl` is `ColumnTransformer(one-hot + impute/scale) -> SMOTE -> classifier`. It accepts a raw DataFrame with the feature columns, and SMOTE is skipped at prediction time.

## 4. Models (paper Table 4 hyper-parameters)

| Model | Hyper-parameters |
|---|---|
| XGBoost | defaults (`XGBClassifier(random_state=42)`) |
| Random Forest | `n_estimators=10, max_depth=3, max_features=1, random_state=42` |
| *Extra:* Random Forest, notebook config | `n_estimators=100`, no depth limit, `random_state=42` (paper dataset only; see Observation 2) |

Training and test sizes:

* **Paper dataset:** {ps['train']:,} train ({ps['train_pos']:,} buggy) and {ps['test']:,} test ({ps['test_pos']:,} buggy).
* **Provided CSV:** {ls['train']:,} train ({ls['train_pos']:,} smelly) and {ls['test']:,} test ({ls['test_pos']:,} smelly).

---

## 5. Results

### 5.1 Paper dataset (security bugs). This is the actual replication.
{dataset_block(PAP, "paper_dataset", ["Non-buggy (0)", "Buggy (1)"], [("Random Forest (notebook config)", nb)])}

### 5.2 Provided CSV (code smells)
{dataset_block(LOC, "local_smell_dataset", ["Clean (0)", "Smelly (1)"])}

---

## 6. Comparison with the paper

The paper's Table 15 reports precision, recall and F1 for the **positive (buggy) class**. Accuracy confirms this: at 9.84% prevalence, P = 0.95 and R = 0.82 imply an accuracy of about 0.978, which matches the reported 0.98, whereas weighted averages would all be about 0.98. The "Reproduced" columns therefore show positive-class values. Weighted averages are in Section 5.

{comparison_rows()}

### Sensitivity analyses (to explain the differences)

{sensitivity_rows()}

Per-fold recall with the paper's time-series protocol:
* XGBoost: {tsx['per_fold_recall']}
* Random Forest (Table 4 config): {tsr['per_fold_recall']}

---

## 7. Observations

1. **XGBoost replicates well.** On the authors' data, accuracy ({px['accuracy']:.3f} vs 0.98), F1 ({px['f1']:.3f} vs 0.88) and MCC ({px['mcc']:.3f} vs 0.87) are within about 0.01 of Table 15. My model trades a little precision ({px['precision']:.3f} vs 0.95) for higher recall ({px['recall']:.3f} vs 0.82). With the paper's own time-series protocol, XGBoost gives F1 {tsx['f1']:.3f}, MCC {tsx['mcc']:.3f} and recall {tsx['recall']:.3f}, which is also close. The paper's headline result holds up.

2. **Random Forest does not replicate with Table 4's hyper-parameters, but does with the authors' notebook settings.** With `max_depth=3, n_estimators=10, max_features=1` (Table 4), the forest is too shallow to learn the SMOTE-balanced boundary. It flags {prf['cm'][0][1]:,} of {prf['cm'][0][0] + prf['cm'][0][1]:,} clean files, giving precision {prf['precision']:.3f}, recall {prf['recall']:.3f} and F1 {prf['f1']:.3f} (paper: 0.89 / 0.82 / 0.85). The same shortfall appears under the time-series protocol (F1 {tsr['f1']:.3f}), so the split is not the cause. The RandomForest in the authors' notebook (`n_estimators=100`, no depth limit) gives Acc {nb['accuracy']:.3f}, P {nb['precision']:.3f}, R {nb['recall']:.3f}, F1 {nb['f1']:.3f} and MCC {nb['mcc']:.3f}, which is in line with Table 15. **Most likely the Table 15 RF numbers came from the notebook configuration, and Table 4 is mis-reported.**

3. **The paper's ROC-AUC is most likely computed from hard 0/1 predictions, not probabilities.** My probability-based AUC is {px['roc_auc']:.3f} (XGBoost), far above the paper's 0.91. AUC computed on hard labels equals (TPR + TNR)/2. Applying that to the paper's own Table 15 reproduces every row I checked: XGBoost (0.82 + ~0.995)/2 ≈ 0.91; RF ≈ 0.90; KNN (0.48 + ~0.985)/2 ≈ 0.73; LR (0.03 + ~0.99)/2 ≈ 0.51. My hard-label AUC (XGBoost {px['roc_auc_hard']:.3f}, notebook RF {nb['roc_auc_hard']:.3f}) is in the same range as the paper's values. If this is right, the paper *understates* its models' ranking ability.

4. **Random vs time-ordered split.** The random 80/20 split mixes years, so the test set comes from the same projects and periods as the training set. Under the paper's chronological TimeSeriesSplit, XGBoost recall falls from {tsx['per_fold_recall'][0]:.2f} in the earliest fold to {min(tsx['per_fold_recall']):.2f} in later folds. This is the concept drift the paper mentions, and it is why the random-split numbers are somewhat optimistic.

5. **Duplicate rows inflate accuracy, not the minority-class metrics.** 63% of rows in the paper's data share a feature vector and label with another row. In 96% of those groups, the rows are the same `file_path` repeated: the same unchanged file was re-collected for each security-fixing commit in the repository. The paper says these were removed (Sec. 4.3, step 1), but the published file and the reported N still contain them. When I keep one row per unique (features, label) pair, accuracy falls from {px['accuracy']:.3f} to {ddx['accuracy']:.3f}, because many easy, repeated clean files are gone. F1 ({ddx['f1']:.3f}) and MCC ({ddx['mcc']:.3f}) barely change, so the predictive signal is real. Note also that the post-fix version of every buggy file is labelled 0 while its nearly identical pre-fix version is labelled 1. This label noise is built into the dataset design.

6. **Some of the performance may come from ecosystem and file type rather than security-specific structure.** I did not test this directly. Buggy rates differ a lot by ecosystem (OSS-Fuzz 12.5% vs npm 2.9%), and `extension` is among the paper's top-6 features. Paper data: `commit_date` was excluded, though the authors' notebook includes it. Including it would likely raise random-split scores further through project and time leakage.

7. **Internal inconsistencies in Table 15** (not in my two models): Linear SVM (P 0.35, R 0.30, F1 0.52) and RBF SVM (P 0.29, R 0.36, F1 0.47) report F1 values that are mathematically impossible. F1 is the harmonic mean, so it must lie between P and R (about 0.32 in both cases).

8. **Provided CSV (code smells).** The same pipeline is much weaker at code-smell detection: XGBoost reaches F1 {lx['f1']:.3f}, MCC {lx['mcc']:.3f} and probability ROC-AUC {lx['roc_auc']:.3f}. The paper-style shallow RF reaches high recall ({lrf['recall']:.3f}) but very low precision ({lrf['precision']:.3f}), the same failure mode as on the paper data. There are two likely reasons:
   * Size and complexity metrics suit some smells better than others. XGBoost's recall on the test set is 0.93 for *God class*, 0.61 for *Long method*, 0.53 for *Feature envy* and 0.43 for *Data class*. Feature envy (coupling to other classes) and Data class (accessor-only classes) are not size/complexity properties, and the any-smell target mixes all four.
   * The dataset has no ecosystem or extension signal and is single-language.

   Feature-level dedup changes little (XGBoost F1 {lddx['f1']:.3f}). These numbers are a demonstration of the method on a different task, **not** a replication of the paper's numbers.

---

## 8. Files produced

```
replicate.py                      # full pipeline (both datasets)
make_summary.py                   # builds this report from outputs/results.json
results_summary.md                # this file
outputs/results.json              # every metric, confusion matrix and classification report
outputs/paper_dataset/
    xgboost_model.pkl  random_forest_model.pkl  random_forest_notebook_config_model.pkl
    confusion_matrix_xgboost.png  confusion_matrix_random_forest.png  confusion_matrix_random_forest_notebook_config.png
    classification_report_*.txt   data_inspection.txt
outputs/local_smell_dataset/
    xgboost_model.pkl  random_forest_model.pkl
    confusion_matrix_xgboost.png  confusion_matrix_random_forest.png
    classification_report_*.txt   data_inspection.txt
paper_dataset/                    # authors' dataset + notebook (downloaded from GitHub)
```

To reproduce: `pip install scikit-learn imbalanced-learn xgboost pandas matplotlib joblib`, then run `python replicate.py`, then `python make_summary.py`.

Load a model with `joblib.load("outputs/paper_dataset/xgboost_model.pkl").predict(df[feature_cols])`.

Environment: Python 3.13, scikit-learn, imbalanced-learn, xgboost (see `pip list`).
"""
(ROOT / "results_summary.md").write_text(md, encoding="utf-8")
print("wrote results_summary.md")
