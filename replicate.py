"""
Replication of Diouf et al. (2026), "Predicting Software Security Bugs Using Machine
Learning and Quality Metrics: An Empirical Study", CMC 87(3) - RQ3 (ML prediction).

Two models from the paper (Table 4 hyper-parameters):
  * XGBoost       - default parameters (paper's best model)
  * Random Forest - n_estimators=10, max_depth=3, max_features=1

Preprocessing (paper Sec. 3.3 / 4.3 + authors' published notebook ML.ipynb):
  * remove exact duplicate records
  * categorical -> most-frequent impute + One-Hot encode
  * numeric     -> mean impute + StandardScaler
  * SMOTE(random_state=42) applied to the TRAINING data only
Split: 80/20, stratified, random_state=42 (as requested; the paper used time-series CV).

Runs on two datasets:
  A. multi-smell-dataset-v1_2.csv  (the file provided in this folder)
  B. paper_dataset/result_total_bon.csv  (authors' public dataset, GitHub PredictSecBugs)
"""
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from matplotlib.colors import LinearSegmentedColormap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             matthews_corrcoef, precision_recall_fscore_support,
                             roc_auc_score)
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
SEED = 42

SMELLS = ["Long method", "God class", "Feature envy", "Data class"]
PAPER_METRICS = [
    "AvgLineCode", "CountDeclClass", "RatioCommentToCode", "CountStmtExe", "AvgCyclomaticStrict",
    "CountLine", "SumCyclomatic", "AvgCyclomatic", "SumEssential", "MaxCyclomatic",
    "AvgLineComment", "AvgCyclomaticModified", "AvgEssential", "SumCyclomaticModified",
    "CountLineComment", "CountLineCode", "MaxCyclomaticModified", "CountLineBlank",
    "CountStmtDecl", "AvgLine", "MaxEssential", "CountDeclFunction", "MaxNesting",
    "AvgLineBlank", "SumCyclomaticStrict", "CountStmt",
]

# Paper Table 15 (mean over 5 time-series CV folds; P/R/F1 are for the buggy class)
PAPER_RESULTS = {
    "XGBoost": dict(accuracy=0.98, precision=0.95, recall=0.82, f1=0.88, mcc=0.87, roc_auc=0.91),
    "Random Forest": dict(accuracy=0.97, precision=0.89, recall=0.82, f1=0.85, mcc=0.84, roc_auc=0.90),
}

# Chart tokens (sequential blue ramp, light chart surface)
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def log(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------- data
def load_local_smell():
    path = ROOT / "multi-smell-dataset-v1_2.csv"
    df = pd.read_csv(path)
    info = {"raw_shape": df.shape}
    # Paper step 1: remove duplicate records (exact copies across ALL columns, incl. source code)
    n0 = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    info["exact_duplicates_removed"] = n0 - len(df)
    df = df.drop(columns=["Code"])
    num = [c for c in df.columns[2:16]]  # Halstead + cyclomatic metrics
    df["smelly"] = df[SMELLS].max(axis=1)
    return dict(
        key="local_smell_dataset", title="Multi-smell dataset (provided CSV)",
        df=df, num=num, cat=["Project"], target="smelly",
        class_names=["Clean (0)", "Smelly (1)"], info=info, date_col=None,
        extra_cols=SMELLS + ["File", "Class"],
    )


def load_paper():
    path = ROOT / "paper_dataset" / "result_total_bon.csv"
    df = pd.read_csv(path, low_memory=False)
    info = {"raw_shape": df.shape}
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    n0 = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    info["exact_duplicates_removed"] = n0 - len(df)
    df["buggy"] = df["buggy"].astype(int)
    return dict(
        key="paper_dataset", title="Paper dataset (Diouf et al., PredictSecBugs)",
        df=df, num=PAPER_METRICS, cat=["extension", "Ecosystem"], target="buggy",
        class_names=["Non-buggy (0)", "Buggy (1)"], info=info, date_col="commit_date",
        extra_cols=["Source"],
    )


def inspect(ds, fh):
    df, t = ds["df"], ds["target"]
    w = lambda s="": fh.write(str(s) + "\n")
    w(f"# Data inspection - {ds['title']}")
    w(f"raw shape: {ds['info']['raw_shape']}")
    w(f"exact duplicate rows removed: {ds['info']['exact_duplicates_removed']}")
    w(f"shape after dedup: {df.shape}\n")
    w("## dtypes"); w(df.dtypes.to_string()); w()
    w("## missing values per column"); w(df.isna().sum().to_string()); w()
    w(f"## target '{t}' distribution"); vc = df[t].value_counts().sort_index()
    w(pd.DataFrame({"count": vc, "pct": (vc / len(df) * 100).round(2)}).to_string()); w()
    for c in ds["extra_cols"]:
        if df[c].nunique() < 50:
            w(f"## {c} x {t}"); w(pd.crosstab(df[c], df[t]).to_string()); w()
    for c in ds["cat"]:
        w(f"## categorical '{c}' ({df[c].nunique()} levels) - top 15")
        w(df[c].value_counts().head(15).to_string()); w()
    w("## numeric feature summary"); w(df[ds["num"]].describe().T.to_string()); w()
    d = df.duplicated(ds["num"] + ds["cat"] + [t]).sum()
    w(f"rows whose feature vector + label duplicates another row: {d} ({d / len(df):.1%})")


# --------------------------------------------------------------------------- models
def make_models():
    return {
        "XGBoost": XGBClassifier(random_state=SEED, n_jobs=-1),
        "Random Forest": RandomForestClassifier(n_estimators=10, max_depth=3, max_features=1,
                                                random_state=SEED, n_jobs=-1),
    }


def make_pipeline(num, cat, model):
    prep = ColumnTransformer([
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
        ("num", Pipeline([("imputer", SimpleImputer(strategy="mean")),
                          ("scaler", StandardScaler())]), num),
    ])
    # imblearn Pipeline: SMOTE runs only during fit (training data), never at predict time
    return ImbPipeline([("prep", prep), ("smote", SMOTE(random_state=SEED)), ("clf", model)])


def evaluate(y_true, y_pred, y_prob, class_names):
    rep_txt = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    rep = classification_report(y_true, y_pred, target_names=class_names, digits=4, output_dict=True)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=1)
    return dict(
        accuracy=accuracy_score(y_true, y_pred),
        precision=p, recall=r, f1=f,  # positive (minority) class - what the paper reports
        mcc=matthews_corrcoef(y_true, y_pred),
        roc_auc=roc_auc_score(y_true, y_prob),
        # AUC computed from hard 0/1 predictions (= balanced accuracy); the paper's ROC-AUC
        # values are consistent with this variant, so we report it for a like-for-like comparison
        roc_auc_hard=roc_auc_score(y_true, y_pred),
        report=rep, report_text=rep_txt,
        cm=confusion_matrix(y_true, y_pred).tolist(),
    )


def plot_cm(cm, class_names, title, subtitle, path):
    cm = np.asarray(cm)
    row_pct = cm / cm.sum(axis=1, keepdims=True)
    cmap = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)
    fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=150)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    im = ax.imshow(row_pct, cmap=cmap, vmin=0, vmax=1)
    # 2px surface gaps between cells
    ax.set_xticks(np.arange(-.5, 2, 1), minor=True); ax.set_yticks(np.arange(-.5, 2, 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=4); ax.tick_params(which="minor", length=0)
    for i in range(2):
        for j in range(2):
            color = "#ffffff" if row_pct[i, j] > 0.5 else INK
            ax.text(j, i - 0.07, f"{cm[i, j]:,}", ha="center", va="center", fontsize=15,
                    fontweight="bold", color=color)
            ax.text(j, i + 0.13, f"{row_pct[i, j]:.1%} of row", ha="center", va="center",
                    fontsize=9.5, color=color)
    ax.set_xticks([0, 1], class_names, color=INK2); ax.set_yticks([0, 1], class_names, color=INK2)
    ax.set_xlabel("Predicted", color=INK2, labelpad=8); ax.set_ylabel("Actual", color=INK2, labelpad=8)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.outline.set_visible(False); cb.ax.tick_params(colors=MUTED, length=0)
    cb.set_label("Share of actual class (row %)", color=MUTED)
    cb.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    fig.suptitle(title, x=0.03, ha="left", fontsize=13, fontweight="bold", color=INK)
    ax.set_title(subtitle.replace(" | ", "\n", 1), loc="left", fontsize=9, color=INK2, pad=10)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def strip(m):
    return {k: v for k, v in m.items() if k not in ("report", "report_text")}


# --------------------------------------------------------------------------- runs
def run_dataset(ds):
    out = OUT / ds["key"]; out.mkdir(parents=True, exist_ok=True)
    with open(out / "data_inspection.txt", "w", encoding="utf-8") as fh:
        inspect(ds, fh)
    log((out / "data_inspection.txt").read_text(encoding="utf-8"))

    df, num, cat, t = ds["df"], ds["num"], ds["cat"], ds["target"]
    X, y = df[num + cat], df[t].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    log(f"train {X_tr.shape} (pos={y_tr.sum()})  test {X_te.shape} (pos={y_te.sum()})")
    res = {"split": dict(train=len(y_tr), test=len(y_te), train_pos=int(y_tr.sum()), test_pos=int(y_te.sum())),
           "main": {}, "sensitivity": {}}

    for name, model in make_models().items():
        t0 = time.time()
        pipe = make_pipeline(num, cat, model).fit(X_tr, y_tr)
        m = evaluate(y_te, pipe.predict(X_te), pipe.predict_proba(X_te)[:, 1], ds["class_names"])
        m["fit_seconds"] = round(time.time() - t0, 1)
        slug = name.lower().replace(" ", "_")
        joblib.dump(pipe, out / f"{slug}_model.pkl")
        plot_cm(m["cm"], ds["class_names"], f"{name} - confusion matrix",
                f"{ds['title']} | 80/20 stratified test set (n={len(y_te):,}) | SMOTE on train only",
                out / f"confusion_matrix_{slug}.png")
        (out / f"classification_report_{slug}.txt").write_text(m["report_text"], encoding="utf-8")
        log(f"\n===== {ds['title']} :: {name} =====")
        log(f"Accuracy {m['accuracy']:.4f} | MCC {m['mcc']:.4f} | ROC-AUC {m['roc_auc']:.4f}")
        log(m["report_text"]); log("Confusion matrix:", m["cm"])
        res["main"][name] = m

    # ---- sensitivity analyses (metrics only; explain gaps vs. the paper)
    # S1: drop rows whose feature vector + label duplicates another row
    dd = df.drop_duplicates(num + cat + [t])
    Xd, yd = dd[num + cat], dd[t].values
    a, b, c, d = train_test_split(Xd, yd, test_size=0.2, random_state=SEED, stratify=yd)
    for name, model in make_models().items():
        p = make_pipeline(num, cat, model).fit(a, c)
        res["sensitivity"][f"{name} | feature-level dedup (n={len(dd):,})"] = strip(
            evaluate(d, p.predict(b), p.predict_proba(b)[:, 1], ds["class_names"]))
        log("S1 done", name)

    if ds["date_col"]:
        # S2: the paper's own protocol - chronological 5-fold TimeSeriesSplit, SMOTE in train folds
        ds_sorted = df.assign(_d=pd.to_datetime(df[ds["date_col"]])).sort_values("_d", kind="stable")
        Xs, ys = ds_sorted[num + cat], ds_sorted[t].values
        for name in make_models():
            folds = []
            for tr, te in TimeSeriesSplit(n_splits=5).split(Xs):
                p = make_pipeline(num, cat, make_models()[name]).fit(Xs.iloc[tr], ys[tr])
                folds.append(strip(evaluate(ys[te], p.predict(Xs.iloc[te]),
                                            p.predict_proba(Xs.iloc[te])[:, 1], ds["class_names"])))
            keys = ["accuracy", "precision", "recall", "f1", "mcc", "roc_auc"]
            res["sensitivity"][f"{name} | paper protocol: TimeSeriesSplit(5) mean"] = {
                k: float(np.mean([f[k] for f in folds])) for k in keys}
            res["sensitivity"][f"{name} | paper protocol: TimeSeriesSplit(5) mean"]["per_fold_recall"] = [
                round(f["recall"], 4) for f in folds]
            log("S2 done", name)

        # S3: Random Forest as configured in the authors' notebook (100 trees, unlimited depth)
        p = make_pipeline(num, cat, RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
                          ).fit(X_tr, y_tr)
        m = evaluate(y_te, p.predict(X_te), p.predict_proba(X_te)[:, 1], ds["class_names"])
        joblib.dump(p, out / "random_forest_notebook_config_model.pkl")
        plot_cm(m["cm"], ds["class_names"], "Random Forest (notebook config) - confusion matrix",
                f"{ds['title']} | 100 trees, no depth limit | 80/20 stratified test set (n={len(y_te):,})",
                out / "confusion_matrix_random_forest_notebook_config.png")
        (out / "classification_report_random_forest_notebook_config.txt").write_text(
            m["report_text"], encoding="utf-8")
        log(m["report_text"])
        m.pop("report_text")  # keep the per-class report dict for the summary
        res["sensitivity"]["Random Forest | notebook config (100 trees, no depth limit), 80/20"] = m
        log("S3 done")

    for m in res["main"].values():
        m.pop("report_text")
    return res


if __name__ == "__main__":
    which = sys.argv[1:] or ["local", "paper"]
    loaders = {"local": load_local_smell, "paper": load_paper}
    all_res = {}
    rf = OUT / "results.json"
    if rf.exists():
        all_res = json.loads(rf.read_text())
    OUT.mkdir(exist_ok=True)
    for w in which:
        ds = loaders[w]()
        all_res[ds["key"]] = {"title": ds["title"], "info": {k: (list(v) if isinstance(v, tuple) else v)
                                                              for k, v in ds["info"].items()},
                              **run_dataset(ds)}
        rf.write_text(json.dumps(all_res, indent=2, default=float))
    log("\nDONE")
