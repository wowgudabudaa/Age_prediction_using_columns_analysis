#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
save_model_subject_rows.py
Reads train/test result CSVs (per model directories defined in MODEL_DIRS).
Learns fold/rep calibration on TRAIN files (if present) and applies to TEST files.
Saves ONE CSV per model containing subject-level rows across all test folds:
  subject_id, true_age, pred_age_raw, pred_age_corr, BAG_raw, cBAG_corr, fold, rep
"""

from pathlib import Path
import argparse
import re
import numpy as np
import pandas as pd

# =================== EDIT MODEL DIRS IF NEEDED ===================
MODEL_DIRS = {
    "FA": "NEW COLUMNS GNC/FA/column_md_results/",
    "MD": "NEW COLUMNS GNC/MD/column_md_results/",
    "Thickness": "NEW COLUMNS GNC/Thickness/thickness_model_results/",
    "QSM": "NEW COLUMNS GNC/QSM/column_md_results/",
    "MD+Thickness": "NEW COLUMNS GNC/MD+thickness/column_md_results/",
    "QSM+MD+Thickness": "NEW COLUMNS GNC/MD+thickness+QSM_dual_channel/column_md_results/",
    "MD+Thickness+QSM+PCs": "NEW COLUMNS GNC/MD+thickness+QSM+PCs/column_md_results/",
}

TRAIN_PATTERN = "**/*train_results*_ref.csv"
TEST_PATTERN  = "**/*test_results*_ref.csv"

SUBJECT_CANDIDATES = [
    "subject_id","subject","sub_id","participant_id","participant",
    "rid","RID","id","ID","ptid","PTID"
]
FOLD_CANDS = ["fold","fold_id","Fold","FoldID"]
REP_CANDS  = ["rep","rep_id","Rep","RepID"]

# =================== helpers: columns & ids ===================
def detect_cols(df):
    cols_l = {c.lower(): c for c in df.columns}
    y_true = cols_l.get("y_true") or cols_l.get("age") or cols_l.get("true_age")
    y_pred = cols_l.get("y_pred") or cols_l.get("yhat") or cols_l.get("y_hat") \
             or cols_l.get("pred") or cols_l.get("pred_age") or cols_l.get("predicted_age")
    if y_true is None or y_pred is None:
        raise ValueError(f"Missing y_true/y_pred. Got columns: {list(df.columns)}")
    y_true = next(c for c in df.columns if c.lower()==y_true)
    y_pred = next(c for c in df.columns if c.lower()==y_pred)
    subj = next((c for c in SUBJECT_CANDIDATES if c in df.columns), None)
    return y_true, y_pred, subj

_fold_rep_re = re.compile(r"fold[_\-]?(\d+).*?rep[_\-]?(\d+)", re.IGNORECASE)
_fold_re     = re.compile(r"fold[_\-]?(\d+)", re.IGNORECASE)
_rep_re      = re.compile(r"rep[_\-]?(\d+)",  re.IGNORECASE)

def parse_fold_rep_from_filename(p: Path):
    s = p.name
    m = _fold_rep_re.search(s)
    if m:
        return int(m.group(1)), int(m.group(2))
    f = _fold_re.search(s)
    r = _rep_re.search(s)
    fold = int(f.group(1)) if f else None
    rep  = int(r.group(1)) if r else None
    return fold, rep

def detect_fold_rep(df, path: Path):
    fold_col = next((c for c in FOLD_CANDS if c in df.columns), None)
    rep_col  = next((c for c in REP_CANDS  if c in df.columns), None)
    if fold_col is not None and rep_col is not None:
        return pd.Series({"fold": df[fold_col].iloc[0], "rep": df[rep_col].iloc[0]})
    fold, rep = parse_fold_rep_from_filename(path)
    return pd.Series({"fold": fold, "rep": rep})

def collect_files(model_dir: Path, pattern: str):
    return sorted([p for p in model_dir.glob(pattern) if p.is_file() and p.suffix.lower()==".csv"])

# =================== calibration ===================
def learn_calibration_params(train_df, method="pred_on_age"):
    yt = train_df["y_true"].values
    yp = train_df["y_pred"].values
    if method == "pred_on_age":
        # fit y_pred = a + b * y_true
        b, a = np.polyfit(yt, yp, deg=1)
        return {"a": float(a), "b": float(b), "mean_age_train": float(np.mean(yt))}
    else:
        # fit BAG = g0 + g1*(y_true - m)
        m = float(np.mean(yt))
        bag = yp - yt
        g1, g0 = np.polyfit(yt - m, bag, deg=1)
        return {"gamma0": float(g0), "gamma1": float(g1), "mean_age_train": m}

def apply_calibration(test_df, params, method="pred_on_age", eps=1e-6):
    df = test_df.copy()
    if method == "pred_on_age":
        a = params["a"]; b = params["b"]
        b_safe = b if abs(b) >= eps else (np.sign(b) * eps if b != 0 else eps)
        df["y_pred_corr"] = (df["y_pred"] - a) / b_safe
    else:
        g0 = params["gamma0"]; g1 = params["gamma1"]; m = params["mean_age_train"]
        bag = df["y_pred"] - df["y_true"]
        bag_corr = bag - (g0 + g1*(df["y_true"] - m))
        df["y_pred_corr"] = df["y_true"] + bag_corr
    return df

# =================== main ===================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="Compare_Model", help="Output directory")
    ap.add_argument("--calib", choices=["pred_on_age","bag_on_age"], default="pred_on_age",
                    help="Fold-wise calibration learned on TRAIN and applied to TEST")
    args = ap.parse_args()

    outdir = Path(args.outdir).expanduser(); outdir.mkdir(parents=True, exist_ok=True)

    any_processed = False

    for model, d in MODEL_DIRS.items():
        mdir = Path(d).expanduser()
        train_files = collect_files(mdir, TRAIN_PATTERN)
        test_files  = collect_files(mdir, TEST_PATTERN)

        if not test_files:
            print(f"[WARN] No *test_results*.csv under {mdir} for {model}; skipping.")
            continue

        # learn calibration per fold/rep (if train files exist)
        calib_map = {}
        for tf in train_files:
            df = pd.read_csv(tf)
            ytc, ypc, _ = detect_cols(df)
            df = df[[ytc, ypc]].rename(columns={ytc:"y_true", ypc:"y_pred"}).copy()
            fr = detect_fold_rep(df, tf)
            key = (int(fr["fold"]) if pd.notna(fr["fold"]) else None,
                   int(fr["rep"])  if pd.notna(fr["rep"])  else None)
            calib_map[key] = learn_calibration_params(df, method=args.calib)

        rows = []
        for f in test_files:
            df = pd.read_csv(f)
            ytc, ypc, subj = detect_cols(df)
            take = df[[ytc, ypc]].rename(columns={ytc:"y_true", ypc:"y_pred"}).copy()
            fr = detect_fold_rep(df, f)
            key = (int(fr["fold"]) if pd.notna(fr["fold"]) else None,
                   int(fr["rep"])  if pd.notna(fr["rep"])  else None)
            take["subject_id"] = (df[subj].astype(str).values if subj is not None
                                  else np.arange(take.shape[0]).astype(str))
            # apply calibration if available for this fold/rep
            if key in calib_map:
                take = apply_calibration(take, calib_map[key], method=args.calib)
            else:
                take["y_pred_corr"] = take["y_pred"]
                print(f"[WARN] No calibration for fold/rep {key} in {model}; keeping raw preds for this file.")
            take["fold"] = key[0]; take["rep"] = key[1]
            # BAG (raw) and cBAG (corrected)
            take["bag_raw"]  = take["y_pred"]      - take["y_true"]
            take["bag_corr"] = take["y_pred_corr"] - take["y_true"]

            rows.append(take)

        # combine all test rows for this model
        all_test = pd.concat(rows, ignore_index=True)

        # build the minimal output table with requested column names
        out_df = pd.DataFrame({
            "subject_id": all_test["subject_id"].astype(str),
            "true_age":   all_test["y_true"],
            "pred_age_raw":  all_test["y_pred"],
            "pred_age_corr": all_test["y_pred_corr"],
            "BAG_raw":    all_test["bag_raw"],
            "cBAG_corr":  all_test["bag_corr"],
            "fold":       all_test["fold"],
            "rep":        all_test["rep"]
        })

        # Save one CSV per model containing all rows across folds/reps
        fname = outdir / f"{model}_subject_cBAG_all_folds.csv"
        out_df.to_csv(fname, index=False)
        print(f"[INFO] Saved {len(out_df)} rows for model '{model}' -> {fname}")
        any_processed = True

        # === New: compute per-subject mean and std across folds/reps ===
        # aggregate requested columns across fold/rep for each subject
        agg_funcs = {
            "true_age": ["mean"],
            "pred_age_raw": ["mean", "std"],
            "pred_age_corr": ["mean", "std"],
            "BAG_raw": ["mean", "std"],
            "cBAG_corr": ["mean", "std"]
        }
        grouped = out_df.groupby("subject_id").agg(agg_funcs)
        # flatten multiindex columns
        grouped.columns = [
            f"{col}_{stat}" if col != "true_age" else "true_age_mean"
            for col, stat in grouped.columns
        ]
        # ensure a sensible order and fill std NaNs (single measurement) with 0
        cols_order = ["true_age_mean",
                      "pred_age_raw_mean", "pred_age_raw_std",
                      "pred_age_corr_mean", "pred_age_corr_std",
                      "BAG_raw_mean", "BAG_raw_std",
                      "cBAG_corr_mean", "cBAG_corr_std"]
        # some columns may be missing if input lacked them; keep available ones
        available_cols = [c for c in cols_order if c in grouped.columns]
        # reset_index will create a subject_id column; keep it and available cols
        agg_df = grouped.reset_index()
        cols_to_keep = ["subject_id"] + available_cols
        cols_to_keep = [c for c in cols_to_keep if c in agg_df.columns]
        agg_df = agg_df[cols_to_keep]
        std_cols = [c for c in agg_df.columns if c.endswith("_std")]
        agg_df[std_cols] = agg_df[std_cols].fillna(0.0)

        # add count of observations per subject (how many fold/rep rows)
        counts = out_df.groupby("subject_id").size().reset_index(name="n_obs")
        agg_df = agg_df.merge(counts, on="subject_id", how="left")

        out_avg_name = outdir / (
            f"{model}_subject_cBAG_mean_std_across_folds.csv"
        )
        agg_df.to_csv(out_avg_name, index=False)
        print(
            "[INFO] Saved per-subject mean/std for model "
            f"'{model}' -> {out_avg_name}"
        )

    if not any_processed:
        raise SystemExit("No models processed. Check directories/patterns.")

    print("\nDone. Per-model subject-row CSVs saved to:", outdir)

 
if __name__ == "__main__":
    main()
