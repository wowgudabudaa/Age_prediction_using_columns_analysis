#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 00:20:47 2025

@author: alex
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grant figure from HABS metadata (AB):
Panels A–F reproducing cBAG/BAG contrasts and HbA1c associations.

Inputs (auto-detected, case-insensitive):
- cBAG_AB, Delta_cBAG_AB, BAG_DeepBrainNet
- CDX_Cog or IMH_Cog or IMH_Cog_Brain  (0=CN, 1=MCI, 2=AD; 9/NA ignored)
- APOE4_positivity
- BW_HBA1c
- Year, Subject (or SubjectRoot), Age, Sex  (for pairing & covariates)

Outputs:
- PNG + PDF multi-panel figure with A–F labels
- Console summary of Ns and model terms

Author: you
"""

import os, datetime, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

# ------------------ CONFIG ------------------
CSV = "/Users/alex/AlexBadea_MyGrants/BAG_R01_100325/metadata/HABS_metadata_AB_enriched_v3.csv"
OUT_DIR = os.path.dirname(CSV)
STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
FIG_PNG = os.path.join(OUT_DIR, f"grant_figure_AB_{STAMP}.png")
FIG_PDF = os.path.join(OUT_DIR, f"grant_figure_AB_{STAMP}.pdf")

# ------------------ UTILS -------------------
def tonum(x): return pd.to_numeric(x, errors="coerce")

def find_col(df, *cands):
    cols = {c.lower(): c for c in df.columns}
    for c in cands:
        if c is None: continue
        key = c.lower()
        if key in cols: return cols[key]
    # try contains (for odd suffixes)
    for want in cands:
        for c in df.columns:
            if want and want.lower() in c.lower():
                return c
    return None

def to_binary_series(s):
    if s is None: return None
    if s.dtype.kind in "biu":
        return s.astype(float)
    t = s.astype(str).str.strip().str.lower()
    yes = {"1","yes","y","true","t","pos","positive","apoe4+","apoe4_pos","apoe4positive"}
    no  = {"0","no","n","false","f","neg","negative","apoe4-","apoe4_neg"}
    out = []
    for v in t:
        if v in yes: out.append(1.0)
        elif v in no: out.append(0.0)
        else:
            try:
                vv = float(v)
                out.append(1.0 if vv>0 else 0.0)
            except: out.append(np.nan)
    return pd.Series(out, index=s.index, dtype=float)

def hedges_g(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a)<2 or len(b)<2: return np.nan
    na, nb = len(a), len(b)
    sa2, sb2 = a.var(ddof=1), b.var(ddof=1)
    sp = np.sqrt(((na-1)*sa2 + (nb-1)*sb2) / (na+nb-2))
    d = (b.mean() - a.mean()) / sp
    J = 1 - 3/(4*(na+nb)-9)
    return float(d*J)

def welch_t(a,b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a)<2 or len(b)<2: return np.nan
    return stats.ttest_ind(a, b, equal_var=False, nan_policy="omit").pvalue

def add_const(X): return sm.add_constant(X, has_constant="add")
def ci95(fit, name):
    L, U = fit.conf_int().loc[name]
    return float(L), float(U)

# ------------------ LOAD --------------------
df = pd.read_csv(CSV)
print(f"Loaded {df.shape[0]} rows, {df.shape[1]} cols")

# IDs, time, basics
ID  = find_col(df, "Subject", "SubjectRoot")
YEAR= find_col(df, "Year")
AGE = find_col(df, "Age")
SEX = find_col(df, "Sex")
if SEX:
    # map sex to 0/1 if strings
    sx = df[SEX].astype(str).str.strip().str.lower()
    msk = sx.isin(["m","male","1","true","t"])
    df[SEX] = np.where(msk, 1.0, np.where(sx.isin(["f","female","0","false","f"]), 0.0, tonum(df[SEX])))

for col in [YEAR, AGE]:
    if col: df[col] = tonum(df[col])

# Key measures
CBAG   = find_col(df, "cBAG_AB")
DBNET  = find_col(df, "BAG_DeepBrainNet")
DELTA  = find_col(df, "Delta_cBAG_AB")
HBA1C  = find_col(df, "BW_HBA1c")
APOE   = find_col(df, "APOE4_Positivity", "APOE4_positivity")

COG    = find_col(df, "CDX_Cog", "IMH_Cog", "IMH_Cog_Brain")
if COG:
    cog_num = tonum(df[COG])
    cog_num = cog_num.replace(9, np.nan)  # unknown to NaN
    any_imp = np.where(cog_num.isin([1,2]), 1.0, np.where(cog_num==0, 0.0, np.nan))
    df["AnyImpair"] = any_imp
else:
    df["AnyImpair"] = np.nan

if APOE:
    df["APOE4"] = to_binary_series(df[APOE])
else:
    df["APOE4"] = np.nan

# ------------------ PAIRS FOR Δ/yr -----------
pair_ok = (ID is not None) and (YEAR is not None) and (CBAG is not None) and (HBA1C is not None)
pairs = None
if pair_ok:
    sub = df[df[YEAR].isin([0,2])].copy()
    cbag_w = sub.pivot(index=ID, columns=YEAR, values=CBAG).dropna(subset=[0,2])
    hba_w  = sub.pivot(index=ID, columns=YEAR, values=HBA1C).dropna(subset=[0,2])
    ids = cbag_w.index.intersection(hba_w.index)
    pairs = pd.DataFrame(index=ids)
    pairs["cBAG_Y0"] = cbag_w.loc[ids, 0]; pairs["cBAG_Y2"] = cbag_w.loc[ids, 2]
    pairs["HBA1c_Y0"]= hba_w.loc[ids, 0];  pairs["HBA1c_Y2"]= hba_w.loc[ids, 2]
    # carry baseline covariates
    if AGE:
        pairs["Age_Y0"] = sub[sub[YEAR]==0].set_index(ID)[AGE].reindex(ids)
    if SEX:
        pairs["Sex"]    = sub[sub[YEAR]==0].set_index(ID)[SEX].reindex(ids)
    if APOE:
        pairs["APOE4"]  = sub[sub[YEAR]==0].set_index(ID)["APOE4"].reindex(ids)
    # fixed 2-y denom to match earlier figures
    pairs["Delta_cBAG_per_yr"]  = (pairs["cBAG_Y2"]  - pairs["cBAG_Y0"]) / 2.0
    pairs["Delta_HbA1c_per_yr"] = (pairs["HBA1c_Y2"] - pairs["HBA1c_Y0"]) / 2.0
    print(f"Pairs for Y0–Y2: {len(pairs)}")

# ------------------ FIGURE LAYOUT -----------
plt.rcParams.update({"font.size": 10})
fig = plt.figure(figsize=(9.0, 9.8))
gs  = fig.add_gridspec(3, 2, hspace=0.28, wspace=0.20)

def panel_label(ax, lbl):
    ax.text(-0.08, 1.08, lbl, transform=ax.transAxes, fontsize=16, fontweight="bold", va="bottom", ha="right")

# ---------- Panel A: cBAG by cognition ----------
axA = fig.add_subplot(gs[0,0])
if CBAG and "AnyImpair" in df.columns and df["AnyImpair"].notna().any():
    dd = df[[CBAG, "AnyImpair"]].dropna()
    g0 = dd.loc[dd["AnyImpair"]==0, CBAG].values
    g1 = dd.loc[dd["AnyImpair"]==1, CBAG].values
    vp = axA.violinplot([g0, g1], showmeans=False, showmedians=True)
    axA.boxplot([g0, g1], positions=[1,2], widths=0.15, manage_ticks=False)
    axA.scatter(np.ones_like(g0)*1, g0, s=12, alpha=0.45)
    axA.scatter(np.ones_like(g1)*2, g1, s=12, alpha=0.45)
    axA.set_xticks([1,2]); axA.set_xticklabels(["CN", "MCI/AD"])
    axA.set_ylabel("cBAG_AB (years)")
    n0, n1 = len(g0), len(g1)
    dmu = float(np.nanmean(g1) - np.nanmean(g0))
    g = hedges_g(g0, g1); p = welch_t(g0, g1)
    axA.set_title(f"cBAG by cognition  Δμ={dmu:.2f} y, g={g:.2f}, p={p:.3g}")
    axA.text(0.5, -0.22, f"n={n0}", transform=axA.transAxes, ha="center")
    axA.text(0.85, -0.22, f"n={n1}", transform=axA.transAxes, ha="center")
else:
    axA.text(0.5,0.5,"No cognition/cBAG data",ha="center",va="center"); axA.set_axis_off()
panel_label(axA, "A")

# ---------- Panel B: cBAG by APOE4 ----------
axB = fig.add_subplot(gs[0,1])
if CBAG and APOE:
    dd = df[[CBAG, "APOE4"]].dropna()
    g0 = dd.loc[dd["APOE4"]==0, CBAG].values
    g1 = dd.loc[dd["APOE4"]==1, CBAG].values
    if len(g0)>=2 and len(g1)>=2:
        axB.violinplot([g0, g1], showmedians=True)
        axB.boxplot([g0, g1], positions=[1,2], widths=0.15, manage_ticks=False)
        axB.scatter(np.ones_like(g0)*1, g0, s=12, alpha=0.45)
        axB.scatter(np.ones_like(g1)*2, g1, s=12, alpha=0.45)
        axB.set_xticks([1,2]); axB.set_xticklabels(["APOE4−","APOE4+"])
        axB.set_ylabel("cBAG_AB (years)")
        dmu = float(np.nanmean(g1) - np.nanmean(g0))
        g = hedges_g(g0,g1); p = welch_t(g0,g1)
        axB.set_title(f"cBAG by APOE4  Δμ={dmu:.2f} y, g={g:.2f}, p={p:.3g}")
        axB.text(0.5, -0.22, f"n={len(g0)}", transform=axB.transAxes, ha="center")
        axB.text(0.85, -0.22, f"n={len(g1)}", transform=axB.transAxes, ha="center")
    else:
        axB.text(0.5,0.5,"APOE4 groups too small",ha="center",va="center"); axB.set_axis_off()
else:
    axB.text(0.5,0.5,"No APOE4/cBAG data",ha="center",va="center"); axB.set_axis_off()
panel_label(axB, "B")

# ---------- Panel C: BAG_DeepBrainNet by APOE4×Cognition ----------
axC = fig.add_subplot(gs[1,0])
if DBNET and "AnyImpair" in df.columns and APOE:
    temp = df[[DBNET,"AnyImpair","APOE4"]].dropna()
    if not temp.empty:
        groups = []
        labels = []
        for ap in [0.0,1.0]:
            for imp in [0.0,1.0]:
                vals = temp.loc[(temp["APOE4"]==ap) & (temp["AnyImpair"]==imp), DBNET].values
                groups.append(vals); labels.append(f"{'APOE4+' if ap==1 else 'APOE4−'}/{ 'MCI/AD' if imp==1 else 'CN'} (n={len(vals)})")
        pos = np.arange(1, len(groups)+1)
        axC.violinplot(groups, positions=pos, showmedians=True)
        for i,gvals in enumerate(groups, start=1):
            axC.boxplot(gvals, positions=[i], widths=0.15, manage_ticks=False)
            axC.scatter(np.ones_like(gvals)*i, gvals, s=10, alpha=0.35)
        axC.set_xticks(pos); axC.set_xticklabels(labels, rotation=15, ha="right")
        axC.set_ylabel("BAG_DeepBrainNet (years)")
        # simple ANOVA interaction p (descriptive)
        try:
            from statsmodels.formula.api import ols
            m = ols(f"{DBNET} ~ C(AnyImpair)*C(APOE4)", data=temp).fit()
            from statsmodels.stats.anova import anova_lm
            an = anova_lm(m, typ=2)
            p_int = float(an.loc["C(AnyImpair):C(APOE4)","PR(>F)"])
            axC.set_title(f"APOE4 × Cognition interaction p={p_int:.3g}")
        except Exception:
            axC.set_title("APOE4 × Cognition")
    else:
        axC.text(0.5,0.5,"No overlapping data",ha="center",va="center"); axC.set_axis_off()
else:
    axC.text(0.5,0.5,"Missing BAG_DeepBrainNet/APOE4/Cog",ha="center",va="center"); axC.set_axis_off()
panel_label(axC, "C")

# ---------- Panel D: ΔcBAG by cognition ----------
axD = fig.add_subplot(gs[1,1])
if DELTA and "AnyImpair" in df.columns:
    dd = df[[DELTA,"AnyImpair"]].dropna()
    g0 = dd.loc[dd["AnyImpair"]==0, DELTA].values
    g1 = dd.loc[dd["AnyImpair"]==1, DELTA].values
    if len(g0)>=2 and len(g1)>=2:
        axD.violinplot([g0,g1], showmedians=True)
        axD.boxplot([g0,g1], positions=[1,2], widths=0.15, manage_ticks=False)
        axD.scatter(np.ones_like(g0)*1, g0, s=12, alpha=0.45)
        axD.scatter(np.ones_like(g1)*2, g1, s=12, alpha=0.45)
        axD.set_xticks([1,2]); axD.set_xticklabels(["CN","MCI/AD"])
        axD.set_ylabel("ΔcBAG_AB (years over 2y)")
        dmu = float(np.nanmean(g1) - np.nanmean(g0))
        g = hedges_g(g0,g1); p = welch_t(g0,g1)
        axD.set_title(f"ΔcBAG by cognition  Δμ={dmu:.2f} y, g={g:.2f}, p={p:.3g}")
        axD.text(0.5, -0.22, f"n={len(g0)}", transform=axD.transAxes, ha="center")
        axD.text(0.85, -0.22, f"n={len(g1)}", transform=axD.transAxes, ha="center")
    else:
        axD.text(0.5,0.5,"ΔcBAG groups too small",ha="center",va="center"); axD.set_axis_off()
else:
    axD.text(0.5,0.5,"No ΔcBAG/cognition data",ha="center",va="center"); axD.set_axis_off()
panel_label(axD, "D")

# ---------- Panel E: Adjusted partial (Δ per year) ----------
axE = fig.add_subplot(gs[2,0])
if pairs is not None:
    need = ["Delta_cBAG_per_yr","Delta_HbA1c_per_yr","cBAG_Y0","HBA1c_Y0"]
    covs = ["cBAG_Y0","HBA1c_Y0"]
    if "Age_Y0" in pairs.columns: need.append("Age_Y0"); covs.append("Age_Y0")
    if "Sex"    in pairs.columns: need.append("Sex");    covs.append("Sex")
    if "APOE4"  in pairs.columns: need.append("APOE4");  covs.append("APOE4")
    adj = pairs[need].dropna()
    if len(adj) >= 8:
        Xa = add_const(adj[["Delta_HbA1c_per_yr"] + covs])
        ya = adj["Delta_cBAG_per_yr"]
        fit = sm.OLS(ya, Xa).fit(cov_type="HC3")
        b = float(fit.params["Delta_HbA1c_per_yr"]); L,U = ci95(fit,"Delta_HbA1c_per_yr")
        p = float(fit.pvalues["Delta_HbA1c_per_yr"]); R2 = float(fit.rsquared)

        # partial residuals
        Z = add_const(adj[covs])
        y_res = sm.OLS(ya, Z).fit().resid
        x_res = sm.OLS(adj["Delta_HbA1c_per_yr"], Z).fit().resid
        m = np.polyfit(x_res, y_res, 1); xs = np.linspace(x_res.min(), x_res.max(), 200); ys = m[0]*xs + m[1]
        axE.scatter(x_res, y_res, s=18, alpha=0.7, edgecolors="k")
        axE.plot(xs, ys, linewidth=2)
        axE.set_xlabel("ΔHbA1c per year (residualized)")
        axE.set_ylabel("ΔcBAG_AB per year (residualized)")
        axE.set_title(f"Adjusted partial (Y0–Y2): β={b:.3f} [{L:.3f},{U:.3f}] • p={p:.4g} • N={len(adj)} • R²={R2:.3f}")
    else:
        axE.text(0.5,0.5,"Too few paired subjects for adjusted plot",ha="center",va="center"); axE.set_axis_off()
else:
    axE.text(0.5,0.5,"No Y0–Y2 pairs available",ha="center",va="center"); axE.set_axis_off()
panel_label(axE, "E")

# ---------- Panel F: Baseline scatter cBAG vs HbA1c ----------
axF = fig.add_subplot(gs[2,1])
if CBAG and HBA1C:
    base = df.copy()
    if YEAR: base = base[base[YEAR]==0]
    base = base[[CBAG, HBA1C]].dropna()
    if len(base) >= 8:
        x = tonum(base[HBA1C]); y = tonum(base[CBAG])
        X = add_const(x)
        fit = sm.OLS(y, X).fit(cov_type="HC3")
        b = float(fit.params[HBA1C]); L,U = ci95(fit, HBA1C)
        p = float(fit.pvalues[HBA1C]); R2 = float(fit.rsquared)
        xs = np.linspace(x.min(), x.max(), 200); ys = float(fit.params["const"]) + b*xs
        axF.scatter(x, y, s=18, alpha=0.7, edgecolors="k")
        axF.plot(xs, ys, linewidth=2)
        axF.set_xlabel("HbA1c (%)"); axF.set_ylabel("cBAG_AB (years)")
        axF.set_title(f"Baseline: β={b:.3f} [{L:.3f},{U:.3f}] • R²={R2:.3f} • p={p:.4g} • N={len(base)}")
    else:
        axF.text(0.5,0.5,"Too few baseline cBAG/HbA1c",ha="center",va="center"); axF.set_axis_off()
else:
    axF.text(0.5,0.5,"Missing cBAG/HbA1c",ha="center",va="center"); axF.set_axis_off()
panel_label(axF, "F")

# ------------------ SAVE ---------------------
for ax in fig.axes:
    ax.grid(alpha=0.25)
fig.suptitle("Brain-age contrasts and glycemic associations (HABS-HD)", y=1.01, fontsize=13)
plt.tight_layout()
fig.savefig(FIG_PNG, dpi=500, bbox_inches="tight")
fig.savefig(FIG_PDF, bbox_inches="tight")
print(f"Saved figure → {FIG_PNG}\nSaved figure → {FIG_PDF}")
