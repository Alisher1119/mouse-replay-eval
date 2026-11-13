"""95% confidence intervals over seeds and exact paired sign-flip tests.

E1 vs E2:   is replay recall lower when replay is unseen?   (paired by seed, one-sided)
model vs RF: does a model differ from Random Forest in E2?  (paired by seed, two-sided)
With 5 seeds the smallest attainable one-sided p is 1/32 = 0.031.
"""
import sys, itertools
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

R = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "results")

def ci(x, clip=True):
    x = np.asarray(x, float); m = x.mean()
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
    return (m, max(m - h, 0), min(m + h, 1)) if clip else (m, m - h, m + h)

def signflip(d, alternative="greater"):
    """Exact sign-flip permutation test on paired differences d (H0: symmetric around 0)."""
    d = np.asarray(d, float); obs = d.mean()
    null = np.array([(d * s).mean() for s in itertools.product([1, -1], repeat=len(d))])
    if alternative == "greater": return float((null >= obs - 1e-12).mean())
    return float((np.abs(null) >= abs(obs) - 1e-12).mean())

def fmt(t): return f"{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]"

rows = []
v5 = pd.read_csv(R / "captchasolve30k_v5_raw.csv")
for m in ["Logistic regression (stat)", "Random Forest (stat)", "LightGBM (stat)", "GS: concat", "GS-Fusion (gated)"]:
    e1 = v5[(v5.model == m) & v5.experiment.str.startswith("E1")].set_index("seed").human_replay
    e2 = v5[(v5.model == m) & v5.experiment.str.startswith("E2")].set_index("seed").human_replay
    rows.append(dict(dataset="CaptchaSolve30k", model=m, E1=fmt(ci(e1)), E2=fmt(ci(e2)),
                     drop=fmt(ci(e1 - e2, clip=False)), p_drop=signflip((e1 - e2).values)))
rf2 = v5[(v5.model == "Random Forest (stat)") & v5.experiment.str.startswith("E2")].set_index("seed").human_replay
cx = pd.read_csv(R / "cross_dataset_raw.csv")
for m in ["Logistic regression", "Random Forest", "LightGBM", "GMM (human-only)"]:
    e1 = cx[(cx.exp == "X2-E1") & (cx.model == m)].set_index("seed").replay
    e2 = cx[(cx.exp == "X2-E2") & (cx.model == m)].set_index("seed").replay
    rows.append(dict(dataset="SapiMouse", model=m, E1=fmt(ci(e1)), E2=fmt(ci(e2)),
                     drop=fmt(ci(e1 - e2, clip=False)), p_drop=signflip((e1 - e2).values)))
print("Replay recall at 5% FPR: mean [95% CI over seeds]; p = exact one-sided sign-flip test of E1 > E2\n")
print(pd.DataFrame(rows).to_string(index=False))

dp = R / "deep_baseline_raw.csv"
if dp.exists():
    v5 = pd.concat([v5, pd.read_csv(dp)], ignore_index=True)
    for m in pd.read_csv(dp).model.unique():
        e1 = v5[(v5.model == m) & v5.experiment.str.startswith("E1")].set_index("seed").human_replay
        e2 = v5[(v5.model == m) & v5.experiment.str.startswith("E2")].set_index("seed").human_replay
        print(f"  {m:28s} E1 {fmt(ci(e1))}  E2 {fmt(ci(e2))}  drop p = {signflip((e1 - e2).values):.3f}")

print("\nCaptchaSolve30k E2: model minus Random Forest (two-sided sign-flip test)")
for m in ["LightGBM (stat)", "GS: concat", "GS-Fusion (gated)", "Logistic regression (stat)"] + \
         ([m for m in pd.read_csv(dp).model.unique()] if dp.exists() else []):
    e2 = v5[(v5.model == m) & v5.experiment.str.startswith("E2")].set_index("seed").human_replay
    d = (e2 - rf2).values
    print(f"  {m:28s} diff {fmt(ci(d, clip=False)).ljust(24)} worse in {int((d<0).sum())}/5 seeds  p = {signflip(d,'two'):.3f}")
