"""Rebuild the numbers of Tables 2-5 of the paper from the raw per-seed CSV files in results/.

Usage:  python scripts/summarize_results.py [results_dir]
"""
import sys
from pathlib import Path
import pandas as pd

R = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "results")
SYN = ["linear", "linear_jitter", "bezier_fitts", "bezier_tremor"]
REP = ["human_replay", "human_replay_raw"]
pd.set_option("display.width", 160)


def table2():
    d = pd.read_csv(R / "captchasolve30k_v5_raw.csv")
    models = {"Logistic regression (stat)": "Logistic regression", "Random Forest (stat)": "Random Forest",
              "LightGBM (stat)": "LightGBM", "Random Forest (graph only)": "Random Forest, graph only",
              "GS: concat": "Concatenation (3 views)", "Any view flags (stat | seq | graph)": "Any view flags (3 views)",
              "GS-Fusion (gated)": "GS-Fusion (gated)"}
    rows = []
    for m, name in models.items():
        e1 = d[(d.model == m) & d.experiment.str.startswith("E1")]
        e2 = d[(d.model == m) & d.experiment.str.startswith("E2")]
        rows.append({"model": name,
                     "E1 replay / raw": f"{e1.human_replay.mean():.2f} / {e1.human_replay_raw.mean():.2f}",
                     "E2 replay / raw": f"{e2.human_replay.mean():.2f} ± {e2.human_replay.std():.2f} / "
                                        f"{e2.human_replay_raw.mean():.2f} ± {e2.human_replay_raw.std():.2f}",
                     "E2 L1-L4 min": round(e2[SYN].mean().min(), 2), "E2 FPR": round(e2.FPR.mean(), 3)})
    print("\nTable 2 - CaptchaSolve30k, recall at FPR <= 5% (mean over seeds)\n", pd.DataFrame(rows).to_string(index=False))
    g = pd.read_csv(R / "captchasolve30k_v5_gates.csv")
    print("\nGate weights (stat / seq / graph), mean over seeds\n",
          g.groupby(["experiment", "profile"])[["g_stat", "g_seq", "g_graph"]].mean().round(2).to_string())


def table3():
    v5 = pd.read_csv(R / "captchasolve30k_v5_raw.csv")
    v5 = v5[v5.experiment.str.startswith("E2")].assign(rep=lambda x: x[REP].mean(axis=1))
    ho = pd.read_csv(R / "human_only_raw.csv").rename(columns={"replay_mean": "rep"})
    co = pd.read_csv(R / "consistency_raw.csv").assign(rep=lambda x: x[REP].mean(axis=1))

    def test(df, model, rf="RF (stat)"):
        a = df[df.model == model].set_index("seed")
        b = df[df.model == rf].set_index("seed")
        won = int(((a.rep - b.rep) >= 0.10).sum())
        return round(a.rep.mean(), 2), round(b.rep.mean(), 2), f"{won}/{len(a)}"

    rows = [("GS-Fusion (gated) vs Random Forest", *test(v5, "GS-Fusion (gated)", "Random Forest (stat)")),
            ("Random Forest | SSL next-step error", *test(ho, "RF | SSL error (primary)")),
            ("Random Forest + consistency features", *test(co, "RF (stat + consistency)")),
            ("Exploratory: GMM (human-only)", *test(ho, "GMM (stat)")),
            ("Exploratory: Random Forest | GMM", *test(ho, "RF | GMM (stat)"))]
    print("\nTable 3 - pre-registered tests, CaptchaSolve30k E2 (replay = mean of rotated and raw replay)\n",
          pd.DataFrame(rows, columns=["model", "E2 replay", "RF in same run", "seeds won"]).to_string(index=False))


def tables45():
    d = pd.read_csv(R / "cross_dataset_raw.csv")
    rows = []
    for m in ["Logistic regression", "Random Forest", "LightGBM", "GMM (human-only)"]:
        e1 = d[(d.exp == "X2-E1") & (d.model == m)]
        e2 = d[(d.exp == "X2-E2") & (d.model == m)]
        rows.append({"model": m, "E1 replay": round(e1.replay.mean(), 3),
                     "E2 replay": f"{e2.replay.mean():.3f} ± {e2.replay.std():.3f}",
                     "E2 L1-L4 min": round(e2.L1_L4_min.mean(), 2), "E2 FPR": round(e2.FPR.mean(), 3)})
    print("\nTable 4 - SapiMouse, split by subject\n", pd.DataFrame(rows).to_string(index=False))
    t = d[d.exp.isin(["X1", "X3"]) & (d.threshold != "SapiMouse humans (re-set)")]
    print("\nTable 5 - human FPR at the source threshold (target 0.05)\n",
          t.groupby(["dataset", "threshold", "model"]).FPR.mean().unstack(0).round(2).to_string())
    rs = d[(d.exp == "X1") & (d.threshold == "SapiMouse humans (re-set)")]
    print("\nX1 with the threshold re-set on SapiMouse humans\n",
          rs.groupby("model")[["FPR", "L1_L4_min", "replay"]].mean().round(3).to_string())


if __name__ == "__main__":
    table2(); table3(); tables45()


def deep():
    """Rows for the two deep baselines in Table 2 (needs results/deep_baseline_raw.csv from notebook §9f)."""
    f = R / "deep_baseline_raw.csv"
    if not f.exists():
        print("\n(deep_baseline_raw.csv not found: run §9f of CaptchaSolve30k_RF_vs_GSFusion_5.ipynb and copy it to results/)"); return
    d = pd.read_csv(f); rows = []
    for m in d.model.unique():
        e1 = d[(d.model == m) & d.experiment.str.startswith("E1")]; e2 = d[(d.model == m) & d.experiment.str.startswith("E2")]
        rows.append({"model": m, "E1 replay / raw": f"{e1.human_replay.mean():.2f} / {e1.human_replay_raw.mean():.2f}",
                     "E2 replay / raw": f"{e2.human_replay.mean():.2f} ± {e2.human_replay.std():.2f} / "
                                        f"{e2.human_replay_raw.mean():.2f} ± {e2.human_replay_raw.std():.2f}",
                     "E2 L1-L4 min": round(e2[SYN].mean().min(), 2), "E2 FPR": round(e2.FPR.mean(), 3)})
    print("\nTable 2, deep baselines\n", pd.DataFrame(rows).to_string(index=False))


deep()
