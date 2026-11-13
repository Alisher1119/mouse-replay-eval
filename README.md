# Mouse-dynamics bot detection against unseen replay attackers

Code and raw results for the paper

> F. Alimova, A. Khayrillaev. *Mouse-Dynamics Bot Detection Against Unseen Replay Attackers: A Two-Dataset Evaluation of Supervised, Human-Only and Deep Detectors.* Submitted to the Journal of Information Security and Applications, 2026.

## What this repository shows

Mouse-movement bot detectors are usually tested on the same kinds of bots they were trained on. This study asks what happens when the attacker is new: a **replay bot** that moves the pointer along recorded human paths and was never seen in training.

- On **CaptchaSolve30k**, Random Forest catches 92% of replay sessions when replay is in training, but only **46%** when it is not (at a 5% false-positive rate). LightGBM and GS-Fusion fall to 25–31%.
- On **SapiMouse** the drop is larger: 88–96% → **6%**, i.e. chance level.
- Deep sequence models on raw trajectories fail the same way: a 1D-CNN catches 100% of known replay but only 28% of unseen replay.
- Three pre-registered remedies (gated fusion, human-only hybrids, motor-style consistency features) all failed.
- Detectors do not transfer between sites: a threshold set on CAPTCHA users flags 24–100% of SapiMouse users and 44–87% of Balabit users.
- Two evaluation pitfalls made the replay attacker look detectable until they were fixed: **device mixing** in stitched replay and **fixed-length windows** that leak session duration (Section 6.4 of the paper).

## Repository layout

```
notebooks/
  CaptchaSolve30k_RF_vs_GSFusion_5.ipynb   main experiment: attacker ladder, E1/E2, GS-Fusion, artefact checks
  CaptchaSolve30k_human_only.ipynb         human-only detectors, self-supervised encoder, hybrids
  CaptchaSolve30k_consistency.ipynb        motor-style consistency features, self-replay control
  cross_dataset_sapimouse.ipynb            SapiMouse replication and transfer to SapiMouse / Balabit
results/                                   raw per-seed outputs behind every number in the paper
scripts/summarize_results.py               rebuilds Tables 2–5 from results/
```

## Where each number in the paper comes from

| Paper | Notebook | File in `results/` (name written by the notebook) |
|---|---|---|
| Table 2, Fig. 1a | `CaptchaSolve30k_RF_vs_GSFusion_5` (§9–10) | `captchasolve30k_v5_raw.csv` (`captcha_raw.csv`) |
| Gate weights, Section 6.1 | same (§9) | `captchasolve30k_v5_gates.csv` (`captcha_gates.csv`) |
| Fig. 2a, device-mixing controls | same (§7b, §9c, §9d) | `captchasolve30k_replay_self_vs_cross.csv`, `captchasolve30k_replay_turn_joins.csv`, `captchasolve30k_replay_submovements.csv`, `captchasolve30k_replay_feature_diagnostics.csv` |
| Table 2 deep-baseline rows (1D-CNN, Transformer) | `CaptchaSolve30k_RF_vs_GSFusion_5` (§9f, GPU) | `deep_baseline_raw.csv` |
| 95% intervals and sign-flip tests in the text | `scripts/stats.py` | all of the above |
| Table 3 rows 2, 4, 5 | `CaptchaSolve30k_human_only` | `human_only_raw.csv` |
| Table 3 row 3, consistency AUCs | `CaptchaSolve30k_consistency` | `consistency_raw.csv`, `consistency_single_feature_auc.csv` |
| Table 3 row 1 | `CaptchaSolve30k_RF_vs_GSFusion_5` | `captchasolve30k_v5_raw.csv` |
| Tables 4–5, Figs. 1b and 3 | `cross_dataset_sapimouse` | `cross_dataset_raw.csv`, `cross_domain_shift.csv` (`cross_raw.csv`, `domain_shift.csv`) |

Check the tables without running any experiment:

```bash
python scripts/summarize_results.py   # Tables 2–5
python scripts/stats.py               # confidence intervals and significance tests
```

**Fig. 2b** (window-length pitfall) compares two runs of `cross_dataset_sapimouse.ipynb`. The released notebook excludes duration and path length (`DROP = ["log_dur", "log_path"]` in §9). To reproduce the leaky run, set `DROP = []`.

## Data

No dataset is included; the notebooks download them.

| Dataset | Use | Source | Licence |
|---|---|---|---|
| CaptchaSolve30k (Capycap Inc., 2026) | humans, replay pool | Hugging Face `Capycap-AI/CaptchaSolve30k`, streamed | Apache-2.0 |
| SapiMouse (Antal et al., 2021) | humans, replay pool | authors' page (`sapimouse.zip`) | see authors' terms |
| Balabit Mouse Dynamics Challenge | humans only | GitHub `balabit/Mouse-Dynamics-Challenge` | see repository |

The full CaptchaSolve30k file is about 5.3 GB; the notebooks stream only the first 2,000 non-touch sessions per game.

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/CaptchaSolve30k_RF_vs_GSFusion_5.ipynb
```

- The notebooks also run in Google Colab (they detect `/content`).
- A GPU is strongly recommended. Without one, the LSTM/GRU models are skipped automatically (`SKIP_SEQ = None`), and the tree models still run.
- Each run writes its outputs to a folder with a date suffix (for example `_2026-11-12`). Progress `.pkl` files let an interrupted run resume.
- Settings used in the paper: `N_PER_GAME = 2000`, `SEEDS = [0, 1, 2, 3, 4]`, `TARGET_FPR = 0.05`, `REPLAY_RESAMPLE = True`, `MATCH_SENSOR = True`.

## Main protocol in brief

- **Attacker ladder:** every bot reuses the waypoints and button states of a real human session. L1 linear, L2 linear + jitter, L3 Bézier–Fitts, L4 Bézier + tremor, L5a replay (nearest human segment from a held-out 20% pool, KD-tree, rotated and scaled), L5b raw replay (translated only).
- **Replay is re-sampled as one device:** the stitched path is rebuilt in continuous time and re-sampled at the source human's update rate. Without this, a single feature separates replay from humans (AUC 0.89).
- **Splits** by source human (CaptchaSolve30k, 64/16/20) or by subject (SapiMouse). Thresholds are set at 5% FPR on validation humans. Five seeds.
- **E1:** all attacker levels in training. **E2:** replay levels appear only in the test set.
- **Artefact check:** the best single-feature AUC per attacker level. A value of 0.95 or more marks a level as suspect.

## Citation

See `CITATION.cff`. Please cite the paper if you use this code or these results.

## AI use

Claude (Anthropic) was used to help write and debug the experiment code and to draft the paper text. The authors designed the study, ran all experiments and checked every reported number against the files in `results/`.
