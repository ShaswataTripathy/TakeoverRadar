# TakeoverRadar

Model weights and the corpus aggregates are also on Hugging Face: [tripathyShaswata/TakeoverRadar](https://huggingface.co/tripathyShaswata/TakeoverRadar).

Risk model for **maintainer takeovers of npm packages**: stolen publish tokens,
phished maintainers, self-spreading worms (Shai-Hulud, s1ngularity) and
slow social takeovers (event-stream, the xz pattern). It has two modes:

1. **Monthly forecast.** For every popular package, how likely is a takeover
   this month, given only what was public before the month began?
2. **Outbreak mode.** Once the first victims of a campaign are reported, which
   other packages are the likely next (or already-hit) victims?

All numbers below are out-of-sample. Campaigns never appear in both training
and evaluation.

## Data

| | |
|---|---|
| Labels | OSV malicious-packages + GHSA: 1,909 takeover events in 70 campaigns, 2015-09 to 2026-10 |
| Universe | 119,994 most-downloaded npm packages (ecosyste.ms) |
| Panel | 6.84M package-months, 2017-01 to 2026-10. A row is at risk once a package has 3 or more versions and is at least 90 days old, until its first takeover |
| Positives in universe | 640 takeover months (235 before 2026, 405 in 2026) |
| Sources per row | packument history (maintainers, publishers, trusted publishing, provenance, cadence), corpus position (maintainer portfolios, dependents), git history of 40k repos (newcomers, maintainer fade, CI/build/fixture edits), contagion (recent reports sharing a maintainer or scope) |

Every feature in a row is computed from information published **before the
first day of that month**. Leakage checks: each label row falls in its event's
month, no rows follow a takeover, and no single feature separates takeovers
on its own (largest single-feature month-stratified AUC is 0.73).

## Models

* `popularity`: dependents within the corpus.
* `gbm_*`: LightGBM discrete-time hazard with a per-month baseline offset, so
  it learns *relative* risk within a month rather than when worm waves hit.
  `_eq` gives every campaign equal weight, so a single 90-package worm does
  not define what risky looks like.
* `pre` features: a causal transformer **pretrained self-supervised** on all
  5.4M pre-2026 package-months to predict each package's next month. It
  contributes a per-month *surprise* (how badly the history predicted this
  month, overall and per feature family) and an 8-dimensional summary of the
  history. A transformer trained directly on the labels memorised its roughly
  100 positives and ranked at chance (AUC 0.47). Pretraining on unlabelled
  history is what made the sequence model useful.
* `blend_*`: mean of the within-month percentiles from popularity and the GBM.

**Deployed model: `blend_pre_eq`.** It was declared primary before the
held-out test was run. Since 2026-10-27 it is trained on the OSV labels plus
81 takeovers mined from deleted versions (see *Unreported takeovers* below).
The OSV-only model is kept in `model_v1_osv_only/`.

## Results: monthly forecast

Metrics are stratified by calendar month: each victim is ranked only against
packages at risk in the same month. AUC is the share of those packages
ranked below the victim. recall@k% is the share of victims inside that month's
top k%. 95% intervals come from a cluster bootstrap over campaigns.

**Held-out 2026 (Jan to Oct), run once: 405 takeovers, 22 campaigns.**

| model | AUC | 95% CI | recall@1% | recall@5% | AUC, mean over campaigns |
|---|---|---|---|---|---|
| popularity | 0.563 | 0.51–0.62 | 0.005 | 0.030 | 0.546 |
| weak-links heuristic | 0.320 | 0.12–0.47 | 0.000 | 0.000 | 0.274 |
| gbm_full | 0.712 | 0.61–0.85 | 0.007 | 0.096 | 0.746 |
| gbm_full_eq | 0.738 | 0.64–0.84 | 0.005 | 0.099 | 0.726 |
| gbm_pre_eq | **0.745** | 0.63–0.87 | **0.035** | **0.141** | 0.730 |
| blend_full_eq | 0.698 | 0.64–0.77 | 0.005 | 0.076 | 0.672 |
| **blend_pre_eq (primary)** | 0.700 | 0.64–0.78 | 0.010 | 0.072 | 0.671 |

**Dev (before 2026, 5-fold campaign-grouped CV): 235 takeovers, 21 campaigns.**

| model | AUC | recall@1% | recall@5% | AUC, mean over campaigns |
|---|---|---|---|---|
| popularity | 0.659 | 0.047 | 0.204 | 0.755 |
| gbm_full | 0.618 | 0.009 | 0.060 | 0.757 |
| supervised transformer | 0.467 | 0.026 | 0.085 | 0.502 |
| gbm_pre_eq | 0.676 | 0.017 | 0.094 | 0.809 |
| blend_pre_eq | **0.706** | 0.043 | 0.179 | 0.836 |

What this says:

* **The primary model held up**: AUC was 0.706 on dev and 0.700 on the held-out test.
* **Popularity stopped working in 2026.** Its AUC fell from 0.66 to 0.56, which
  pulls both blends down. On the test, the GBM with pretrained features alone
  is best (0.745, recall@1% 0.035). This was found after the test, so it is
  reported as an observation, not as the headline result.
* **Pretraining helps**: compared with the same GBM without it, recall@1% rose
  from 0.005 to 0.035 and recall@5% from 0.099 to 0.141 on the test.
* **The ceiling is low and expected.** Most takeovers come from worm waves that
  pick victims within days by following stolen tokens. Data frozen at the start
  of the month cannot see that, which is why outbreak mode exists.
* The "weak links" checklist (single maintainer, no provenance, install scripts,
  stale) is **worse than random** in 2026: attackers went after active,
  well-maintained packages.

## Results: outbreak mode

For the 14 campaigns with 5 or more victims in the universe, the model scores
every other package at the moment the k-th victim is publicly reported (about
92k–108k candidates per campaign). Each campaign is evaluated with a model
trained on the other 13 campaigns.

Features: maintainer, organisation, scope and dependency proximity to reported
victims; the monthly risk prior; and **live publishing activity**, meaning
versions published since the campaign began, publish bursts across the
registry, and time to the nearest known victim's malicious release.

npm deletes malicious versions after the fact (97% are gone from the current
package records), but the packument `time` map keeps their publish
timestamps. The model uses those timestamps, because a live registry feed
shows them at the moment of publishing. It never uses the `unpublished` list
or the publisher of a removed version: both reflect the cleanup after the
report and would leak.

| victims known (k) | AUC | recall@10 | recall@100 | recall@1000 |
|---|---|---|---|---|
| 1 | 0.743 | 0.049 | 0.282 | 0.346 |
| 3 | 0.778 | 0.082 | 0.330 | 0.387 |
| 5 | 0.708 | 0.084 | 0.357 | 0.466 |
| 10 | 0.899 | 0.217 | 0.573 | 0.659 |
| 25 | **0.965** | 0.184 | **0.594** | **0.773** |

The monthly prior now comes from held-out predictions for the 2026 campaigns
too (previously they had none). That raised the prior's own AUC from 0.53 to
0.67–0.73 and the outbreak AUC at k = 1 and 3, while recall@100 at k = 25 fell
from 0.64 to 0.59. The earlier run is kept in `results/outbreak_*_v5.csv`.

With 25 victims known, checking the top 100 of about 100,000 packages finds
59% of the victims not yet reported. Without the live-publish features,
Shai-Hulud waves 1 and 2 ranked at chance (AUC 0.53 and 0.46); with them they
reach 0.97 and 0.94.

## Unreported takeovers mined from deleted versions

`build/mine_unreported.py` scans all 144,526 packuments for versions that are
in the `time` map but gone from `versions` (934,664 deletions in 34,747
packages). A small GBM learns what OSV-labelled malicious deletions look like
against every other eligible deletion. Its strongest signals are a deletion
burst among packages sharing a maintainer within 7 days, many deletions in the
package, and registry-wide deletion bursts. Out-of-fold, it separates OSV
takeover deletions from the rest with AUC 0.999; the top 1,000 packages are 99.9%
OSV takeovers. Packages scoring above the point where the local OSV share drops
below 50%, and not in OSV, become **81 mined takeovers** (58 of them in
2025–26), listed in `results/mined_candidates.csv`. Several match publicly
reported incidents that are missing from OSV, such as `rand-user-agent`
(2025-05) and `@nx/key` (2025-08, the s1ngularity day).

**False-negative audit.** Where the existing model ranked these 76 mined
victims (with a panel row) in their takeover month, all scores out-of-sample:

| score | mean percentile |
|---|---|
| popularity | 0.43 |
| gbm_pre_eq | **0.70** |
| blend_pre_eq | 0.58 |

The GBM ranks unreported takeovers as well as it ranks reported ones. That is
evidence it learned takeover risk, not reporting. Popularity ranks them *below
chance*: unpopular packages get taken over and nobody files a report. So part
of popularity's strength on OSV labels is reporting bias, the same effect that
makes the blend weaker than the GBM on the 2026 test.

**As extra training labels** (evaluation always on OSV events only, mined
packages removed from the risk sets):

| | dev AUC | dev AUC, mean over campaigns | 2026 test AUC | test, mean over campaigns | test recall@1% |
|---|---|---|---|---|---|
| blend_pre_eq, OSV only | 0.706 | 0.836 | 0.700 | 0.671 | 0.010 |
| blend_pre_eq, + mined | 0.721 | 0.827 | 0.681 | 0.687 | 0.010 |
| gbm_pre_eq, OSV only | 0.677 | 0.809 | 0.745 | 0.730 | 0.035 |
| gbm_pre_eq, + mined | 0.691 | 0.771 | 0.706 | 0.743 | 0.015 |

The rule fixed before the test run was: adopt mined labels if dev AUC does not
drop. Dev improved, so they were adopted. The test, run once, does not confirm
the gain: AUC is lower, the mean over campaigns is higher, and every change is
inside the confidence intervals. Treat mined labels as neutral for accuracy;
their main value is the audit above.

## Weekly panel (a negative result)

`build/panel.py --freq week` rebuilds the panel at Monday granularity, so
contagion features can see takeovers reported the previous week. Never-hit
rows are sampled 1 in 4 by a hash of name and week, which gives 7.5M rows. On
dev it was worse than the monthly panel (blend 0.700 against 0.721, GBM 0.666
against 0.691, both with mined labels), so the test was not run on it. The
reason: contagion features are non-zero for only 0.2% of takeover rows in
either panel. Victims of a wave rarely share a maintainer or scope with an
already-*reported* package. Within-wave spread is handled by outbreak mode,
which uses live publishes rather than reports.

## Results: synthetic xz-style takeover (a negative result)

The xz pattern was injected into the real histories of 400 never-compromised
packages: a newcomer ramps up commits and touches CI, build files and test
fixtures; the long-time author fades; the newcomer is added as a maintainer,
then publishes. The share of injected packages in the top 5% of risk *fell*,
from 5.8% to 0.2% by month 7, against roughly 5% for the untouched copies.

**This model does not detect slow social takeovers.** It learned what real
labelled takeovers look like (token theft and worms), and the labels contain
almost no slow takeovers. Adding a maintainer even lowers predicted risk,
because packages with more maintainers are taken over less often in real npm
data. Use a separate rule-based alert for this pattern. The `g_newcomer_*`,
`r_maint_added_*` and `r_new_publisher_*` features are exposed for that
purpose.

## Usage

```bash
pip install -r requirements.txt
# score.py reads the corpus aggregates (122 MB, too large for GitHub):
python -c "from huggingface_hub import hf_hub_download as d; d('tripathyShaswata/TakeoverRadar', 'data/interim/corpus.pkl', local_dir='.')"
python score.py axios left-pad @ctrl/tinycolor
python score.py --file names.txt --json
```

`score.py` fetches the live packument and git history, rebuilds the package's
monthly history, runs the pretrained encoder and reports the blended risk
percentile among about 106k at-risk popular packages, with the features that
moved the score most.

## Limitations

* The universe is today's most-downloaded packages, so packages that were
  popular in the past but have since faded are under-represented.
* Labels are what OSV and GHSA recorded, plus 81 mined takeovers with
  roughly 50%+ precision. Other unreported takeovers are counted as negatives.
* Outbreak results come from 14 campaigns, and the features were designed while
  looking at those same campaigns. Expect some optimism.
* Scores rank risk. They are not calibrated probabilities.

## Data sources and attribution

* Takeover labels: [OpenSSF malicious-packages](https://github.com/ossf/malicious-packages)
  (via [OSV](https://osv.dev)) and the [GitHub Advisory Database](https://github.com/advisories).
  Their data is used under their respective licenses.
* Package metadata: the public npm registry. Package popularity: [ecosyste.ms](https://ecosyste.ms).
* `data/interim/mined_events.jsonl` and `results/mined_candidates.csv` list
  packages the miner flags as *likely* takeovers that are absent from those
  databases. They are statistical candidates (about 50%+ precision), not
  confirmed incidents.

Raw packuments, git histories and the panels are not in this repository (they
are gigabytes); the commands below rebuild them.

## Reproduce

```bash
python -m takeover_radar.collect.osv && python -m takeover_radar.build.labels
python -m takeover_radar.collect.universe && python -m takeover_radar.collect.packuments
python -m takeover_radar.collect.git_history
python -m takeover_radar.build.panel --workers 4        # --merge-only if the final merge was interrupted
python -m takeover_radar.build.mine_unreported --workers 10
python -m takeover_radar.models.pretrain
python -m takeover_radar.experiments --split dev --models popularity,gbm_full_eq,gbm_pre_eq,blend_full_eq,blend_pre_eq
python -m takeover_radar.experiments --split test --models popularity,gbm_full_eq,gbm_pre_eq,blend_full_eq,blend_pre_eq
python -m takeover_radar.experiments --split dev --models popularity,gbm_pre_eq,blend_pre_eq --mined train --tag mined_train
python -m takeover_radar.build.panel --freq week --workers 10   # weekly variant (not used)
python -m takeover_radar.outbreak
python -m takeover_radar.eval.synthetic_xz --n 400
python -m takeover_radar.train_final
```
