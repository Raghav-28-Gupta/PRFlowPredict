# Phase 3 Design — Point-in-Time Features

Written for: whoever implements Phase 3 and whoever trains models on its output in Phase 4.

**Status:** approved in design review 2026-09-17; awaiting spec review.
**Depends on:** Phase 2 gate PASS on the 45-repo cohort (`docs/phase2_eda.md`,
`data/phase2_gate.json`, `data/cohort/kept.json`). Implementation does not start before
that gate passes.
**Produces:** `data/features/features.parquet` (one row per modelling PR), a feature
dictionary, and a Phase 3 gate that proves — by independent recomputation and by manual
inspection — that no feature sees the future.

---

## 1. Purpose

Phase 2 built the label and proved the replay engine leak-free for two features. Phase 3
extends that engine to the full blueprint R1 feature list and assembles the table Phase 4
trains on. The discipline is unchanged: **every value in a row must have been knowable
at that PR's `created_at`.** Where a value can only be approximated (a reconstructed
at-open diff, a reverse-replayed draft flag), the approximation is carried alongside a
fidelity flag so later phases can measure its cost rather than trust it.

Blueprint §3 checkpoint: "After Phase 3, confirm zero leakage by spot-checking 5 rows
manually against raw timestamps." This spec keeps that and adds an automated audit.

## 2. Decisions locked in design review

| Decision | Choice | Why |
|---|---|---|
| Architecture | Four independent feature groups composed by `features.py` | Each group has different testing needs; Phase 4 can ablate by group; the leakage-critical group keeps its brute-force twin. |
| Scope | Blueprint R1 list **+ `author_prior_slow_rate_here`** | "Does this person's PRs usually get reviewed here" mirrors the repo trailing rate at author level and is free once author-history replay exists. |
| Author slow-rate window | Author's **whole** prior history in the repo (no 90-day window) | Most authors have few PRs; a window would usually be empty and fall to the prior. |
| Leakage gate | Automated brute-force audit on **all** replay features **+** manual 5-row spot-check | The audit proves the code; the spot-check proves the data. Two implementations sharing one wrong assumption pass the audit together. |
| Fidelity flags | Carried as columns, **never features** | Phase 4 ablates on them; Phase 6 slices error analysis on them. `body_edited` in particular is post-open information. |
| Sequencing | Design now; **gate first, then build** | Phase 3 code must not be built on an unvalidated cohort. |

## 3. Module map

```
replay.py        (extend) History carries author + merged_at; features_at(t, author)
                 adds trailing-7d, author history, author slow-rate;
                 brute_force_features recomputes every key independently
features.py      (new) static_features, at_open_features, replay_features,
                 repo_features, build(); CLI: build | --audit | --explain <pr_id>
tests/test_features.py, tests/test_replay.py (extend)
docs/feature_dictionary.md   one row per output column
data/features/features.parquet
```

`splits.py`, `labels.py`, `load.py`, `cohort_qc.py` are consumed unchanged.

## 4. Row set

Identical to Phase 2's modelling set: `splits.modelling_prs(pr_tier2)` (human-authored,
in-window) restricted to `cohort_qc.kept_repos()`, inner-joined with the D5 label. Row
count must equal Phase 2's `rows` for the same `kept.json` — gate check #1.

Key columns carried through: `repo, pr_id, number, created_at`.
Label columns carried through: `is_slow, wait_h, wait_h_censored, event_observed,
never_reviewed_30d`.

## 5. Replay extension (`replay.py`)

### 5.1 `History` gains two arrays

`author: np.ndarray[object]` (login; `None` for deleted accounts) and
`merged_at: np.ndarray[datetime64[ns]]` (NaT if not merged). Both sorted with the
existing arrays by `created_at`. `from_frames` reads them from Tier 1 (`author_login`,
`merged_at`).

### 5.2 `features_at(t, global_rate, global_merge_rate, author=None, alpha=5.0)`

Signature change: `global_merge_rate` is added (positional after `global_rate`); `author`
is keyword-only. Existing callers (`baseline.py`, `eda_report.py`) pass `global_merge_rate`
too — it is cheap to compute and keeps one signature.

Existing keys unchanged: `open_backlog_at_t`, `trailing_90d_slow_rate`, `trailing_n`,
`trailing_window_complete`.

New keys, all over the prefix `created_at < t` (bisect, as before):

| Key | Definition |
|---|---|
| `prs_opened_trailing_7d` | count with `created_at >= t − 7d` |
| `is_first_pr_here` | no prefix row with `author == author` |
| `n_prior_prs_here` | count of prefix rows by `author` |
| `n_prior_merged_here` | count of those with `merged_at` not NaT **and `merged_at < t`** — a PR open at `t` and merged later is not prior knowledge |
| `prior_merge_rate_here` | `(n_prior_merged + α·g_merge) / (n_prior + α)` |
| `days_since_first_pr_here` | `(t − min created_at of author's prefix rows)` in days; **NaN** if first |
| `author_prior_slow_rate_here` | over the author's prefix rows with a non-NaN label that are **resolvable at t** (`created_at <= t − 168h` OR `first_event_at < t`): `(n_slow + α·g) / (k + α)` |
| `author_prior_n` | that `k` — how much history the author rate rests on |

When `author is None` (deleted account): `is_first_pr_here = True`, counts 0,
rates = prior, `days_since_first_pr_here = NaN`. Documented as a category, not a crash.

### 5.3 `brute_force_features` extended

Recomputes every key above with pandas boolean masks and shares no code with `History`.
Same signature extension (`global_merge_rate`, `author`).

## 6. Feature groups (`features.py`)

Each is a pure function returning a DataFrame indexed by `pr_id` (or `repo` for repo
features). None reads the network or writes to disk.

### 6.1 `static_features(prs) -> DataFrame`

| Column | Derivation | Status |
|---|---|---|
| `created_hour_utc` | `created_at.hour` | static |
| `created_dayofweek` | `created_at.dayofweek` (Mon=0) | static |
| `is_weekend` | dayofweek ≥ 5 | static |
| `is_cross_repository` | passthrough | static |
| `author_account_age_days` | `(created_at − author_created_at).days`; **NaN** if deleted | static |
| `body_len` | `len(body_current or "")` | approximate — body is editable |
| `has_body` | `body_len > 0` | approximate |
| `body_edited` | `last_edited_at` not NaT | **flag** |

### 6.2 `at_open_features(prs, timeline, repo_meta) -> DataFrame`

Reverse-replay per PR from the current snapshot, using only events with
`event.created_at > pr.created_at` ("post-open"):

| Column | Derivation | Status |
|---|---|---|
| `is_draft_at_open` | start at `is_draft_current`; for post-open `ReadyForReviewEvent`/`ConvertToDraftEvent` walked newest→oldest, invert; result is the state before the earliest post-open event | reconstructed |
| `n_labels_at_open` | `n_labels_current − #post-open LabeledEvent + #post-open UnlabeledEvent`, floored at 0 | reconstructed |
| `title_at_open` | `previous_title` of the **earliest** post-open `RenamedTitleEvent`, else `title_current` | reconstructed |
| `title_len_at_open` | `len(title_at_open)` | reconstructed |
| `base_ref_at_open` | `previous_ref` of the earliest post-open `BaseRefChangedEvent`, else `base_ref_current` | reconstructed |
| `base_is_default` | `base_ref_at_open == repo_meta.default_branch` | reconstructed |
| `reviewer_requested_at_open` | any `ReviewRequestedEvent` with `created_at <= pr.created_at + 60s` | static (event is timestamped) |
| `n_reviewers_requested_at_open` | count of those | static |
| `requested_team_at_open` | any of those with `requested_reviewer_type == "Team"` | static |
| `additions_at_open`, `deletions_at_open`, `n_commits_at_open`, `diff_is_exact` | passthrough from parse.py | reconstructed / **flag** |
| `timeline_may_be_truncated` | the PR has exactly 60 timeline rows (the `first:60` cap) | **flag** |

Events at exactly `pr.created_at` are treated as at-open (not post-open).

### 6.3 `replay_features(rows, tier1, labels, g, g_merge) -> DataFrame`

One `History` per repo; one `features_at(created_at, g, g_merge, author=author_login)`
call per row. `g` and `g_merge` are the D5 slow rate and merge rate over **Scenario A
training rows** (`created_at < 2026-01-01`) — the same convention as Phase 2's baseline,
so the feature table is built once and is valid for Scenario A. For Scenario B, Phase 4
recomputes the two priors per fold if it wants to be strict; the effect of a global
prior shift on a shrunk rate with α=5 is small and is noted in the feature dictionary.

Columns: every key from §5.2.

### 6.4 `repo_features(repo_meta, prs) -> DataFrame`

| Column | Derivation | Status |
|---|---|---|
| `n_assignable_users`, `n_mentionable_users` | passthrough | snapshot (HEAD at collection) |
| `owner_is_org` | `owner_type == "Organization"` | snapshot |
| `has_codeowners`, `has_pr_template`, `has_contributing`, `n_ci_workflows` | passthrough | snapshot |
| `language_dominant` | passthrough, string | snapshot |
| `repo_age_days_at_open` | `(pr.created_at − repo.created_at).days` | static, point-in-time safe |

Snapshot status is stated in the dictionary; these are the blueprint's transferable
cold-start features and are accepted with that limitation.

## 7. Assembly (`features.build`)

```
rows      = modelling rows ∩ kept, joined with D5 label       (§4)
static    = static_features(rows)
at_open   = at_open_features(rows, timeline, repo_meta)
replay    = replay_features(rows, tier1, label_d5, g, g_merge)
repo      = repo_features(repo_meta, rows)
table     = rows[keys + labels].join(static).join(at_open).join(replay).join(repo on repo)
```

Written to `data/features/features.parquet` with a `built_at` and `git_sha` in the
Parquet metadata. Column order: keys, labels, then groups in the order above, then flags.

CLI: `python features.py` builds; `python features.py --audit` runs the gate (§9);
`python features.py --explain <pr_id>` prints the manual spot-check for one row (§9.5).

## 8. Feature dictionary (`docs/feature_dictionary.md`)

One row per output column: `column | group | dtype | status | derivation | nullable`.
Status values: `key`, `label`, `static`, `reconstructed`, `replay` (proven by audit),
`snapshot`, `flag`. Phase 4 trains only on `static`, `reconstructed`, `replay`,
`snapshot`; never on `key`, `label`, `flag`. The dictionary is the contract; a column
absent from it must not be used.

## 9. Phase 3 gate — pre-registered

| # | Check | Pass | On fail |
|---|---|---|---|
| 1 | Row count equals Phase 2's modelling row count for the same `kept.json` | exact | rows were lost in a join — stop |
| 2 | **Brute-force audit**: every `replay` column on ≥500 seeded rows recomputed independently | max \|Δ\| < 1e-9 on rates, exact on counts | **HARD STOP** — replay leaks or diverges |
| 3 | No NaN except in documented-nullable columns (`days_since_first_pr_here`, `author_account_age_days`) | exact | a join or reconstruction produced a hole |
| 4 | `timeline_may_be_truncated` rate | < 2% | raise `first:60` in Phase 1 and re-collect timeline (cheap) |
| 5 | **Manual spot-check**: `--explain` on 5 seeded rows; the user confirms each feature's contributing timestamps against the GitHub UI | user sign-off | a data-level error both implementations share |

Gate 1–4 are printed by `--audit` and written to `data/phase3_gate.json`. Gate 5 is a
human step and is recorded by hand in the gate JSON.

## 10. Testing

Toy fixtures with hand-computed answers, extending the Phase 2 pattern:

- `replay_toy` gains `author_login` and `merged_at`; a new test asserts the author
  keys at `t = 2024-04-21` for author `u1` (P1 and P3: 2 prior, 1 merged before t if
  P1 is given a `merged_at` of 2024-03-05, P3 not merged → `n_prior_merged_here = 1`,
  `days_since_first_pr_here = 51`), and first-PR values for an unseen author and for
  `None`.
- `test_brute_force_matches_features_at_on_toy` is extended to compare all new keys.
- `at_open` toy: 4 PRs covering draft→ready after open, label added-then-removed after
  open, title renamed twice (earliest previous wins), a `ReviewRequestedEvent` at +1s
  and one at +2h (only the first counts), and one PR with exactly 60 events (flag).
- `static` and `repo` toys: deleted author → NaN age; default-branch match.
- `build` on the two pilot repos end-to-end: row count matches `splits.prepare_rows`,
  no unexpected NaN, audit passes.

## 11. Out of scope

Model training, feature scaling/encoding (LightGBM takes raw values and category
strings), feature selection, SHAP — all Phase 4/6. Re-collection to raise the timeline
cap — only if gate #4 fails.
