# Results

## ACT adaptation pilot

Each adapted policy uses two accepted recovery demonstrations. Evaluation starts are shared across policies.

| Policy / selection | Recovery success | Ordinary-task success |
|---|---:|---:|
| Initial ACT | 14/15 | 7/10 |
| Ensemble disagreement | 12/15 | 7/10 |
| Failure-type coverage | 12/15 | 6/10 |
| Random | 13/15 | 9/10 |

The observed recovery difference between failure-type coverage and uncertainty selection is zero percentage points. Both succeed on 11 starts, both fail on two, and each succeeds alone on one. The experiment does not establish an advantage or equivalence between the selectors. None of the adapted policies exceeds the initial policy on this recovery test.

## VLA help detector

The independent test has 16 help labels and 26 continue labels; three unknown labels are excluded. Labels concern progress of the current action prefix, not final episode success.

| Detector | True positives | False positives | False negatives | True negatives | Help recall | False-ask rate | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Strong, INSIGHT-style | 3 | 3 | 13 | 23 | 18.75% | 11.54% | 0.273 |
| Mean token entropy | 4 | 9 | 12 | 17 | 25.00% | 34.62% | 0.276 |
| Mean negative log probability | 9 | 14 | 7 | 12 | 56.25% | 53.85% | 0.462 |

- TP (True Positives) — the expert labels the state as help, and the detector correctly requests assistance.
- FP (False Positives) — the expert labels the state as continue, but the detector unnecessarily requests assistance.
- FN (False Negatives) — the expert labels the state as help, but the detector incorrectly allows the policy to continue.
- TN (True Negatives) — the expert labels the state as continue, and the detector correctly allows the policy to continue.
- Help recall — `TP / (TP + FN)`; the fraction of expert-labeled help states correctly detected.
- False-ask rate — `FP / (FP + TN)`; the fraction of expert-labeled continue states incorrectly flagged for assistance.
- F1 score — `2TP / (2TP + FP + FN)`; the harmonic mean of precision and recall for the help class.

## Selected VLA states and expert acquisition

Each selector requests two states from the same eight-candidate pool. All expert attempts and retries are counted.

| Selector | Requested states | Accepted demonstrations | Attempts including retries | Expert actions including unsuccessful attempts |
|---|---|---:|---:|---:|
| INSIGHT-style | 022 dropped; 023 dropped | 1/2 | 4 | 598 |
| F_geometry | 021 missed grasp; 022 dropped | 1/2 | 3 | 487 |
| Random | 023 missed grasp; 023 dropped | 2/2 | 3 | 456 |

The latest expert attempt for state 022 finishes its sequence without satisfying the benchmark goal. INSIGHT and F_geometry share this start and recording. The other three distinct latest attempts are accepted.
