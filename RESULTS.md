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

The independent test has 16 help labels and 26 continue labels; three unknown labels are excluded. Labels concern progress of the current action prefix, not final episode success. All thresholds are selected on validation.

| Detector | True positives | False positives | False negatives | True negatives | Help recall | False-ask rate | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Strong, INSIGHT-style | 3 | 3 | 13 | 23 | 18.75% | 11.54% | 0.273 |
| Mean token entropy | 4 | 9 | 12 | 17 | 25.00% | 34.62% | 0.276 |
| Mean negative log probability | 9 | 14 | 7 | 12 | 56.25% | 53.85% | 0.462 |

Help recall is the fraction of help labels detected. False-ask rate is the fraction of continue labels incorrectly flagged. Strong makes fewer false requests than the scalar baselines but misses 13 of the 16 help labels. The sample comes from three reserved initial configurations.

## Selected VLA states and expert acquisition

Each selector requests two states from the same eight-candidate pool. All expert attempts and retries are counted.

| Selector | Requested states | Accepted demonstrations | Attempts including retries | Expert actions including unsuccessful attempts |
|---|---|---:|---:|---:|
| INSIGHT-style | 022 dropped; 023 dropped | 1/2 | 4 | 598 |
| F_geometry | 021 missed grasp; 022 dropped | 1/2 | 3 | 487 |
| Random | 023 missed grasp; 023 dropped | 2/2 | 3 | 456 |

The latest expert attempt for state 022 finishes its sequence without satisfying the benchmark goal. INSIGHT and F_geometry share this start and recording. The other three distinct latest attempts are accepted.

These are data-acquisition outcomes. They do not measure the recovery success of an adapted VLA. An equal budget of successful VLA recovery demonstrations has not yet been obtained for all methods.

The accompanying CSV tables contain the same aggregate measurements, transcribed from experiment outputs.
