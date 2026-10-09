# Experimental protocol

## Task and candidate states

All experiments use LIBERO Spatial task 0: pick up the black bowl between the plate and the ramekin and place it on the plate. Initial-state IDs denote scene configurations of this task.

Two controlled interventions create recovery starts. A missed grasp is verified by an empty rising gripper, negligible bowl motion and no bilateral hold. A dropped object requires a confirmed hold before release, loss of bilateral contact and a measured fall before task completion. These labels use privileged simulator geometry and contacts.

Action prefixes are replayed in fresh environments. Simulator state, actuator control and observations are checked against the recorded start. Expert observations precede their paired actions. Control runs at 20 Hz; the gallery shows simulation time.

## VLA selection

The frozen pi0-FAST LIBERO policy supplies per-token aleatoric uncertainty, epistemic uncertainty, entropy and chosen-token log probability. A Strong help classifier is trained on human labels of progress over the current five-action prefix. A help label does not mean that the whole episode must eventually fail. Unknown labels are excluded.

Measured detector splits use initial configurations 0–3 for training, 10–11 for validation and 15–17 for testing. Validation selects the checkpoint and decision threshold. The test has 42 labeled queries from three configurations; queries from the same configuration are correlated.

Selection uses eight verified candidates from configurations 20–23, a budget of two requested starts and seed 0:

- INSIGHT-style selection takes the highest first-query Strong logits. This adapts the help detector to offline ranking.
- F_geometry allocates one request to each failure type, with seeded selection within each type.
- Random samples from the common pool without replacement.

A fixed scripted operational-space controller provides expert continuations. The LIBERO goal predicate determines acceptance. All attempts, including retries and unsuccessful trials, contribute to acquisition cost. Shared starts reuse the same expert recording. Malformed FAST outputs are reported separately from completed continuations.

## ACT pilot

The earlier ACT experiment uses disagreement between three nominal ACT models as its uncertainty baseline. Each adapted policy receives two successful recovery demonstrations and the same initial checkpoint, normalization and training recipe: 1,000 updates, learning rate 1e-5 and seed 0, with nominal-data replay. Deployment replans after each action.

All policies are evaluated on the same 15 reserved recovery starts and ten ordinary initial configurations. This is a single-task, single-seed pilot.

## Scope

VLA detection and expert acquisition have been measured. VLA adaptation on the selected demonstrations and an independent comparison of adapted VLA policies have not been completed. The ACT and VLA results are reported separately.

Software: [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), [LeRobot](https://github.com/huggingface/lerobot), [OpenPI](https://github.com/Physical-Intelligence/openpi).
