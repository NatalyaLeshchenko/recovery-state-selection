# Experimental protocol

## Task and candidate states

All experiments use LIBERO Spatial task 0: pick up the black bowl between the plate and the ramekin and place it on the plate. Initial-state numbers denote scene configurations of this task.

Controlled interventions create two types of recovery state. A missed grasp is recorded when the empty gripper rises while the bowl stays on the table. A dropped object is recorded when the robot holds the bowl, loses contact after forced release, and the bowl falls before the task is complete. Object positions and gripper contacts from the simulator are used to verify these events.

To reproduce a recovery start, a new environment is initialized with the same starting arrangement and the recorded actions leading to the intervention are replayed. The resulting simulator state, actuator commands and observations are compared with the saved snapshot of that recovery start. Expert data pair the observation before each action with that action. Control runs at 20 Hz, so one action corresponds to 0.05 seconds of simulation time.

## VLA features

The pi0-FAST LIBERO policy is used without changing its weights. Its token predictions are used to compute four features: aleatoric uncertainty, epistemic uncertainty, entropy and the log probability of the chosen token.

The help detector uses the Strong classifier architecture from the INSIGHT paper. The classifier is trained for this project on manually annotated data. Each sample contains five executed actions (approximately 0.25 seconds) and is labeled as useful progress, error/no progress, or uncertain. These become continue, help and unknown labels. Unknown labels were excluded from training and evaluation.

The candidate pool contains eight verified recovery starts: one missed-grasp state and one dropped-object state for each of initial-state numbers 20–23. Each selector requests two starts from this pool, using seed 0 where random sampling is required:

- INSIGHT-style selection takes the highest first-query Strong logits. This adapts the help detector to offline ranking.
- F_geometry allocates one request to each failure type, with seeded selection within each type.
- Random samples from the common pool without replacement.

A fixed scripted operational-space controller provides expert continuations. The LIBERO goal predicate determines acceptance. All attempts, including retries and unsuccessful trials, contribute to acquisition cost. Shared starts reuse the same expert recording. Malformed FAST outputs are reported separately from completed continuations.

## ACT pilot

The earlier ACT (Action Chunking with Transformers) experiment uses disagreement between three nominal ACT models as its uncertainty baseline. Each adapted policy receives two successful recovery demonstrations and the same initial checkpoint, normalization and training recipe: 1,000 updates, learning rate 1e-5 and seed 0, with nominal-data replay. Deployment replans after each action.

All policies are evaluated on the same 15 reserved recovery starts and ten ordinary initial configurations. This is a single-task, single-seed pilot.

## Scope

Help detection, candidate selection and expert demonstration collection have been evaluated. VLA adaptation on the selected demonstrations and an independent comparison of adapted VLA policies have not been completed. The ACT and VLA results are reported separately.

Software: [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), [LeRobot](https://github.com/huggingface/lerobot), [OpenPI](https://github.com/Physical-Intelligence/openpi).
