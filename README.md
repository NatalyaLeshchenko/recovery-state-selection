# Recovery State Selection for Vision-Language-Action Models

This project studies how to choose recovery demonstrations after a failed grasp or a dropped object. The main question is whether covering different failure types improves recovery more than selecting states by model uncertainty, given the same demonstration budget.

![Selected recovery demonstrations](assets/recovery-grid.gif)

## Experiment

The experiments use a Panda manipulator in LIBERO Spatial. The robot must pick up a specified black bowl and place it on a plate. Controlled interventions create missed grasps and dropped objects; motion and contact checks verify each failure before adding the state to the candidate pool.

Three selection methods request demonstrations from the same pool:

- **INSIGHT-style selection** ranks states using a help classifier trained on token uncertainty features from a frozen pi0-FAST policy.
- **Failure-type selection (F_geometry)** chooses states from both failure types, using simulator geometry and gripper contacts.
- **Random selection** samples states without replacement.

The videos show a fixed scripted expert continuing from the selected states. Shared states use the same recording. Success is checked by the LIBERO task predicate.

## Results so far

A separate ACT pilot tested adaptation with two successful recovery demonstrations per method. Its uncertainty baseline used ensemble disagreement. On 15 reserved recovery starts, the initial policy succeeded in 14 cases, uncertainty selection and failure-type selection in 12 each, and random selection in 13. This pilot did not show a benefit from recovery fine-tuning or an advantage of failure-type selection over uncertainty.

The pi0-FAST experiments cover token-feature extraction, human progress labels, an independent help-detector test, and expert data collection from selected states. Adaptation and independent recovery evaluation of the VLA remain the next step.

## Protocol and tables

[Experimental protocol](METHOD.md) · [Results](RESULTS.md) · [CSV tables](results/)
