# Feasibility Study

## 1. Problem Definition
The current production model (`model/weights/best_model.pth`) exhibits extreme overfitting, achieving 99.94% Macro-F1 on PlantVillage but completely collapsing to 12.28% Macro-F1 on PlantDoc. This indicates it is not production-ready for real-world images and will reject or misclassify the vast majority of user uploads. 

## 2. Assessment of Existing Resources
- **Datasets**: We already have cleanly split `train.csv`, `val.csv`, and `test.csv` files encompassing both PlantVillage and PlantDoc. A cryptographic duplicate check confirmed there is zero leakage between train and test splits.
- **Codebase**: The `scripts/train.py` file is well-written and implements a `WeightedRandomSampler` that gives PlantDoc samples a weight of 5.0 to aggressively combat class imbalance and domain dominance by PlantVillage. It also implements an evaluation function that computes a unified `field_score` weighted heavily toward PlantDoc performance (65% PD / 35% PV).
- **Metadata Evidence**: The repository contains an abandoned checkpoint metadata file (`models/best_model_metadata.json`) showing that a model was previously trained (epoch 2) using this script, achieving 99.24% on PlantVillage and 73.94% on PlantDoc. 

## 3. Risks and Mitigations
- **Risk**: Retraining might accidentally overwrite the existing 99% PV production checkpoint.
- **Mitigation**: The production checkpoint is safely isolated in `model/weights/`. Training experiments write to `models/` by default. We will ensure strict separation of output paths.
- **Risk**: PV baseline drops significantly when trying to improve PlantDoc.
- **Mitigation**: Using `field_score` for checkpoint selection ensures the model must perform well on both domains simultaneously. 

## 4. Conclusion
**Feasible (Go/No-Go: GO).**
We can safely and rapidly improve the model before the deadline. The codebase already contains the necessary infrastructure to train a model that generalizes significantly better (reaching ~74% PlantDoc F1). By running a proper, tracked training cycle utilizing `scripts/train.py` and deploying the resulting model, we can bridge the generalization gap without integrating external datasets like PlantSeg.
