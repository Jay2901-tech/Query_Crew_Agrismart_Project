# Experiment Plan (Phase 8)

## 1. CURRENT PRODUCTION BASELINE
- **Production Checkpoint**: `model/weights/best_model.pth` (FROZEN - DO NOT MODIFY)
- **SHA256**: `94CD7DFF369884F3C77EA7D15490FDD1556D77439A78A76A27251808D48B31CF`
- **Architecture**: `tf_efficientnetv2_s` (EfficientNetV2-S)
- **Classes**: 38
- **Current Direct Audit Results**:
  - PlantVillage: Accuracy ≈ 99.96%, Macro-F1 ≈ 99.94%
  - PlantDoc: Accuracy ≈ 16.85%, Macro-F1 ≈ 12.28%
- **Artifact Provenance Discrepancy**: A metadata file (`models/best_model_metadata.json`) exists stating `epoch = 2`, `pv_f1 = 99.24%`, `pd_f1 = 73.94%`. This does not match the actual current production baseline evaluation (99.94% / 12.28%). This must be treated strictly as an artifact/checkpoint provenance issue, and the *current* directly measured 12.28% PlantDoc F1 is our starting point.
- **Other Models**: `models/best_model.pth` exists (SHA256: `69B6FA63...`). It must NOT be deleted, overwritten, replaced, or modified.

## 2. EXPERIMENT 1
**Objective**: Train a NEW candidate model using PlantVillage + PlantDoc datasets to improve PlantDoc/field generalization while maintaining acceptable PlantVillage performance.
- The experiment must be fully isolated under `models/experiment_1/`.
- PlantSeg must NOT be downloaded or added to this dataset.

### Required Configuration
- **Architecture**: `tf_efficientnetv2_s` (ImageNet pretrained)
- **Input Size**: 224x224
- **Classes**: 38
- **Optimizer**: AdamW
- **Learning Rate**: 2e-4
- **Weight Decay**: 0.01
- **Scheduler**: Cosine Annealing (`T_max` = 10)
- **Loss Function**: CrossEntropyLoss with Label Smoothing (0.1)
- **AMP**: Enabled
- **Epochs**: 10 (initial experiment)
- **Batch Size**: 64 (GPU has ~7GB VRAM available. If CUDA OOM occurs, fallback to 32). Do not change PyTorch/CUDA versions.

### Data Sampling
- Use a PlantDoc-aware `WeightedRandomSampler`.
- Initial weighting: PlantVillage = 1.0, PlantDoc = 5.0. Do NOT silently change this.
- Implementation/documentation must calculate and record:
  1. Number of PlantVillage training samples
  2. Number of PlantDoc training samples
  3. Effective sampling probability
  4. Approximate number/proportion of PlantDoc samples seen per epoch
  5. Per-class distribution
  6. Any severe class imbalance
  7. Whether the sampler actually produces the intended distribution
- *Note: If weighting needs modification, document as a separate future experiment rather than silently changing Experiment 1.*

### Augmentation Pipeline
Stronger, realistic field-oriented augmentation to reduce dependence on clean backgrounds, centered leaves, laboratory-style imagery, fixed camera conditions, uniform lighting, and predictable image composition. Must include appropriate combinations of:
- Random resized crop
- Horizontal flip
- Vertical flip (where biologically reasonable)
- Small rotations
- Brightness/contrast variation
- Color jitter
- Slight blur
- Mild noise
- Scale variation
- *Note: Avoid unrealistic transformations that destroy disease morphology. Document the exact augmentation pipeline. Do NOT modify production preprocessing.*

### Determinism
- Use a fixed random seed (record it).
- Configure deterministic behavior as reasonably as possible without making training unusably slow. Do not claim cross-machine determinism.
- After training, run a small same-input inference repeatability check on the candidate checkpoint to ensure confidences/predictions remain stable.

### Training History Requirements
Record metrics for EVERY epoch. At minimum record:
- `epoch`
- `train_loss`
- `validation_loss`
- PlantVillage validation Accuracy
- PlantVillage validation Macro-F1
- PlantDoc validation Accuracy
- PlantDoc validation Macro-F1
- `field_score` = `0.35 * PlantVillage Macro-F1 + 0.65 * PlantDoc Macro-F1` (Keep this weighting)
- `learning_rate`
- *Note: NEVER report only `field_score`. Always report individual PV and PD metrics.*

Also record: `configured_epochs`, `completed_epochs`, `best_epoch`, `final_epoch`, `best_model_is_best_epoch`, `best_model_is_final_epoch`. (Mandatory due to previous artifact provenance issues).

### Leakage Checks
Record findings for:
1. Train ↔ validation duplicates
2. Train ↔ test duplicates
3. Validation ↔ test duplicates
4. PlantVillage ↔ PlantDoc duplicates
5. Exact duplicate images within a split
- *Note: The critical question is whether the same image/content crosses evaluation boundaries. Do not automatically declare leakage merely because duplicates exist.*

## 3. FUTURE EXPERIMENT 2 / PLANTSEG
- **Do NOT use PlantSeg in Experiment 1.**
- PlantSeg is deferred to a potential SEPARATE Experiment 2. It will only be utilized if Experiment 1 fails to achieve sufficient field generalization.
- Role of PlantSeg (if used later): segmentation-based leaf/plant extraction, background variation, realistic crop generation, domain augmentation, or auxiliary segmentation learning. NOT blindly concatenated.

## 4. PROMOTION CRITERIA
The candidate model will only be considered for promotion if it passes all the following gates:
1. PlantDoc Macro-F1 improves substantially over the CURRENT DIRECTLY MEASURED production baseline (12.28%).
2. PlantVillage performance does not materially regress.
3. Field score improves.
4. No severe regression in important PlantDoc classes.
5. No class-mapping regression.
6. No safety regression (non-plant rejection, uncertainty behavior, confidence handling, inference determinism). The production threshold (0.60) and `model/predict.py` must NOT change during Experiment 1.
7. No train/test leakage.
8. Deterministic inference remains stable.
9. Checkpoint integrity is verified.
10. All experiment metadata and training history are complete.
*Note: Do NOT define success as "PlantVillage remains above 99%". Real-world generalization is the primary objective.*

## 5. EVALUATION
- **Model Selection**: FIRST evaluate validation data for model selection. TEST DATA MUST NEVER be used for model selection, hyperparameter selection, early stopping, or checkpoint selection.
- **Final Evaluation**: ONLY AFTER training is complete, evaluate test data.
- **Reporting Requirements**:
  - PlantVillage: Accuracy, Macro-F1, per-class F1
  - PlantDoc: Accuracy, Macro-F1, per-class F1
  - Overall `field_score`
  - Worst PlantDoc classes
  - Confusion matrix
  - High-confidence incorrect predictions
  - Confidence distribution
  - Calibration metrics (if the existing evaluation methodology is valid)
  - Non-plant rejection regression
  - Deterministic inference regression
- The frozen production checkpoint must be evaluated directly using the exact same verified evaluation methodology. Do NOT compare against old unverified metadata numbers as if they were verified baseline metrics.

## 6. PRE-EXECUTION CHECKLIST
Prior to any training, the execution phase must verify and explicitly log:
- [ ] Current production checkpoint SHA256 (`model/weights/best_model.pth`)
- [ ] Architecture (`tf_efficientnetv2_s`)
- [ ] Class count (38)
- [ ] Dataset class mapping (Validate the COMPLETE 38-class mapping: checkpoint index → checkpoint class string → canonical dataset class → mapping key/value → guidance mapping → production status). Do NOT assume missing classes are irrelevant. Do NOT modify `config/class_mapping.json`.
- [ ] Training CSV counts
- [ ] Validation CSV counts
- [ ] Test CSV counts
- [ ] Current experiment configuration
- [ ] Existing checkpoint metadata (`models/best_model_metadata.json`)
- [ ] Relationship between `models/best_model_metadata.json`, `models/best_model.pth`, and `model/weights/best_model.pth`. The provenance discrepancy must be explicitly reported.

## 7. EXPECTED EXPERIMENT OUTPUTS
The following completely isolated artifacts must exist after execution:
- `models/experiment_1/best_model.pth`
- `models/experiment_1/final_model.pth`
- `models/experiment_1/training_history.csv`
- `models/experiment_1/metadata.json` (Must contain: architecture, pretrained status, class count, dataset paths, dataset counts, sampler configuration, effective sampling distribution, augmentation configuration, optimizer, learning rate, scheduler, weight decay, loss, label smoothing, AMP status, batch size, actual batch size if OOM fallback occurred, configured epochs, completed epochs, best epoch, final epoch, seed, git/workspace revision if available, checkpoint SHA256 hashes, timestamp, training duration, validation metrics by epoch, test metrics after final evaluation).
- `models/experiment_1/config.json` (If practical)

## 8. DECISION TREE
- **IF Experiment 1 significantly improves PlantDoc**: Keep PlantSeg out and proceed to robustness/promotion evaluation.
- **IF Experiment 1 modestly improves PlantDoc**: Analyze errors and decide whether Experiment 2 is justified.
- **IF Experiment 1 fails**: Investigate PlantSeg/other field-oriented data separately (as Experiment 2).

TRAINING STATUS: NOT AUTHORIZED
