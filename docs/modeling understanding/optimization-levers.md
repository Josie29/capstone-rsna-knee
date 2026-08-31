# Optimization levers

The four independent levers that determine our score. Each can be improved separately
and measured separately (one experiments.md row per pull). Ordered by expected impact.

## 1. Training signal — how many labeled studies, and how right the labels are

Quantity × quality of (study, 12 labels) pairs. Today: 58 gold studies. The report
miner (issue #2) raises quantity to ~4.4k; its error rate caps everything downstream —
a better miner beats a better model. Later: soft labels + noise-aware losses so the
model doesn't confidently learn miner mistakes.

## 2. Input selection — which pixels from each study the model sees

A study has ~5.5 series × ~30 slices; we currently feed one fluid-sensitive sagittal
series, squashed to 224px. Levers: more series types (findings live in different
planes: MCL→coronal, PF OA→axial), slice budgeting for the long tail, physical-space
resampling (mm-per-pixel, orientation) instead of naive resize. Also the main
inference-runtime knob for the efficiency track.

## 3. Architecture — what the network does with those pixels

Today: frozen ImageNet ResNet-34 + mean/max pooling + linear head. Ladder: fine-tune
end to end → attention pooling over slices → multi-series fusion → report-text
leverage at train time (contrastive pretraining / distillation — reports are absent at
test time, so their knowledge must be baked into image weights).

## 4. Fitting & evaluation procedure — how we train and how we know it worked

Today: fit on all 58, in-sample AUC only. Ladder: train/val holdout (issue #3) →
k-fold cross-validation when val noise starts driving decisions → ensembling across
folds/backbones → distill the ensemble back to one model for the efficiency prize.
The eval half is the meta-lever: every other lever is only as good as our ability to
measure it.

## Cross-cutting constraint

9h offline inference for ~1,300 studies, and the efficiency score prices runtime
directly — every lever above pays in AUC-per-second, not AUC alone.
