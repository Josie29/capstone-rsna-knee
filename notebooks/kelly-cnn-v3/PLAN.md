# v3 retrain + submit plan (session 2026-09-07)

## Board
- [ ] 1. Pull v2 kernel + metadata (DONE -> work/v3/v2-pull)
- [ ] 2. Build knee-cnn-v3a (folds 0,1,2) and knee-cnn-v3b (folds 3,4) in work/v3/v3a, work/v3/v3b
- [ ] 3. Push both, poll to COMPLETE (background loop, ~10 min interval)
- [ ] 4. Pull outputs, dedupe log lines, extract per-fold best val_macroAUC + gold58-at-best
- [ ] 5. Gate: if v3 mean val AND mean gold58 both >0.01 worse than v2 (0.8402 / 0.8483) -> STOP, report
- [ ] 6. Pull kellyhe47/knee-submit-v1 -m; edit: BUCKETS (match training incl. any preflight drop),
       DEPTH=24, kernel_sources=[v3a,v3b], per-fold glob with assert all 5 found
- [ ] 7. Push submit-v1, poll COMPLETE, pull output, validate (rows, cmp header vs
       data/sample_submission.csv, [0,1] no NaN, per-study std>0.01, 0 prevalence fallbacks, 5 folds from 2 dirs)
- [ ] 8. Try `kaggle competitions submit -c rsna-knee-abnormality-detection` (code comp - likely refuses);
       else report kernel version + URL for manual submit
- [ ] 9. Final report: v3 vs v2 table, weak-set col deltas, wall times, nf coverage note, validation, version

## Exact v3 changes vs v2 (nothing else)
- --buckets sag_fs,cor_fs,axi_fs,sag_nf,cor_nf (adaptive: preflight drops sag_nf/cor_nf if chosen-series
  npz coverage <50% of studies; buckets.json written to kernel output, identical logic in both kernels)
- --depth 24 (v2 default 16)
- --bs 2 --accum 4 (memory; effective batch 8 unchanged)
- folds split: v3a=0,1,2  v3b=3,4
- If time limit hit: v3a:0-1, v3b:2-3, new v3c:4

## v2 facts (from pulled kernel, do not re-derive)
- 3 cells: [0] GPU guard sm_70+ SystemExit + pip iterative-stratification, [1] writes train.py, [2] runner
- runner: --cache /kaggle/working --volumes VOL --series-csv train_series.csv --labels labels.csv
  --weights label_weights.csv --gold gold_58.csv --fold k --folds 5 --epochs 10 --bs 4 --accum 2
  --model tf_efficientnet_b0 --oof-csv oof_foldk.csv --out /kaggle/working
- defaults: size 224, lr 3e-4, seed 42, DEFAULT_BUCKETS sag_fs,cor_fs,axi_fs, depth 16
- dataset_sources: kellyhe47/knee-labels-v1, barun2104/rsna-knee-mri-processed-3d-volumes
- docker: gcr.io/kaggle-private-byod/python@sha256:37c64f7...d461, machine NvidiaTeslaT4, internet on
- checkpoints: torch.save({"model": sd, "args": vars(args), "auc": auc}) -> fold{k}.pt (verify name in code)
- v2 baseline: mean val 0.8402, mean gold58-at-best 0.8483; weak cols Lat Men .785 / Lat OA .807 / MCL .811 / PF OA .815

## Resume procedure
Kernels live in work/v3/v3a, work/v3/v3b (kernel-metadata.json + ipynb). Status:
`~/.local/bin/kaggle kernels status kellyhe47/knee-cnn-v3a`. Outputs pulled to work/v3/out-v3a etc.
Log metric regex: `ep *N loss=... val_macroAUC=X gold58=Y` — dedupe lines first, best epoch by val.
