1. tie fluid senstiive and fat suppression into 1 for now atleast. Since they are the same on training data there is no actionable difference between the 2 currently.
2. Every per-view model outputs all 12 labels; no hardcoded view->finding masks. View
   relevance is learned in the per-label combiner (and read off per-label val AUC),
   with clinical expectations kept in docs as a prior to sanity-check against — if the
   measured table contradicts textbook radiology, suspect a bug or label noise first.
3. Interim ensemble weighting: the clinically-derived plane×abnormality matrix
   (src/knee/plane_prior.py, methodology in docs/clinical understanding/) serves as
   fixed per-label combiner weights while no validation split exists to learn them.
   Weights, not masks — nothing is zeroed — and #2 still governs: measured per-label
   AUC overrules the prior, and the learned combiner replaces it once issue #3 lands.
4. Local eval protocol (`gold58-cv`): pooled out-of-fold stratified k-fold cross
   validation on the gold studies, evaluated at ensemble level — per fold, fresh
   per-plane heads train on the fold's training studies and the production combiner
   merges the held-out predictions (src/knee/cv_gold.py; defaults 5 folds x 5
   repeats, seed 0). AUC is computed once over the pooled OOF rows, never per fold
   (rare labels make ~12-study folds undefined/noisy); the per-repeat macro spread is
   the error bar. This is what the experiments.md Val AUC column means. It ranks
   levers locally before spending a submission — the public LB stays the arbiter, and
   at n=58 only deltas well outside the repeat spread count as signal.
5. Gold-58 is retired as a special set (E003 onward). Training uses all 4,407 studies
   with the blended soft labels as-is — the 58 gold rows keep their blended values,
   no override, no holdout — and the #4 CV protocol generalizes to the full blended
   pool (`blended-cv`, src/knee/cv.py: heads fit on soft labels; stratification and
   AUC use labels thresholded at 0.5). Consequence accepted with eyes open: local CV
   now measures agreement with the report miner, not ground truth, so the public LB
   remains the truth check and the miner's quality (report-mining lever) caps what
   local numbers can mean. Checkpoints embed `label_source` so gold-era and
   blended-era weights can't be confused.
