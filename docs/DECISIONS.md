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
