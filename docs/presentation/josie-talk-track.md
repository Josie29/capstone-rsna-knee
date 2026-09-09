# Josie — talk-track crib sheet


## Slide scripts

**Slide 1 (~10s):
- Our capstone project was training KneeNet, starting with knee MRI reports through training a harnessed model that provides diagnostic support
- In the end, we achieved 0.887 Macro AUC from 570 GB of weakly labeled image data where only 1.3% of studies were labeled.

**Slide 2 (~50s):** 
- It was a combination of label mining and controlled model experiments that delivered our best macro AUC of 0.887
- Here on the left we see our performance over time/experiemnts ran and on the right side we see a table describing what data and model each experiemnt used to achieve its associated performance
- We ran 12 total experiments, initially scoring just 0.691 on a simple frozen resnet 34 model trained on just the original 58 labeled studies
  - then added both model complexity and data labeling complexity to achieve the additional ~0.20 higher performance
- I'm going to talk about the modeling side first then kelly will get into the data labeling after

**Slide 3 (~40s):**
- This diagram is our best model end to end, left to right — from a study's raw DICOMs
  all the way to the 12 probabilities
- Each study comes in as about 5 and a half series of DICOM slices; before any modeling
  we select the best series for each of 5 series types, order the slices by the
  scanner's geometry, and resample everything to a standard 24 slices at 224 pixels
- Then one shared EfficientNet encodes every slice. The encoder expects RGB values, but
  our images are just grayscale — so to maximize the information added in this step we
  employ a 2.5-dimensionality trick: red gets the slice before, green the slice itself,
  blue the slice after. This matters because real anatomy like a tear or a cyst spans
  neighboring slices, and a single flat slice can't show that
- From there it's attention twice: first each series pools its 24 slices down to one
  embedding, then the study pools its available series — and if a series type is
  missing, the mask just drops it from the vote
- That final study embedding maps to 12 independent probabilities, one per finding

**Handoff:**
- While these modeling improvements helped - a model can only be as good as the labels its trained on, so next I'm going to hand it over to Kelly to talk about the labeling process

## Q&A soundbites

- *The whole model in one breath:* "The encoder describes every slice; slice attention
  picks the slices that matter within each series; series attention picks the series
  that matter within the study; a mask keeps missing series out of the vote."
- *2.5D:* "It's 2.5D because we abuse the RGB channels — instead of color, they hold the
  neighboring slices, so a cheap pretrained 2D encoder still sees a bit of 3D."
- *"Shared" encoder:* "One encoder, not five — every slice of every series goes through
  the same weights; a five-vector bucket embedding tells the model which series type it
  was looking at."
- *~5.5 series:* "5.5 is series per study including duplicates; we keep the best series
  for each of five type-buckets — the sixth type, axial non-fluid, is the rarest and
  wasn't worth its compute."
- *If 0.887 = 0.887 comes up:* "Numerical coincidence — one is a test score on ~1,300
  studies, the other is the miner's agreement with 58 gold labels. Different quantities;
  we don't claim we hit the label ceiling."
