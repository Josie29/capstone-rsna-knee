# Josie — talk-track crib sheet

Speaker notes only — not rendered in the deck. Slide-by-slide scripts plus Q&A
soundbites, one rehearsal page. Timings target ~100s total for slides 1–3.

## Slide scripts

**Slide 1 (~10s):** "Three of us, one Kaggle competition: read a knee MRI, predict
12 findings, with almost no labels. I'll cover the model, Kelly the labels, Ryan the
product."

**Slide 2 (~50s):** "Twelve experiments, every one a controlled A/B. Two aren't on this
chart — spectacular fine-tune crashes, ask me in Q&A. Two lessons in the flat spots: a
fancier DINOv2 backbone moved nothing, E004; and the fine-tune only paid once the recipe
was fixed. The steps that mattered: mined labels, +0.081. Attention pooling. The
laterality fix — knees are mirror images, half our data was anatomically backwards. Then
the rebuild: EfficientNet with attention over slices then series, 0.868 — and non-fluid
series with deeper sampling took it to 0.887."

**Slide 3 (~40s):** "Before any model: pick the right five series out of each study's
pile of DICOMs, order the slices by scanner geometry, and resample everything to the
same shape. Then one shared encoder reads every slice, and attention twice — each series
pools its own slices, then the study pools its available series. Studies are ragged; the
mask handles missing series."

**Handoff:** "The biggest lever isn't in this diagram — +0.081 came from the labels."
(then verbally to Kelly)

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
