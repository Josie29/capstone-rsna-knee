# Plane × abnormality relevance matrix

Which MRI plane best evaluates each of our 12 findings, graded from clinical
literature. Used as fixed per-label ensemble weights (E002) until a validation split
lets us learn them (DECISIONS.md #2/#3). Canonical machine-readable copy:
`src/knee/plane_prior.py` — this table renders it; if they diverge, the code wins.

## Methodology

Consulted (2026-09-01): the RSNA *Radiology* sports-imaging review of knee ligament and
meniscus imaging; Medscape's ACL-MRI and Baker's-cyst imaging references; *Topics in MRI*
knee protocol review; *MRI Clinics* synovitis imaging review; OA/BML literature on
fluid-sensitive sequence assessment (PMC); and the MRNet paper (PLOS Medicine) as the
closest ML precedent (three per-plane CNNs merged per label by logistic regression).

Each (plane, finding) pair is graded **3 = plane of choice, 2 = useful, 1 = limited**.
Deliberately coarse: the literature supports ordinal claims ("axial is the
patellofemoral plane of choice"), not calibrated ratios — finer weights would be false
precision. Weights are renormalized per label over whichever planes a study actually
has, so grades are relative, not absolute.

## Matrix (fluid-sensitive series)

| Finding | Sagittal | Coronal | Axial | Rationale (one line) |
|---|---|---|---|---|
| ACL | 3 | 2 | 2 | Runs in ~sagittal plane; coronal/axial confirm femoral attachment |
| MCL | 1 | 3 | 2 | Vertical medial structure — coronal shows full length |
| Medial Meniscus | 3 | 3 | 1 | Horns on sagittal, body/radial tears on coronal — coequal |
| Lateral Meniscus | 3 | 3 | 1 | Same as medial |
| Medial OA | 2 | 3 | 1 | Weight-bearing compartment cartilage/joint space: coronal |
| Lateral OA | 2 | 3 | 1 | Same as medial |
| PF OA | 2 | 1 | 3 | Patellofemoral cartilage: axial is the plane of choice |
| Effusion | 2 | 1 | 3 | Suprapatellar recess graded on axial; sagittal shows extent |
| Synovitis | 2 | 1 | 3 | Axial fat-sat shows synovial thickening; scored axial+sagittal |
| Baker's | 2 | 1 | 3 | Diagnostic neck (gastrocnemius–semimembranosus) is an axial finding |
| Contusion | 2 | 2 | 2 | Marrow edema lights up on any fluid-sensitive plane — flat prior |
| Fracture | 2 | 2 | 2 | Multiplanar by nature (plateau/condyle/patella) — flat prior |

## Known uncertainties

- **Flat rows (Contusion, Fracture)** reduce to the old uniform mean — the prior
  claims nothing where the literature doesn't.
- **MRNet caution:** their *learned* weights found axial unexpectedly strong for ACL —
  human plane-of-choice and CNN plane-of-usefulness are not the same thing. This is
  exactly why the prior is interim: measured per-label AUC overrules this table, and
  the learned combiner replaces it once issue #3 lands.
- Grades assume fluid-sensitive contrast; non-fluid rows get added if model #4 ever
  joins the ensemble.

Sources: [Radiology sports imaging series](https://pubs.rsna.org/doi/full/10.1148/radiol.2016152320),
[Medscape ACL MRI](https://emedicine.medscape.com/article/400547-overview),
[Medscape Baker cyst imaging](https://emedicine.medscape.com/article/387399-overview),
[Topics in MRI: The Knee](https://journals.lww.com/topicsinmri/fulltext/2015/08000/the_knee.3.aspx),
[MRI Clinics: knee synovitis](https://www.mri.theclinics.com/article/S1064-9689(21)00732-7/pdf),
[BML assessment on fluid-sensitive sequences](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5112734/),
[MRNet, PLOS Medicine](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002699).
