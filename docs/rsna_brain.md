# RSNA Knee Abnormality Detection — Discussion Digest

**Scope:** Companion to `rsna_knee_rules_reference.md`. That file covers Overview/Data/Rules; this one covers the **Discussion tab**, which has real technical signal directly bearing on the label-blending pipeline, the gold-set overfitting risk, and the architecture bet.

**Status: complete.** All ~92 threads on the board were mapped. 38 threads were read in full — every thread judged substantive (labels, gold-set statistics, rules/licensing, DICOM/metadata/CV, modeling/architecture, data-quality bugs). The remaining ~50 threads are pure meta/logistics (team-finding posts, prize-payment questions, eligibility/age questions, individual notebook-error reports, "how do I download the data" — listed by ID at the end of §1) and were deliberately not opened; their titles are unambiguous and none bear on methodology.

**On verbatim quoting:** Forum posts can't be reproduced at length — copyright policy caps this at short attributed fragments, not paragraphs, even for personal reference use. What follows is paraphrased with precise numbers preserved (numbers/findings aren't copyrightable, phrasing is) — the same approach already used throughout `rsna_knee_rules_reference.md`.

---

## 0. TL;DR — what actually matters for your pipeline

| Finding | Why it matters |
|---|---|
| A competitor independently derived the **same Effusion→Synovitis relationship** you're using: P(synovitis\|effusion)=0.63, P(synovitis\|no effusion)=0.22 | 0.22 + 0.41 × effusion ≈ 0.63 — your Tier-2 formula's coefficients are the *same numbers* found independently (Disc 733932). |
| That same competitor **generalized** the correlated-column-fill approach to all twelve labels and it got *worse* overall (0.8873 → 0.8805) — Baker's, Contusion, Lateral OA all dropped, because for those findings report *silence itself* is highly diagnostic | Direct evidence against running your planned Baker's↔Effusion / Fracture↔Contusion proxy tests as currently scoped. §3.2. |
| **A rigorous statistical treatment of the 58-study gold set** (Disc 733876): paired σ ≈ 0.0125; a model truly 0.01 better wins only 78% of the time; reliable only past ~0.02 | The single most important thread for your flagged overfitting risk. Recommends ranking models on weak labels across all 4,407 reports, reserving the 58 for calibration only. §3.4. |
| **A second, independent pre-registered SOFT-vs-HARD replication** (Disc 734105) on a strong DINOv2 pipeline hit the same wall: SOFT beat HARD on all 3 seeds, but the 95% CI on the 58-study gain **crossed zero** | A second, methodologically serious confirmation this gold set can't cleanly resolve target-design choices below ~0.01–0.02 macro AUC. |
| **A checkpoint-normalization bug** cost one team an entire experiment: hardcoded ImageNet mean/std silently broke a medical-pretrained backbone (different, greyscale stats), producing a "plausible, wrong" result that looked like "medical pretraining doesn't transfer" (Disc 735154) | Direct, concrete check-item before wiring in OrthoDiffusion: read the checkpoint's own preprocessing config, never assume ImageNet constants. |
| **A live, unresolved tension in the external-data rules**: four separate participants asked whether the KneeCoT ruling (formal institutional agreement → disallowed) retroactively threatens MRNet/OAI-style datasets, which also require signed agreements. **None of the four have been answered.** | Complicates the "MRNet/OAI are green" reading already in your rules doc. Treat as an open risk, not settled. §3.6. |
| **Host directly confirmed** (Disc 733826): image-derived labels are authoritative over report text on disagreement; bilateral studies were individually reviewed and disambiguated; negative = annotated absent, not merely unannotated | Upgrades rules_reference.md §7.4 from 🔶 to ✅. Came with 8 concrete report-vs-gold discrepancy examples. |
| **Turkish negates *after* the term** ("efüzyon izlenmedi" = no effusion) — a standard left-scoped negation window silently inverts the second-largest language group in the corpus (Disc 734106) | Concrete, checkable bug in any regex/NegEx-style extraction step. |
| **Two training studies have corrupted pixel data** (byte count exactly half of what the DICOM header expects), confirmed by multiple participants and the RSNA data curator | `1.2.826.0.1.3680043.8.498.34685905030370793639196564723935583035` and `...37833587429731221455928642963031995680` — blacklist both. |
| A third public label table ("stevenleehans") benchmarks at **0.887–0.893 macro AUC vs. the 58 gold studies**, vs. **0.870 for `report_labels_v2`** (pilkwang); a fourth practitioner reported LLM label quality plateauing around **0.89–0.91** regardless of which LLM was used | Your current blend may be anchored on the weaker of several known public tables — worth a three-way comparison. |
| **OA compartment labels score *below chance* under naive keyword matching** (0.47 Lateral OA) because radiologists almost never write "osteoarthritis" — mining consequence-words (osteophytes, joint space narrowing, chondral loss, "tricompartmental") took it to 0.83/0.75 (Disc 734095) | Directly relevant to your "OA compartments" proxy-testing plan — the fix here is vocabulary, not correlation-based imputation. |
| **Community consensus across ~6 independent posts**: single well-tuned models match or beat 20+-model ensembles; DINOv2-Small vs. Base is a *null* at 224px (+0.0011 against a 0.0020 noise floor); one controlled test found DINOv2-Small *worse* than a ResNet34 control | Given your timeline, a real argument for weighting effort toward the label pipeline and image geometry over further backbone sophistication. |
| **Slice geometry beats resolution**: contiguous/adjacent slices around an anchor point outperform evenly-spread slices across the full stack by a wide margin in two independent ablations; filename-sort matches true anatomical order only **~5% of the time** on this corpus | Concrete, reusable preprocessing recipe (§2.19, §2.20) — sort by physical position, use adjacent-slice triplets, not evenly-spaced single slices. |
| Public LB is systematically **easier** than local OOF for multiple competitors, because LB scores against clean image-derived labels while local val typically uses noisy report-derived weak labels | Don't be alarmed if your LB beats your local weak-label validation — that asymmetry is expected and widely reported. |

---

## 1. Full thread index (92 topics, mapped 2026-08-28/29)

Legend: **[R]** = read in full. Category: RUL=rules/data-use, LBL=labels/gold-set, DAT=data/metadata/CV, MOD=modeling, MET=meta/logistics.

### Pinned
| ID | Title | Author | ↑ | 💬 | Cat |
|---|---|---|---|---|---|
| 733965 | Use of Commercially Hosted LLMs | Po-Hao "Howard" Chen (host) | 16 | 9 | RUL — already in rules_reference.md §7.1 |
| 733343 | Knee Abnormality Detection AI Challenge Overview | Po-Hao "Howard" Chen (host) | 62 | 8 | LBL — already in rules_reference.md §9 |
| 733375 | Welcome to Knee Abnormality Detection Challenge! | Po-Hao "Howard" Chen (host) | 15 | 1 | MET |
| 730709 | How to get started + Competition's Official Discord | María Cruz | 5 | 1 | MET |

### Labels / gold-set / report-extraction methodology — 16 read
| ID | Title | Author | ↑ | 💬 | |
|---|---|---|---|---|---|
| 733932 | "Not addressed" is a label too — 4,407 knee reports with an LLM | stevenleehans | 35 | 5 | **[R]** §2.1 |
| 737566 | Why everyone's Synovitis AUC is stuck around 0.6-0.7 | starkhushi | 2 | 0 | **[R]** §2.2 |
| 737454 | Benchmarked the public report-label tables against the 58 annotated studies | starkhushi | 0 | 0 | **[R]** §2.3 |
| 737155 | Annotation criteria question: how were Synovitis and Effusion graded? | Beyonder | 9 | 0 | **[R]** §2.4 — unanswered |
| 733826 | Possible inconsistencies between MRI reports and provided labels | Cho Royou (host replied) | 22 | 2 | **[R]** §2.5 |
| 734117 | Weak labels for all 12 findings + how recoverable each is | Luka Duvanov ("nekkon") | 1 | 0 | **[R]** §2.6 |
| 737650 | Unmentioned finding = negative, or unknown? | Alejandro Zorrilla Bejarano | 1 | 1 | **[R]** §2.13 |
| 734095 | Labeling the 4,349 report-only studies without an LLM | Busya PRIME | 0 | 0 | **[R]** §2.14 |
| 733864 | How the ground truth labels are labelled? | Dennis | 2 | 1 | **[R]** §2.15 |
| 733491 | Data/Reporting Inconsistencies | avg-HU (host replied) | 7 | 3 | **[R]** §2.16 |
| 737312 | Fluid_Sensitive and Fat_Suppression appear identical | RoshiBear (host replied) | 1 | 1 | **[R]** §2.17 |
| 734623 | Classifying labels into 12 abnormalities using Report | Malav D Modi | 1 | 2 | **[R]** §2.18 — low signal |
| 735596 | Looking for guidance on MRI report terminology | Kaustubh Ratna | 1 | 6 | **[R]** §2.18 |
| 733836 | Clarification on Training Data | agr hmmm | 3 | 4 | **[R]** §2.29 |
| 734055 | train.csv has 4,407 studies and 58 labels | maximo lorenzo y losada | 3 | 3 | **[R]** §2.9 |
| 734106 | 58 labelled studies out of 4,407 | Luka Duvanov | 6 | 3 | **[R]** §2.10 |

### Gold set size / overfitting risk — 5 read
| ID | Title | Author | ↑ | 💬 | |
|---|---|---|---|---|---|
| 737418 | Only 58 Gold Labels. | Jaideep726 | 0 | 2 | **[R]** §2.11 |
| 735855 | EDA: Only 58/4407 training studies have ground-truth labels | Istiyak Amin Santo | 1 | 1 | **[R]** §2.8 |
| 733876 | 58 studies cannot see a 0.01 gain - gold set enriched ~2x | Luka Duvanov ("nekkon") | 0 | 0 | **[R]** §2.7 |
| 734105 | Strong-pipeline replication: SOFT wins 3/3 seeds, 58-study CI crosses zero | FHZ982 | 2 | 0 | **[R]** §2.12 |
| 738172 | what is u label auc on gold? | Myo Min Htet(wnp) | 1 | 1 | **[R]** §2.11 |

### Rules / external data / licensing — 9 read
| ID | Title | Author | ↑ | 💬 | |
|---|---|---|---|---|---|
| 733652 | Rules clarification: external knee-MRI datasets + LLM API | Fernando Faria | 20 | 7 | already in rules_reference.md |
| 734109 | Is the gated KneeCoT dataset permitted as external data? | Mazzutti (host replied) | 8 | 2 | **[R]** §2.20 — **KneeCoT NOT allowed** |
| 733873 | Rules clarification: may Competition Data be sent to third-party LLM APIs? | FHZ982 (host replied) | 3 | 1 | **[R]** §2.19 — precursor to 733965 |
| 737950 | External data behind an institutional DUA — KneeCoT ruling generalize? | Ercan Gurvit | 2 | 0 | **[R]** §2.21 — unanswered |
| 735497 | Rule 2.6(a) and registration-gated public datasets (MRNet, OAI) | Demir Poyraz Elcin | 5 | 0 | **[R]** §2.21 — unanswered |
| 738156 | External data: ungated, openly-licensed OAI-derivative datasets | Yet, WiaLive! | 1 | 1 | **[R]** §2.21 — unanswered |
| 738111 | External data rule — free but research-use agreements | pizzaboy | 1 | 0 | **[R]** §2.21 — unanswered |
| 735121 | CC-BY-NC pretrained weights compatible with winners open-licence? | dk2lone | 6 | 0 | **[R]** §2.22 — unanswered |
| 734131 | Clarification on MIRA Section 6 | hangglider5 | 0 | 0 | **[R]** §2.23 — unanswered |

### Data / metadata / CV strategy — 6 read
| ID | Title | Author | ↑ | 💬 | |
|---|---|---|---|---|---|
| 733517 | 0.932 LB within one day. DICOM metadata shortcut | Oleksii Zhukov | 17 | 2 | already in rules_reference.md §13.5 |
| 734004 | DICOM metadata findings: scanner-grouped CV and PatientSex priors | morningduck | 5 | 0 | **[R]** §2.24 |
| 734681 | Public/private test split — stratified by site? | Matteo Vitali | 5 | 3 | **[R]** §2.25 — unanswered |
| 734118 | reports will be unavailable for the hidden test set? | Nicolas Pantoja (host replied) | 0 | 1 | **[R]** §2.26 |
| 733423 | train.csv: PatientSex documented but not present | epicfangs (host replied) | 1 | 3 | **[R]** §2.27 |
| 735639 | Two knees are better than one? | Tim Krige | 8 | 4 | **[R]** §2.28 |

### Data quality bugs — 2 read
| ID | Title | Author | ↑ | 💬 | |
|---|---|---|---|---|---|
| 738708 | Corrupted DICOM Pixel Data Found in 2 Training Studies | Tahmid Tawsif | 0 | 0 | **[R]** §2.30 |
| 737163 | corrupt files | Yee Ng (host/curator replied) | 6 | 4 | **[R]** §2.30 — same two studies confirmed |

### Modeling / architecture — 13 read
| ID | Title | Author | ↑ | 💬 | |
|---|---|---|---|---|---|
| 735304 | Best single-model score | roy214 | 21 | 37 | **[R]** §2.31 |
| 735826 | What is the real bottleneck now: labels, image pipeline, or model capacity? | Oscar Yáñez Feijóo | 9 | 11 | **[R]** §2.32 |
| 735767 | What Could the Final Ceiling Be for RSNA 2026? | Tony Li | 13 | 6 | **[R]** §2.33 |
| 738339 | Stuck around 0.74 OOF - what separates the 0.87 tier from 0.93+? | Abdullah Wasee | 0 | 2 | **[R]** §2.34 |
| 737597 | What we've ruled out at 0.79 OOF — and what's left | Berat Kirbiyik | 4 | 0 | **[R]** §2.35 |
| 735154 | Scaling the encoder bought us nothing (+0.0011) | stevenleehans | 11 | 3 | **[R]** §2.36 |
| 737696 | RSNA Raptor Weights | Dread Development | 9 | 5 | **[R]** §2.37 |
| 733313 | Using Dino3 | Ryuhki Kimura | 9 | 15 | **[R]** §2.38 |
| 738096 | Dinov2 Base vs Small at 224 resolution | Komil Parmar | 5 | 2 | **[R]** §2.39 |
| 736999 | YOLO / ROI cropping for the knee joint? | Mehmet Özer | 3 | 0 | **[R]** §2.40 — unanswered |
| 738495 | GRU/LSTM head over slices? | Abdullah Wasee | 0 | 0 | **[R]** §2.41 — unanswered |
| 736301 | Are you actually looking at the data? | Chikuwabu | 11 | 2 | **[R]** §2.42 |
| 736635 | LB OOF relationship | Nicolai Karcher | 4 | 3 | **[R]** §2.9 (folded into gold-set section) |
| 734357 | YOLO used to be useful in such competition | Handudu | 3 | 1 | **[R]** §2.40 — low signal |

### Meta / logistics / eligibility / individual bug reports — reviewed by title, not opened
733497, 735348, 735323, 736348, 736098, 736457, 734676, 736498, 736497, 735599, 735854, 734671, 735296, 735295, 735017, 733971, 734062, 733797, 733934, 733478, 733592, 733753, 733475, 736170, 737449, 738036, 737350, 736678, 737333, 736720, 736711, 737018, 736268, 738129, 737820, 737635.

---

## 2. Deep dives — threads read in full

### 2.1 Disc 733932 — "Not addressed" is a label too (stevenleehans, 35↑, 5 comments)

The most load-bearing thread on the board. Posted by the author of one of the public LLM-derived label tables (credits Pilkwang Kim's `rsna-knee-llm-labels` as the first, `barun2104`'s stratified-folds soft labels, and `lixin73`'s "LLM Report Labels (GPT-5.6-Sol)" — 🔶 plausibly the source `report_labels_gpt56sol.csv` traces to, given the name match, not directly confirmed).

- Regex/lexicon extraction scores **0.8136** macro AUC vs. the 58 gold studies; an LLM reading the same reports scores **0.8780**.
- **25.4% of all label cells** came back "not addressed," unevenly: Synovitis 83.7%, Baker's 48.2%, Fracture 42.9%, ACL 8.3%, Medial Meniscus 5.5%. Per-finding gold AUC tracks this — Synovitis 0.678 (worst), Baker's 0.946, ACL 0.993.
- Despite 84% "not addressed," **27 of the 58 gold studies have synovitis** — common but rarely written down.
- **Pre-registered test:** LLM Effusion field predicts gold Synovitis better (0.7115) than the Synovitis field itself (0.6780). Co-occurrence: **P(synovitis\|effusion) = 0.63, P(synovitis\|no effusion) = 0.22.** Filling only undecided Synovitis cells from Effusion moved that column 0.678→0.790, overall key 0.8780→0.8873.
- **Generalized to all twelve findings** (ridge model per finding, fit on the full corpus, gold used only to score once): worse overall, 0.8805 vs. the Synovitis-only 0.8873. Split: Synovitis +0.056, PF OA +0.023, Fracture +0.015 — but Baker's −0.029, Contusion −0.031, Lateral OA −0.019.
- **The mechanism:** gold-positive rate conditional on report silence vs. speaking, per finding — Synovitis 0.34 vs 0.76, PF OA 0.21 vs 0.41, Baker's **0.03 vs 0.44**, Medial OA 0.00 vs 0.36. Where silence carries almost no information (Synovitis), imputing helps. Where silence is nearly as informative as an explicit negative (Baker's, Medial OA), imputing destroys signal. "Silence ratio" correlates **+0.59** with how much imputation helped, across nine measurable findings (their own caveat: suggestive, not proven).
- Honest limits: differences under **~0.02 macro AUC not measurable** on 58 studies; **three separate readings overturned by the leaderboard**; a gold-informed version of the imputation selection still only hit 0.8845, below the disciplined blind version's 0.8873; gold prevalence is the annotation sample's prevalence, not disease prevalence — **every gold study has ≥1 positive finding, mean 4.14 positives/study**.

### 2.2 Disc 737566 — Why everyone's Synovitis AUC is stuck around 0.6-0.7 (starkhushi, 0 comments)

Independent confirmation: only **~16%** of the 4,407 reports mention any synovitis term across nine languages searched. Report-derived labels agree with gold at **~0.79 AUC on Synovitis vs. ~0.99 on ACL**. A RadImageNet-pretrained encoder scored **0.78 on Synovitis** vs. **0.62–0.72** for otherwise-identical ImageNet-pretrained models — the only architecture change that moved this specific label for them. Independently flags Effusion↔Synovitis correlation (~0.5 in annotated studies) as underexploited.

### 2.3 Disc 737454 — Benchmarked the public report-label tables against the 58 annotated studies (starkhushi, 0 comments)

Direct macro-AUC-vs-gold comparison of public label tables:

| Label source | Macro AUC vs. 58 gold |
|---|---|
| stevenleehans v4 (blend) | 0.893 |
| stevenleehans v2 | 0.887 |
| pilkwang `report_labels_v2` | 0.870 |

Averaging two independent tables beat either alone on several labels. Notebook (not opened): `kaggle.com/code/starkhushi/rsna-knee-which-report-labels-should-you-train-on`. Your `report_labels_v2` component is the lowest-scoring of the three tables benchmarked here — doesn't mean your blend underperforms (averaging can beat either input), but worth knowing.

### 2.4 Disc 737155 — Annotation criteria question: Synovitis and Effusion grading (Beyonder, 0 comments)

Open, unanswered: whether Synovitis was scored via effusion-related surrogates (MOAKS-style) or only explicit synovial thickening, and whether Effusion has a size threshold. **Zero replies.** Your Tier-2 rule's underlying relationship remains empirically strong but not officially confirmed as the annotation logic.

### 2.5 Disc 733826 — Possible inconsistencies between MRI reports and provided labels (Cho Royou, 22↑, host replied)

A manual audit of 20 of the 58 gold studies against strict report-only labeling rules found **82.5% overall agreement** (positive predictive agreement 73.1%, recall 80.0%; TP 68, FP 25, FN 17, TN 130 across 240 decisions). Included concrete discrepancy examples: a report stating synovitis and massive effusion where gold recorded Effusion=1/Synovitis=0; a Spanish report mentioning mild effusion with minimal synovitis where gold recorded both as 0; a German report explicitly naming a Baker's cyst (with a measured size, "up to 2 x 1.5 x 3.5 cm" — per a related question in Disc 733864 quoting the same study) where gold recorded Baker's=0; a report stating both medial and lateral compartment OA plus PF OA where gold recorded only PF OA (+Effusion/Synovitis) positive; a Turkish report stating the lateral meniscus was normal where gold recorded a lateral meniscus tear as positive; a Bulgarian report describing an osteochondral fracture where gold recorded Fracture=0 (poser's own read: likely a deliberate distinction between osteochondral injury and the competition's "acute fracture" definition, not an error).

**Host reply, directly relevant:**
- Labels were assigned **independently from the images**, not extracted from the reports — confirmed.
- When image and report disagree, the **image-derived label is authoritative** — confirmed. Only a small sample of provided data contains both report and gold label, specifically to let participants discover this.
- A negative label means the finding was **annotated as absent**, not merely unannotated (this describes the *gold* label semantics — separate from the "not addressed" category that applies to *report-derived* weak labels).
- Bilateral studies/reports were **individually reviewed and the report text or DICOM metadata adjusted** so participants can disambiguate — confirmed.
- Host's own explanation for the systematic gap: clinical reports have **one signing radiologist writing for clinical care**, while gold labels use **multiple readers with stricter image-based thresholds** — matches the "systematic monotone shift" framing already in your working notes almost exactly.
- Host referenced "a related discussion" outlining the design principles behind the perceived inconsistencies — link not captured this session, most likely the pinned Overview post (733343) already known.

### 2.6 Disc 734117 — Weak labels for all 12 findings + recoverability (Luka Duvanov / "nekkon", 0 comments)

Companion notebook extracting all 12 findings from all 4,407 reports (`weak_labels.csv`). Per-finding balanced accuracy vs. the 58 gold ranges **0.82 (Baker's cyst) to 0.56 (Medial Meniscus)**, and the pattern tracks the *kind* of finding: named objects (Baker's, ACL) extract well; graded severities (Effusion) fail because a binary keyword match has nowhere to encode "minimal" vs. "trace" vs. "moderate" — that's a modeling problem, not a vocabulary one; unstated inferences (Fracture) fail hardest — 0.93 specificity but only **0.44 sensitivity**, because more than half of gold-positive fractures are described by appearance without the word "fracture" ever appearing. The pure lexicon extractor finds **2.6 findings per study** where annotators recorded **4.1** (consistent with §2.1's 4.14 figure) and returns nothing for 23% of studies — treat those as unlabeled, not all-negative.

### 2.7 Disc 733876 — 58 studies cannot see a 0.01 gain, and the gold set is enriched ~2x (Luka Duvanov / "nekkon", 0 comments)

The most statistically rigorous thread on the board. From 20,000 simulated head-to-heads on the real gold labels with correlated errors modeled the way two models of one task actually correlate:

- Paired σ of a macro-AUC comparison on 58 studies ≈ **0.0125**.
- A model truly 0.01 better wins the comparison ~78% of the time; 0.005 better wins ~66%; only past **~0.02** does it become reliable (~94%).
- Correlation structure matters a lot: at ρ=0.98 (two seeds of one architecture) σ halves to 0.006 and the true winner reliably emerges — **ablations are measurable on 58 studies**. At ρ=0 (e.g. a CNN vs. a report-only model) σ doubles to 0.025 and even a 0.02 gap only wins ~79% — **architecture-family comparisons are not** reliably measurable here.
- Macro-averaging gives MCL (9 positives) the same 1/12 weight as Effusion (35 positives), but MCL contributes 14.4% of the total variance; the three rarest findings are 25% of the weight and 37.6% of the noise.
- **Enrichment:** running one fixed report extractor over both the 58 and the full 4,407 (so the extractor's own bias cancels out), it fires **3.1× more often for Fracture** on the 58 than on the full corpus; median enrichment across the twelve findings is **1.53×**, and 11 of 12 findings are enriched in the same direction. Priors/thresholds/class weights fit on the 58 are calibrated for a sicker population than the real training/test distribution, and AUC measured on an enriched sample skews optimistic.
- **Practical recommendation the author has adopted:** rank models on weak labels over all 4,407 reports; keep the 58 gold studies for calibration only, not model selection.

### 2.8 Disc 735855 — EDA: only 58/4,407 have ground truth (Istiyak Amin Santo, 1 comment)

Full label prevalence table among the 58 gold studies — **Effusion 60%, Synovitis 47%, Medial Meniscus 45%, ACL 41%, Lateral Meniscus 40%, PF OA 36%, Contusion 33%, Fracture 31%, Medial OA 26%, Baker's 21%, Lateral OA 19%, MCL 16%** (sums to ~4.15 expected positives/study — matches §2.1's 4.14 mean almost exactly, good cross-validation). Confirms series structure: median 5 series/study (range 3–14). Confirms Fluid_Sensitive/Fat_Suppression correlation, and that only 6 distinct (plane, sequence-type) combinations occur in the data despite the description's hedge. DICOM image dimensions vary a lot even within one study (640×640 up to 960×960, at least one non-square 640×1280 spotted) — don't assume a fixed input shape.

### 2.9 Disc 734055 & 736635 — Parsing traps, multi-script analysis, and LB-vs-OOF (maximo lorenzo y losada / Nicolai Karcher)

**734055** confirms the labeling is all-or-nothing (58 studies have all 12 columns, the other 4,349 have none — no partial labeling) and flags two parsing traps: (1) the CSV is 58,556 *lines* but only 4,407 *rows* — reports contain embedded newlines, so `wc -l` overstates the dataset ~13×, use a real CSV reader; (2) empty cells are not zero — filling them with 0 invents ~4,349 unasserted negatives per class and produces a nonsense 0.5% prevalence. Independently reproduces the prevalence table from §2.8 to the decimal (Effusion 60.3% … MCL 15.5%, mean 4.1/study, max 9).

Multi-script correction (same thread, follow-up post): a stopword-keyword language heuristic was replaced with Unicode-script counting. Over all 4,407 reports (a report can contain more than one script): **Latin 4,301 (97.6%, later corrected to 4,266 on a stricter codepoint range), Greek 321 (7.3%), Cyrillic 220 (5.0%)** — the Greek is genuine prose (321/321 contain ≥3 consecutive Greek letters), not stray symbols. Flags a real encoding gotcha: U+03BC (Greek small mu) vs. U+00B5 (micro sign) render identically but don't string-match.

**The important follow-up test:** does a text pipeline built on Latin-script terms silently fail (extract nothing → confident negative) on the non-Latin reports? Tested against stevenleehans' labeler (§2.1) using its explicit "not addressed" refusal signal: refusal rate on Latin-only reports 25.9% vs. 21.9% on reports containing Greek/Cyrillic (541 studies, 12.3% of corpus) — a gap of 4.0 points, permutation-tested at p<0.0005. **So the labeler does not appear to fail silently on non-Latin script overall** — but per-finding, **Synovitis (+8.5 points refusal, 82.7%→91.1%) and Baker's (+7.8 points, 47.3%→55.1%) move the opposite direction**, i.e. refused *more* on non-Latin reports. Those are exactly the two columns §2.1's silence-ratio logic leans on hardest — worth treating the silence-ratio numbers there as possibly confounded by script/language mix specifically for those two labels. Only 6 of the 58 gold studies contain non-Latin script, so accuracy (as opposed to refusal rate) on the multilingual subset can't currently be validated.

**736635** (LB OOF relationship): multiple competitors independently report that public LB is systematically *easier* than local OOF, especially at the low end of macro-AUC, with a nonlinear-looking relationship. The most-voted explanation: local validation typically uses noisy report-derived weak labels while the LB scores against clean image-derived gold labels, so the asymmetry is expected rather than a leakage red flag. One competitor with only 3 self-trained models reports the opposite pattern for them specifically — their gold-set validation tracks LB better than their weak-label validation does, because their weak labels are the noisier signal in their particular pipeline.

### 2.10 Disc 734106 — 58 labelled studies out of 4,407 (Luka Duvanov, 3 comments, 1 appreciation)

Ground-truth facts, independently useful even where they overlap §2.8/§2.9: labels present in seven languages — English, Turkish, Spanish, German, Greek (178 reports in Greek script — note this doesn't match §2.9's 321-study figure exactly, likely a different counting threshold; treat both as approximate), Dutch, French — plus 428 reports a stopword detector can't place. **Turkish negates *after* the term** — "efüzyon izlenmedi" means NO effusion — so a left-only negation window (the default in most English NegEx ports) silently inverts the second-largest language group in the corpus. Confirms Fluid_Sensitive/Fat_Suppression identical on all 24,371 series (matches §2.17's host-confirmed exact count). Confirms every study has all three planes and both contrast types — no fallback path needed. An untuned multilingual keyword matcher (vocabulary not fit on the 58) scores 0.86 on Baker's cyst, 0.66 on Effusion — offered as "the floor to beat." Confirms enrichment directly: **ACL prevalence 41% in the 58 vs. ~20% in the full corpus** — a ~2× figure that matches the thread title of §2.7.

Follow-up comment from the same author gives a concrete, reusable extraction method: don't machine-translate the corpus (O(4,407 reports) through a seq2seq model); translate the *vocabulary* instead (O(40–60 terms/finding × 7 languages) — a few hundred strings translated once, then plain string matching, seconds on CPU). Detect language first via stopword frequency (free), then apply that language's lexicon and negation direction (right-scoped for Turkish, left-scoped for the Germanic/Romance languages). If going the embedding route instead of string matching, suggests multilingual sentence embeddings (LaBSE, multilingual-e5) as comparable across all seven languages in one forward pass, rather than translating first.

### 2.11 Disc 737418 & 738172 — practical calibration points on label/LB numbers

**737418** ("Only 58 Gold Labels"): one competitor reports LLM label-extraction performance plateauing around **0.89 macro AUC vs. gold regardless of which LLM they tried** — consistent with the 0.887–0.893 range independently found in §2.3, suggesting ~0.89 may be close to a practical ceiling for this style of extraction. Their own read: gold labels themselves aren't perfectly reliable either (different radiologists, different experience levels), so chasing metric gains past this point may just be overfitting to the ruler.

**738172**: one competitor reports label AUC on gold = 0.9083 (the highest such figure seen in this pass) against an LB score of 0.924 — their trained image model modestly *exceeds* their own label extractor's accuracy, which by Tucker Arrants' diagnostic (§2.32) suggests they're past the point where labels are the binding constraint, though not by a wide margin.

### 2.12 Disc 734105 — Strong-pipeline replication: SOFT wins 3/3 seeds, 58-study CI crosses zero (FHZ982, 0 comments)

A genuinely pre-registered replication comparing HARD (binary: affirmed=1, else 0) vs. SOFT (graded: affirmed=1, hedged=0.3–0.8, negated=0.04–0.16, unmentioned=0.28) report-derived targets. Setup: DINOv2 backbone (12 blocks, last 6 trainable), three paired seeds, checkpoint selection separated from evaluation, 58 gold studies held fully out of training/selection, an independent 430-study holdout (`va_ev`) scored only against the binarized lexicon surrogate.

- On the 58 gold studies: SOFT beat HARD in **all 3 paired seeds** (+0.0104, +0.0083, +0.0079). Rank-mean ensemble: SOFT 0.8071 vs HARD 0.7930, difference **+0.0143**, 95% paired bootstrap interval **[-0.0041, +0.0330]** — crosses zero, doesn't establish significance, but also doesn't fit inside their pre-specified ±0.01 equivalence margin, so not evidence of "no difference" either. Their own framing: the interval crossing zero means *unresolved*, not *interchangeable*.
- On the independent 430-study `va_ev` set (scored against the same binarized lexicon HARD was trained to reproduce): **HARD wins**, -0.0132 for SOFT, 95% CI [-0.0207, -0.0059] — excludes zero. Their own caveat: this endpoint measures "does the model reproduce the lexicon," circular in HARD's favor, not "does the model match the radiologist."
- Their proposed mechanism (SOFT should help most where binarization destroys the most target signal) was pre-registered and **failed**: Spearman ρ = -0.161 between destroyed-signal and per-finding SOFT improvement — the earlier version of this experiment's causal story was withdrawn.
- Per-finding SOFT-HARD gap on the 58 gold: ACL +0.081, Fracture +0.044, Effusion +0.039, Synovitis +0.030, PF OA +0.017, Lateral Meniscus +0.009 — but Lateral OA -0.044, Baker's -0.006, Contusion -0.005, Medial OA -0.003.
- **Four preprocessing findings, independent of the SOFT/HARD question:**
  1. Don't sort `.dcm` files by filename (SOPInstanceUID is unique, not ordered) — sort by physical position derived from `ImageOrientationPatient` × `ImagePositionPatient`.
  2. A fixed-mm physical crop before resize matters — they use a 130mm field at 336px (~0.387mm/px); verify the requested crop actually fits the volume, or it silently becomes a no-op.
  3. Laterality should be derived from image-center geometry in patient space, not a corner coordinate; coronal/axial need horizontal normalization, sagittal stacks need the medial-lateral stack-axis order normalized.
  4. Don't stack multiple slices as input channels and average the pretrained first-conv weights — it destroys the encoder's learned input interface. Feed 3-slice groups through the untouched backbone and aggregate feature vectors afterward.
- Corpus facts: **4,407 studies, 4,276 unique report texts** (131 duplicates, grouped before splitting); Turkish case-folding needs explicit dotted/dotless-i handling; labels here came from a deterministic multilingual lexicon, not an LLM.

### 2.13 Disc 737650 — Unmentioned finding = negative, or unknown? (Alejandro Zorrilla Bejarano, 1 comment)

Open design question about the closed-world assumption for weak labels, answered by a self-identified radiologist in the comments: for Fracture, meniscal, or ligamentous tears, not mentioning a finding in a report genuinely does mean "I didn't see it" — radiologists report what they see. **Osteoarthritis is the exception** — it's frequently described *without* the word "osteoarthritis" via synonyms (cartilage loss, chondromalacia, cartilage fissuring, cartilage thinning, osteophytes), so silence there is much less reliable as a negative signal unless the extraction vocabulary explicitly covers those terms. Directly corroborates §2.14's finding.

### 2.14 Disc 734095 — Labeling the 4,349 report-only studies without an LLM (Busya PRIME, 0 comments)

A zero-cost, zero-GPU, deterministic rule-based labeler (chosen specifically to sidestep the "reasonably accessible" ambiguity around paid LLM APIs) reads each report in its own detected language (English, Spanish, German, Dutch, French, Greek, Turkish — at least eight detected), fires a per-finding vocabulary, and applies negation/uncertainty scoping so "no meniscal tear," "sin rotura," "geen scheur," "kein" all correctly resolve to negative.

**The single biggest lever they found:** Osteoarthritis is almost never written as the word "osteoarthritis." Naive keyword reading scores **Lateral OA 0.47 (below chance!) and Medial OA 0.59** — essentially no usable signal. Mining the *consequences* radiologists actually write instead — osteophytes, joint space narrowing, chondral loss, chondrosis, "gonarthrose" (German), plus a rule that "tricompartmental" fires all three OA labels — took **Lateral OA to 0.83 and Medial OA to 0.75**.

Full ablation on the 58 gold (macro AUC over all 12): naive keyword presence 0.638 → add sentence-scope negation 0.667 → add the OA consequence vocabulary 0.727. Negation mostly buys precision, not ranking (Fracture precision 0.53→0.80, Baker's 0.42→0.62). Per-label range: Synovitis 0.61 (hardest — can't cleanly separate from simple effusion on non-contrast MRI), Effusion 0.63 (surprisingly low despite being the most common finding), up to ACL and Lateral OA at 0.83. Macro 0.727, bootstrap interval ~0.67–0.78. Honest caveat: the vocabulary was refined by reading the 58 gold reports directly, so 0.727 is in-sample and probably optimistic; a good LLM will likely beat it on the subtle classes.

### 2.15 Disc 733864 — How the ground truth labels are labelled? (Dennis, 1 comment)

Concrete illustration of the report-vs-gold gap, walking through one specific study whose report explicitly describes "a Baker's cyst... measuring up to 2 x 1.5 x [CC] (3.5 cm)" alongside other findings, where the gold label for Baker's is 0. Even an explicit measurement in the report doesn't guarantee a positive gold label — the "moderate or large" threshold judgment happens at the image level, independent of what the report states or measures. Only comment redirects to the host's answer in §2.5 (733826), which is the authoritative source.

### 2.16 Disc 733491 — Data/Reporting Inconsistencies (avg-HU, host replied) — the source of your original example quote

This is the exact thread your original example passage (about deidentified reports, ambiguous wording, the two-reader-plus-adjudicator process, and the meniscal-tear definition) was drawn from — confirmed by an exact text match. The poster also flagged that the underlying *contributor reports themselves* can be internally self-contradictory independent of the report-vs-gold gap: one report describes bone marrow edema plus a meniscal tear alongside "no acute fracture or bone bruise" and normal cartilage in the same dictation; another describes a chondral lesion in the lateral patellar facet (anatomically patellofemoral) but the impression lists "lateral compartment chondromalacia"; a third contains an internally contradictory statement about whether the meniscus is normal or torn. **Host's reply (paraphrased, matching your pasted example closely):** reports are deidentified contributor originals that may contain ambiguous or internally inconsistent wording not mapping cleanly to the binary targets — a realistic reflection of routine clinical reporting; sample labels come from image review by two independent readers with a third adjudicating; for meniscal tears specifically, the target is a *definite* tear, with intrasubstance degenerative signal not reaching the articular surface graded negative — the same process was used for the test set; marrow edema, cartilage findings, and narrative terminology don't by themselves determine the contusion/OA/other labels, which follow the standardized image-review rubric independent of report wording.

### 2.17 Disc 737312 — Fluid_Sensitive and Fat_Suppression appear identical (RoshiBear, host replied)

Host-confirmed: across all **24,371 series** in `train_series.csv`, Fluid_Sensitive and Fat_Suppression are identical in every case (10,361 series both=0, 14,010 series both=1, zero series differ). The host gave a detailed clinical explanation of why they usually co-occur in practice (fat suppression makes fluid-related abnormalities easier to see, so most clinically useful fluid-sensitive MSK sequences happen to also be fat-suppressed) but explicitly warned this equality **should not be assumed to hold in the test set or any other dataset** — the two remain conceptually distinct imaging characteristics. Directly upgrades the existing 🔶 caution in rules_reference.md §8.2 to a host-confirmed ⚠️ with the exact counts.

### 2.18 Disc 734623 & 735596 — general beginner Q&A on report→label extraction

Low-signal beginner threads. **734623** (Malav D Modi): a generic "how do I turn reports into labels" question; the one substantive reply frames the competition as fundamentally two parts — get the labels right, then get the model right — consistent with the broader thread consensus elsewhere in this digest. **735596** (Kaustubh Ratna): a request for report vocabulary guidance drew mostly meta-commentary rather than a vocabulary list; the one durable point (echoed by a self-described 45-year veteran of human classification-judgment work, PC Jimmmy) is that with ~4,700 different report-writing radiologists worldwide contributing to the corpus against a small, disciplined annotation team for the gold set, both random and systematic disagreement should be expected as a baseline, not treated as noise to be engineered away entirely. A general MRI-terminology reference was shared: `radiologymasterclass.co.uk/tutorials/mri/mri_system`.

### 2.19 Disc 733873 — Rules clarification: may Competition Data be sent to third-party LLM APIs? (FHZ982, host replied)

A precursor question thread to the pinned 733965 ruling — laid out the same tension between Rule 2.6 (external tools generally acceptable) and Rule 2.4.b / MIRA §4-5 (no redistribution/sharing of Competition Data) that 733965 goes on to resolve. Host's only reply here: acknowledges the question is worth settling clearly and points to the newly-posted 733965 as the answer. No independent new information beyond what's already captured from 733965 in `rsna_knee_rules_reference.md` §7.1.

### 2.20 Disc 734109 — Is the gated KneeCoT dataset permitted as external data? (Mazzutti, 8↑, host replied)

**New official ruling, not yet in rules_reference.md.** KneeCoT (`huggingface.co/datasets/YiHui0124/KneeCoT`, gated, CC BY-NC 4.0, requires a formal ethical-use agreement with the affiliated hospital) was ruled **not permitted**. Host's stated reasoning: requiring a formal institutional data-use agreement creates an uneven playing field between participants who can and can't complete it — the host's own interpretation of the accessibility rule, not a blanket ban on all gated datasets. The host explicitly left room to revisit if HuggingFace's access policy or the dataset organizers' stated intent changes.

### 2.21 The unresolved external-data-boundary cluster (Disc 737950, 735497, 738156, 738111 — all unanswered as of this read)

Four separate participants asked the host, in the days following the KneeCoT ruling (§2.20), whether that ruling's logic generalizes to other gated-but-free knee-MRI datasets — **none have received a reply yet**:

- **737950** (Ercan Gurvit): does the KneeCoT reasoning apply to OAI (requires a signed Data Use Certification from an institution with an active Federal Wide Assurance) or MRNet (requires a signed Research Use Agreement from Stanford)? Also asks whether publishing weights pretrained on such data would satisfy the winners' public-weights obligation, or whether the obligation independently rules out that pretraining regardless of license.
- **735497** (Demir Poyraz Elcin): explicitly notes "at least two earlier participant questions on this point appear to be unanswered" — a formal request for a written ruling on whether registration-gated datasets (MRNet, OAI) satisfy Rule 2.6(a)'s "equally accessible at no cost" language.
- **738156** (Yet, WiaLive!): a sharper version of the same question — what about datasets *derived* from a gated source (e.g. OAIZIB-CM on HuggingFace: ungated, CC BY-NC 4.0, re-hosts 507 OAI DESS exams with segmentation masks) but re-released without any gate? Does the accessibility test apply to the dataset actually being downloaded, or to the provenance of its source images? A commenter's aside is worth noting: they suspect KneeCoT "would have been allowed if someone had not asked" — a reminder that raising a question can sometimes produce a stricter ruling than silence would have.
- **738111** (pizzaboy): the same question for a different category — datasets like the Stanford AIMI releases, free to access but behind a signed research-use agreement carrying non-commercial and no-derivative-works terms.

**This directly complicates the "MRNet/OAI/fastMRI+/SKM-TEA are green" reading already in `rsna_knee_rules_reference.md` §7.2**, which was based on the earlier 733652 ruling (simple registration/click-through is fine; formal institution-specific agreements are the risk boundary). The KneeCoT ruling's stated reasoning — a *formal agreement* creates an uneven playing field — could plausibly extend to OAI/MRNet's own signed agreements, and the host hasn't yet said whether it does or doesn't. **Treat MRNet/OAI-style pretraining as an open risk, not a settled green light**, until one of these four threads gets a reply.

### 2.22 Disc 735121 — Are CC-BY-NC pretrained weights compatible with the winners open-licence obligation? (dk2lone, 0 comments, unanswered)

Asks whether a backbone under CC-BY-NC-SA-4.0 (the RadImageNet ResNet-50 checkpoint used in several public notebooks is the concrete example) can satisfy the winners' obligation to release under CC-BY-NC 4.0, given non-commercial/share-alike terms may not be compatible with an open release. No host answer yet. Not directly a concern for your OrthoDiffusion choice specifically (MIT is unambiguously compatible with everything), but worth knowing rules_reference.md §10.1 Carve-out B — "input data or pretrained models with an incompatible license → you do not need to grant an open-source license for that data/model" — likely already resolves this favorably even without an explicit host confirmation on this exact thread.

### 2.23 Disc 734131 — Clarification on MIRA Section 6 (hangglider5, 0 comments, unanswered)

Asks whether MIRA §6's prohibition on modification/derivative works applies to ordinary competition preprocessing and derived training artifacts. No host reply. Restates the same tension already flagged in rules_reference.md §6.4's "Tension note" — unresolved on the forum as well as in the written rules.

### 2.24 Disc 734004 — DICOM metadata findings: scanner-grouped CV and PatientSex priors (morningduck, 0 comments)

- ~13 distinct Manufacturer×MagneticFieldStrength scanner groups across the 4,407 training studies (Siemens_1.5T 1,148; Siemens_3T 781; GE_1.5T 698; Philips_3T 663; Philips_1.5T 619, plus smaller groups). **Note the granularity gap**: this is coarser than the ~265 "distinct scanner fingerprints" 🔶 figure already in rules_reference.md §13.5 (likely station-name/serial-level, not just manufacturer+field-strength) — worth confirming which granularity your own fold key uses.
- A metadata-only (no-pixel) classifier: **random 5-fold macro-AUC 0.652 vs. scanner-grouped 0.598**, a ~0.05 gap — independently corroborates the leak-probe numbers already in rules_reference.md (0.6515 / 0.5981) to within 0.001, from a different author using a different methodology. OA targets showed the largest drop under grouped folds (0.07–0.09), consistent with field-strength-dependent cartilage contrast.
- `PatientSex` (tag 0010,0040) **is present in test DICOM headers**. Training distribution M=2,076/F=1,894. Target prevalence by sex: ACL ~54% M vs ~32% F; Medial OA ~12% M vs ~45% F — males skew toward traumatic findings, females toward degenerative, consistent with general orthopedic epidemiology and directly corroborated by the host's own explanation in §2.27. Treat as a clinically-grounded relative likelihood, not a training-set base rate to bake in directly.
- Manufacturer, MagneticFieldStrength, SeriesDescription, and ImageOrientationPatient are all readable via pydicom at inference time — usable for calibration/ensembling without violating the no-report-at-test constraint.

### 2.25 Disc 734681 — Public/private test split — stratified by site? (Matteo Vitali, 3 comments, unanswered by host)

Asks whether the public/private split is stratified across all sites (~16 sites mentioned, 🔶 not independently verified against the official contributor list) or whether entire sites are held out for the private set — a binary question the host has not answered. One commenter, examining six past RSNA competitions, found leaderboard shakeup ranging from none to 1,039 places for 1st place, suggesting some RSNA competitions do use non-random splits. A second commenter makes the strategically important point directly: **the answer determines whether your scanner-fingerprint-grouped CV is measuring the right thing.** If private test holds out entire unseen sites, grouped folds are the correct design and the OOF cost is worth paying. If both splits sample from the same 16 sites, grouped folds may be paying a cost for a distribution shift the private test doesn't actually contain, and a random-fold OOF would be the cheaper, more accurate guide. **This is unresolved** — worth treating your grouped-fold decision as a reasonable hedge (robust either way) rather than a confirmed match to the actual test design.

### 2.26 Disc 734118 — reports will be unavailable for the hidden test set? (Nicolas Pantoja, host replied)

Direct host confirmation, single line: reports are not available for the hidden test set. Reaffirms what's already established elsewhere (rules_reference.md §8.1, §13.1) with no new detail.

### 2.27 Disc 733423 — train.csv: PatientSex documented but not present (epicfangs, host replied)

Host confirmed PatientSex was originally built into the training CSV but deliberately removed before release, since it's already available directly in each study's DICOM header and was judged redundant to duplicate. The data page was to be updated to reflect this. Upgrades the PatientSex-availability fact in §2.24 from forum-observed to host-confirmed.

### 2.28 Disc 735639 — Two knees are better than one? (Tim Krige, 4 comments)

Practical image-side edge case: roughly 7 training studies contain **both knees in the same DICOM series**. A naive fixed crop tuned for a single knee can miss the joint of interest entirely — one competitor discovered their crop was missing the bone altogether, which spuriously triggered a "fracture" flag from their pipeline. Worth an explicit sanity check that any crop/localization logic actually contains a knee, independent of the report/metadata-level bilateral handling the host already confirmed in §2.5. One competitor's laterality-detection approach: derive left/right from an anatomical model plus patient positioning metadata, described as reasonably straightforward once you look at how the knee bends in the images. Do not attempt to convert one knee's laterality into the other via image flipping for these studies — flagged explicitly as clinically wrong by a commenter.

### 2.29 Disc 733836 — Clarification on Training Data (agr hmmm, 3 comments)

General framing discussion about the learning setup, with one substantive reply from Tucker Arrants worth preserving: there's no "proper" ground truth here in the usual sense — a radiologist's report is itself a diagnostic judgment call, and different radiologists disagree, which is true of every medical-imaging competition. What's unusual here is that the host explicitly told participants *how* the labels were made (two readers against a published rubric, third adjudicating, same process for test), and that this process disagrees with the reports in a *known, one-directional* way — annotators use stricter thresholds than a radiologist writing for clinical care, so the reports systematically over-call findings relative to gold. The reports are, in that framing, a richer signal than a binary label would be — but the reason to use more than the 58 provided labels is to calibrate your extractor against the competition's specific annotation logic, not to treat report language as ground truth in itself.

### 2.30 Disc 738708 & 737163 — two confirmed corrupted training studies

Two independent bug reports (Tahmid Tawsif, then Yee Ng four days later) converge on the exact same two `StudyInstanceUID`s, each with a pixel-data byte count exactly half of what the DICOM header (Group 0028) declares:

- `1.2.826.0.1.3680043.8.498.34685905030370793639196564723935583035` — 409,600 bytes present vs. 819,200 expected.
- `1.2.826.0.1.3680043.8.498.37833587429731221455928642963031995680` — 173,056 bytes present vs. 346,112 expected.

Multiple participants independently hit the identical `ValueError` on `pixel_array` decode. One participant confirmed via byte-count analysis that this is a genuine dataset defect, not a corrupted download — redownloading won't fix it. **Jason Sho (RSNA data curator, listed in the competition's Data Curators credits) acknowledged the report and said the files would be reviewed.** Community practice in the meantime: blacklist both study IDs during preprocessing/training rather than trying to decode them.

### 2.31 Disc 735304 — Best single-model score (roy214, 21↑, ~16-37 comments visible)

The largest engagement thread on the board. Self-reported single-model / single-architecture scores (public LB, self-reported, unverified):

| Competitor | Score | Setup |
|---|---|---|
| Scott Willis | 0.938 → 0.947 | Single model/single fold → later a 5-fold ensemble; deliberately started as small as possible, targeting the efficiency leaderboard first; reported spending significant effort "working around" low-quality labels (mechanism left unspecified as of this read) |
| Tucker Arrants | 0.934 → 0.942 | 224px, 5-fold ensemble, "some tweaks" |
| k256.dev | 0.926 | 5-fold, 336px, LLM output labels only |
| diet1236364 | 0.929 | 5-fold, 224px, LLM output labels only |
| Yann Majewski | 0.936 | Single fold, small ResNet, 224×224 |
| Tim Krige | 0.92 | Single model; reports OOF AUC = LB AUC within noise |
| Tom Aindow | 0.915 | DINOv2-based; later detail: ~392px, 150mm center crop (≈0.383mm/px), random bag of 32 slices/study |
| Berat Kirbiyik | stuck at 0.924 | Ran 7 ablations (larger backbone, cross-slot attention, top-k pooling, EMA, mixup, longer schedules, slice-count sweeps) all within 0.008 of each other — reads as a noise floor, and they explicitly point at label quality as the likely upstream cause; separately reports getting a CoAtNet-384 run down from 65 to 17 minutes by pipelining preprocessing against the GPU |

Recurring themes in the comments: several top-10 competitors argue the very top public LB scores (0.92–0.94+) are **not** achieved via 20+-model ensembles despite that being the pattern in public notebooks — cross-checked against the efficiency leaderboard, many high scorers also have short runtimes, inconsistent with mega-ensembling; more than one competitor characterizes the public-notebook ensembling pattern as more about visibility/kudos than genuine score gains. Tim Krige's advice: look directly at what the model sees (is 192px actually enough to resolve mm-scale pathology?) and invest in understanding what the labels need to encode before assuming a model problem. General DINOv2 sizing guidance from Tim Krige: score correlates with how much of the backbone is unfrozen, which trades directly against VRAM/speed; on limited hardware (3070/5070ti, ~1 day per 5-fold run) the smallest DINOv2 variant with the most unfreezing feasible tends to beat a larger, more-frozen one.

### 2.32 Disc 735826 — What is the real bottleneck now: labels, image pipeline, or model capacity? (Oscar Yáñez Feijóo, 9↑, 11 comments)

Direct community poll on where effort is paying off. Selected substantive replies:

- **k256.dev**: found gold-label inconsistencies that go beyond ordinary interpretation noise — filtered to reports whose effusion sentence used a small-size qualifier ("small"/"minimal"/"trace") and found this group contains **both** Effusion=0 and Effusion=1 gold labels; two reports using an identical English template sentence got **opposite** gold labels; at least one report stating a "moderate" effusion was labeled Effusion=0. Their read: even the gold set is not perfectly rule-following near the decision boundary, so this isn't purely a report-extraction problem.
- **Seung Sup Lee**: ran a controlled swap — froze a CPU preprocessing cache and label pipeline, then compared a pretrained ResNet34 (OOF BCE ≈ 0.664, control) against pretrained DINOv2-Small with slot attention on identical data/labels (best 5-fold candidate OOF BCE ≈ 0.674, rank correlation ~0.391) — **DINOv2 was worse**, not better. Their working (self-described as unproven) hypothesis after this: image geometry/slice selection + label quality matter more than backbone size. Next planned experiment: hold the successful ResNet34 pipeline fixed and vary only preprocessing (physical slice ordering, stack-end trimming, anatomical slice grouping, fixed-mm crop) to isolate the pipeline's contribution from the backbone's.
- **Tucker Arrants**: practical diagnostic — measure your label extractor's own AUC against the 58 gold, then measure your image model's AUC against the same 58 gold. If the image model doesn't beat the extractor "teacher," there's real modeling headroom; if it beats the extractor by a wide margin already, labels are likely the binding constraint.
- **Tom Aindow**: even the image-derived gold labels sometimes appear to depart from the competition's own stated threshold (e.g., a report explicitly describing a "small" effusion under a rubric requiring "moderate or large," yet labeled Effusion=1) — read as ordinary radiologist disagreement near a subjective boundary, not necessarily an error. Flags a related risk: an LLM label-extractor can report high confidence in applying a *textual* rule ("small" → negative) while being blind to how noisy the underlying word-choice process actually is — textual confidence isn't the same thing as diagnostic confidence.
- **PC Jimmmy**: characterizes the competition as fundamentally about label construction, paraphrasing host commentary that the gold-set annotators did not use the reports as a starting point (consistent with §2.5's host confirmation) — and cautions that because participants are largely creating their own training targets, "trust your CV" is a shakier heuristic here than usual.
- **Neelkant Newra**: built a "gold standard" extraction prompt validated against the 58 labeled studies, reporting **79% accuracy** (a different metric than the macro-AUC figures elsewhere in this digest, not directly comparable), then ran it across the full corpus (~14 GPU-hours, no failures).

### 2.33 Disc 735767 — What Could the Final Ceiling Be for RSNA 2026? (Tony Li, 13↑, 6 comments)

Mostly speculative leaderboard-ceiling discussion, useful for calibrating expectations. Context at time of posting: current #1 public LB ~0.951, publicly shared solutions ~0.92; competition growing unusually fast (1,832 teams at 12 days in, vs. 1,149 teams total for the entire RSNA 2025 competition). Predictions ranged from 0.97 (optimistic) to a more skeptical private-LB estimate of gold ≈0.94+, winner ≈0.96. The most substantive comment (PC Jimmmy) predicts a large shakeup and widespread overfitting given that participants are constructing their own labels, reiterating that "trust your CV" is unreliable here, and that models are effectively being trained to predict "the Kaggle truth" (the specific annotation rubric) rather than clinical ground truth in the abstract. A second commenter (Tim Krige) raises a distinct risk worth noting: increasingly fast, automated/agentic leaderboard-probing may itself inflate public LB scores in ways that don't generalize to private — a caution about over-indexing on public LB feedback for final submission selection.

### 2.34 Disc 738339 — Stuck around 0.74 OOF - what separates the 0.87 tier from 0.93+? (Abdullah Wasee, 2 comments)

A detailed self-reported ablation log: 0.739 macro AUC on the 58 gold (5-fold), report extraction itself at 0.725 vs. the same 58. Pipeline: rule-based multilingual extraction with clause-scoped bidirectional negation and severity grading, 2.5D ResNet-34 with gated attention pooling, 4 sequence slots × 3 adjacent-slice triplets at 256px, 140mm physical crop. Three concrete, numbered wins:

1. Sorting slices by `ImagePositionPatient` projected onto the slice normal rather than by filename — described as their single biggest bug fix (filenames are SOP Instance UIDs, effectively random order).
2. Grading mild findings as negative rather than scoring every mention at a flat high confidence: caught via a German report saying "geringer Gelenkerguss" (slight effusion) that was gold-negative. Fixing this moved Effusion 0.628→0.726 and ACL 0.775→0.855.
3. Sharpening soft labels away from a cluster near 0.5 (which barely trains under BCE) moved gold 0.653→0.725.

Remaining weak spots: Medial Meniscus 0.59, MCL 0.60 (thin structures) vs. Medial OA 0.89, Effusion 0.88 (large/diffuse findings) — the same large-vs-thin pattern documented independently in §2.35. Open questions raised: whether 336px+3-adjacent-slices beats more-slices-at-lower-resolution because of resolution or because of slice adjacency specifically (both are claimed by different public notebooks); how much of a ResNet→DINOv2 gap is real given §2.36's null result on Small→Base scaling; and whether the gap between public-notebook plateau (~0.87) and the top of the leaderboard (~0.952) is ensembling/scale or a genuinely different technique class. A commenter's tip (using stevenleehans' public label table, §2.1/§2.3) reportedly helped "more than expected."

### 2.35 Disc 737597 — What we've ruled out at 0.79 OOF — and what's left (Berat Kirbiyik, 0 comments)

A rigorous, single-variable ablation log. Setup: EfficientNet-B0, 4 sequence slots (Axial_FS, Sagittal_FS, Coronal_FS, Sagittal_T1), 9 adjacent slices at 288px from a 140mm physical crop, task-specific masked attention over slots, weighted BCE, 5-fold scanner-aware split.

- **Slice geometry beat resolution**: 192px/no-crop/24 equal-spaced slices → 0.7695 OOF; 288px/140mm crop/12 adjacent → 0.7777; 288px/140mm crop/center-9-of-24 → 0.7815. Center-9-adjacent beat 9-equal-spaced-across-24 by +0.018 on fold 0 — slice *position* mattered more than count or resolution alone.
- **Regularization was within noise** against a baseline of 0.7914: EMA +0.003, Mixup +0.002, longer cosine schedule +0.001. **Asymmetric loss actively hurt, -0.139** — a specific warning against that loss choice for this task.
- **Laterality-flip risk tested directly and largely ruled out** for their pipeline: mean |right-left| AUC gap was actually *smaller* for the five side-dependent targets (0.014) than for all other targets (0.031) — a genuine nuance against the "no horizontal flip for side-specific labels" caution (the underlying caution likely still holds as a design principle; this just shows their particular pipeline wasn't measurably hurt by it). Separately: the `Laterality` DICOM tag is present on only 2,185/4,407 studies, but **deriving side from the image center in patient coordinates resolves 4,274/4,407 (97%) and agrees with the tag 98.4% of the time** where both exist — a validated, high-coverage geometric method, corroborating the geometry-based approach recommended in §2.12.
- Per-label AUC at 0.78 OOF: Baker's 0.855, ACL 0.853, Effusion 0.843, Fracture 0.832 (large, discrete, positionally fixed findings) vs. Lateral Meniscus 0.740, PF OA 0.748, Lateral OA 0.751, MCL 0.758 (thin, small, graded findings) — the same large-vs-thin pattern as §2.34.
- **Open question, unanswered on the thread:** whether the remaining gap to 0.90+ requires an ROI crop around the joint line or spatial attention pooling instead of global average pooling — reasoning that a 3mm meniscal tear is roughly 0.01% of a 288px frame, so global pooling may be diluting the signal past usefulness. Directly related to the unanswered YOLO/ROI question in §2.40.

### 2.36 Disc 735154 — Scaling the encoder bought us nothing (+0.0011) (stevenleehans, 11↑, 2 comments)

The richest single engineering thread found this session, from the same author as §2.1.

**The headline result:** swapping DINOv2-Small→Base (22M→87M params, 3.9× compute/step) moved CV by +0.0011 against a measured paired noise floor of 0.0020 — a null. Per-label breakdown shows only 5 of 12 labels moved in Base's favor, and the entire macro gain rests on MCL (+0.0200), which has only 9 expert positives in the 58 gold studies — their own least-reliable label. For comparison, a real effect they'd found earlier (a crop-geometry fix, +0.0059) moved 10 of 12 labels in the same direction — "that's what a real effect looks like." Their stated per-label gate for treating a result as signal rather than noise: 0.03.

**The most transferable lesson — a checkpoint carries its own preprocessing contract.** Testing RAD-DINO (a ViT-B/14 finetuned from dinov2-base on 882,775 chest radiographs — architecturally identical to their dinov2-base arm, so a clean single-lever test of pretraining domain) it started losing immediately and the gap widened every epoch. The apparent conclusion ("medical pretraining doesn't transfer to MRI") was wrong: their code **hardcoded ImageNet normalization constants** (mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225]) as buffers, correct for the DINOv2 checkpoints but wrong for RAD-DINO, which expects different greyscale statistics (mean 0.5307, std 0.2583) and a native 518px input vs. the 224px they were feeding it. The fix: read `image_mean`/`image_std` from the checkpoint's own `preprocessor_config.json` rather than assuming. **Directly actionable before wiring in OrthoDiffusion**: verify its normalization constants and native resolution are read from its own config, not inherited from a different backbone's code path.

**Four other measurement pitfalls, all cheap to hit:** (1) a silent backbone fallback — off Kaggle there's no `/kaggle/input`, so a missing-weights lookup returned `None` and the builder silently fell back to ResNet-18, which trained fine and logged a plausible score; always log which backbone actually loaded. (2) A narrow timing probe (one part of the loop) predicted 0.64h/fold; the real fold took 1.11h because the probe ignored validation, memory-mapped reads, and augmentation — a probe is a go/no-go signal, never a schedule. (3) Apple's MPS backend agreed with itself to four decimals for two epochs, then diverged — a noise floor measured on one backend doesn't transfer to another; measure where you actually run. (4) Their own comparison-script bug: a tool written to compare two seeds of one config was fed a control-vs-treatment pair instead and mislabeled the *effect* as the *noise floor* under its "NOISE FLOOR" heading.

**Slice-selection methodology (from a comment reply), directly reusable:** sort the stack by physical position first — projecting `ImagePositionPatient` onto the normal from `ImageOrientationPatient`, with fallback to `SliceLocation`→`InstanceNumber`→filename — which they note resolves by geometry on 100% of series in this corpus and matters far more than the sampling step; **sorted-filename order matches true anatomical order only about 5% of the time on this corpus**, the most precisely quantified version of this warning found anywhere in this digest. Then place 3 anchors evenly across a clipped (not full) window of the sorted stack, and take the 3 physically-adjacent slices around each anchor (3×3=9 total), with each group of 3 becoming the R/G/B channels of one encoder input — reasoning that adjacent slices form a genuine ~10mm 2.5D neighborhood, whereas 9 evenly-spread slices would hand the encoder three views ~20mm apart, "a slab, not a triplet." Interpolation is deliberately avoided because slice gaps vary a lot across the corpus; physical scale is instead fixed via a constant-mm crop before resize (field of view ranges over 71 distinct values, median 160mm). Honest caveat: their finding that 3 slices beat 9 on CV is confounded with a change in window centering between the one-anchor and three-anchor configurations, so they can't yet attribute the win to slice count vs. slice position specifically.

**Practical infrastructure note:** the entire visual input after preprocessing (4,407 studies × 6 slots × 9 slices × 224×224, uint8) is only **11.12 GiB** — small enough to persist as a memory-mapped cache and iterate on a laptop; they measured a full fold at 67 min on an Apple M4 Pro (zero GPU quota) vs. 76 min on a Kaggle T4, because the laptop run never rebuilds the ~55-minute decode-and-crop cache that a fresh Kaggle kernel pays every time.

### 2.37 Disc 737696 — RSNA Raptor Weights (Dread Development, 9↑, comments)

Public pretrained-weights release from a top-10 competitor: a single (non-ensembled, non-5-fold) CoAtNet-384 model trained on the full labeled set with a small held-out set for checkpoint selection, currently at 0.924 public with an unreleased update reportedly at 0.936–0.938. Not a general-purpose alternative backbone in the OrthoDiffusion sense — it's inference weights plus a specific windowing scheme, not a pretrained encoder you'd substitute in. Useful mainly as an efficiency-track reference point: their full pipeline scores 94 windows per study and takes ~9 hours for the full test set (near the runtime cap), almost entirely window count rather than model cost; a trimmed 24-window version scored ~0.02 lower and a 42-window version only ~0.01 lower than the full 94-window run, suggesting steep diminishing returns on window count past a point — relevant if your own architecture ends up doing multi-window/multi-crop inference and needs to hit the sub-30-minute target.

### 2.38 Disc 733313 — Using Dino3 (Ryuhki Kimura, 9↑, several replies)

A licensing-friction thread, not directly about OrthoDiffusion but relevant background on how backbone licensing gets scrutinized in this competition. DINOv3 (Meta) requires an application/approval process that isn't a simple registration — one participant reported being rejected outright, raising the same "is this reasonably accessible to everyone" question already live for other gated resources (§2.20, §2.21). Empirically, one competitor (PC Jimmmy) found DINOv3 scored *worse* than DINOv2 on an otherwise-matched setup (0.763 vs. 0.775) — another data point, alongside §2.36 and §2.39, that a newer/larger backbone doesn't automatically help on this task.

### 2.39 Disc 738096 — Dinov2 Base vs Small at 224 resolution (Komil Parmar, 5↑, 2 comments)

A clean, matched ablation (same data, schedule, augmentation, seed — only the backbone swapped): DINOv2-Base converges faster, holding up to a +0.015 lead through early/mid training, but **both variants converge to the same final score given a long enough schedule** — the endpoint difference is within noise. Practical takeaway shared by the poster and a commenter: if compute- or time-bound, Base reaches a given score in fewer epochs; if training to convergence, Small gets the same result at ~4× less compute. Both competitors independently describe the same operating pattern: cheap configs (single fold, smallest backbone, 224px) for ablations/experimentation, then a single expensive final configuration (5-fold, largest backbone, higher resolution) only near the end of the competition — directly matching the "frozen-then-unfreeze… pooled feature caching for iteration, full spatial features for final runs" strategy already in your plan. Caveat: single fold/seed, 224px only; the poster believes (untested) that larger variants may pull ahead at higher resolutions.

### 2.40 Disc 736999 & 734357 — ROI cropping / YOLO-style detection (both unanswered)

**736999** (Mehmet Özer): asks directly whether anyone has tried a detector→crop→classify pipeline for the knee joint, specifically as a way to help the subtle meniscus/ligament findings. No replies, but the question connects directly to the open architectural question at the end of §2.35 (does the thin-structure gap need spatial attention or ROI cropping instead of global average pooling?). **734357** (Handudu): the same idea posed more generally ("has anyone manually labeled slice coordinates and used YOLO"), also unanswered. Between the two, this reads as a genuinely open architectural idea nobody has publicly reported results on yet — worth treating as a live option rather than a dead end, given how directly it targets the exact failure mode (thin structures, global pooling diluting a tiny signal) that multiple independent ablations (§2.34, §2.35) converge on.

### 2.41 Disc 738495 — Has anyone tried a GRU/LSTM head over slices here? (Abdullah Wasee, 0 comments, unanswered)

Notes that several past RSNA competitions' winning solutions used a recurrent head (GRU/LSTM) to aggregate slices rather than pooling — 2019 hemorrhage detection, 2022 cervical spine, 2023 abdominal trauma (where the winning writeup reportedly found GRU beat self-attention), 2024 lumbar spine — yet this competition's public consensus leans toward attention pooling over sequence slots instead, with no visible discussion of recurrent heads. The poster's own hypothesis for why: those prior competitions used CT volumes with ~96 slices, giving a recurrent head a real sequence to model, whereas most setups here use only 3–9 slices per plane, which may be too short a sequence for a GRU/LSTM to add anything over attention pooling. No replies to confirm or refute. Worth noting your plan's "attention-based cross-plane fusion" is already aligned with the community's current default rather than the recurrent-head pattern from prior years — and the poster's own reasoning (short sequences) would apply to your setup too if you were considering a recurrent alternative.

### 2.42 Disc 736301 — Are you actually looking at the data? (Chikuwabu, 11↑, 2 comments)

A joke thread referencing the well-known "invisible gorilla" radiology attentional-blindness study, with a real point in the comments: one competitor (PC Jimmmy, a veteran of many prior medical-imaging competitions) admits that in this competition specifically, leaning on coding agents/LLMs led them to skip the EDA phase they'd normally spend the first couple of weeks on — looking at hundreds or thousands of actual images and reports by hand. Worth taking as a general process reminder rather than a technical finding: confirm your team has actually looked at a meaningful sample of raw reports and images directly, not only through an automated extraction/preprocessing lens.

---

## 3. Direct implications for your pipeline

**3.1 — Tier 2 (Synovitis linear proxy) has strong external corroboration.** Your `0.22 + 0.41 × pooled_effusion` and the community's `P(syn|no eff)=0.22`, `P(syn|eff)=0.63` are numerically consistent (0.22 + 0.41 = 0.63). Two independent derivations landing on the same numbers is a good sign this isn't an artifact of your particular gold subset or LLM prompt — though it's still built on the same 58-study ruler (§2.7), so treat the *coefficients* as well-supported and the *precision* as bounded by the same ~0.02 noise floor. **New caveat from §2.9**: the underlying silence-ratio logic may be partly confounded by report language/script specifically for Synovitis and Baker's — the two columns where refusal rate moved most between Latin and non-Latin reports. Worth a quick check of whether your own pooled Synovitis predictions behave differently on the ~12% of studies containing Greek/Cyrillic script.

**3.2 — Re-scope the Tier 3 proxy-pair testing plan before running it.** The exact test you're planning (Baker's↔Effusion, Fracture↔Contusion) was already run in generalized form by stevenleehans (§2.1) — and it hurt Baker's, Contusion, and Lateral OA specifically, because for those findings report *silence* is itself strongly diagnostic (Baker's: 3% positive when silent vs. 44% when mentioned). Before testing a candidate proxy, measure that target's own silence-ratio first — a near-flat silence-ratio (like Synovitis) means a correlate-based proxy is likely to help; a steep one (like Baker's) means it's likely to subtract signal. For OA compartments specifically, §2.14 suggests the fix isn't a correlate proxy at all but a vocabulary problem: naive OA extraction scores below chance because the finding is almost never named directly, only described by its consequences (osteophytes, joint-space narrowing, chondral loss, "tricompartmental").

**3.3 — Consider evaluating a third label source.** stevenleehans' table benchmarks at 0.887–0.893 vs. your `report_labels_v2` component's 0.870 (§2.3), and a separate practitioner reports LLM-based extraction plateauing around 0.89 regardless of which LLM is used (§2.11) — suggesting that figure may be close to a practical ceiling worth benchmarking your own blend against. Worth a three-way comparison, or at minimum a diff to see where stevenleehans' table disagrees with your pooled labels.

**3.4 — Your overfitting flag has real statistical backing, independently confirmed twice.** Disc 733876 (§2.7) quantifies exactly what you flagged: paired σ ≈ 0.0125, a reliable signal needs ≈0.02 macro AUC of separation, and the gold set is enriched roughly 1.5–3× relative to the full corpus depending on the finding (independently corroborated at ~2× for ACL alone in §2.10). Disc 734105 (§2.12) is a second, methodologically serious confirmation on a much stronger pipeline — a genuine pre-registered SOFT-vs-HARD replication that still couldn't exclude zero on the 58 gold studies. The practice both threads converge on — rank/select on weak labels across all 4,407 reports, use the 58 gold only for calibration, never for architecture or major design selection — is a concrete, better-specified version of the "reserve a holdout slice" idea already on your list, worth adopting directly.

**3.5 — Mixed-to-negative signal on further backbone investment, converging from many directions.** In favor of medical-domain pretraining: a RadImageNet encoder specifically helped the hardest label, Synovitis (§2.2). Against further architecture investment generally: DINOv2 Small→Base was a null at 224px in two independent tests (§2.36, §2.39); one controlled test found DINOv2-Small *worse* than a ResNet34 control (§2.32); DINOv3 underperformed DINOv2 for one competitor (§2.38); and the broad thread consensus (§2.31, §2.32, §2.34, §2.35) leans toward "labels and input geometry dominate over backbone sophistication" for this competition specifically. Given the timeline pressure already in your working notes, this is a reasonable moment to sanity-check how much further architecture investment is worth relative to the label pipeline and preprocessing — several concrete preprocessing specifics surfaced this pass (130–150mm crop, ~0.38mm/px, physical slice ordering by `ImagePositionPatient` projection, 3-adjacent-slice triplets over evenly-spread single slices, laterality from geometry rather than the sparse `Laterality` tag) are worth a direct diff against your current pipeline regardless of the backbone question. If you do proceed with OrthoDiffusion, verify its normalization constants and native resolution are read from its own config rather than assumed — §2.36's checkpoint-normalization bug is the single most concrete, avoidable failure mode found in this entire pass.

**3.6 — Treat the external-data rules as genuinely unsettled, not just under-documented.** Four separate, unanswered threads (§2.21) directly ask whether the KneeCoT ruling (§2.20) retroactively threatens MRNet/OAI-style registration-gated datasets already assumed "green" in `rsna_knee_rules_reference.md` §7.2. This is a live tension in the rules themselves, not a documentation gap on your end — worth periodically re-checking those four threads (or asking directly) before committing meaningful compute to pretraining on any dataset behind a signed agreement, even one that felt like ordinary registration before the KneeCoT ruling landed.

**3.7 — Two data-quality items to bake into preprocessing now.** Blacklist the two confirmed-corrupted study IDs in §2.30. Add an explicit check that your Turkish-language negation handling scopes to the *right* of the term, not the left (§2.10) — the opposite of the English/Romance/Germanic default — and, if you're doing any bilateral-study detection or fixed-region cropping, verify the crop actually contains a knee for the small number of both-knees-in-one-series studies (§2.28).

**3.8 — Update the rules reference doc.** Several items from this pass belong in `rsna_knee_rules_reference.md`: (a) §7.4 upgraded from 🔶 to ✅ using the host quote in §2.5; (b) a new ruling added under §6.3/§7 — KneeCoT is explicitly disallowed (§2.20), with the four-thread unresolved-generalization cluster (§2.21) flagged as a live open question rather than resolved; (c) §8.2's Fluid_Sensitive/Fat_Suppression caution upgraded to host-confirmed with exact counts (§2.17); (d) PatientSex availability upgraded to host-confirmed (§2.27); (e) the two corrupted study IDs (§2.30) added as a known-bad list.

---

## 4. Provenance

Thread index built 2026-08-29 by paginating the Discussion tab (sort=hotness, 5 pages, ~92 topics) and extracting thread IDs via the page DOM. 38 threads read in full across three passes via direct navigation; contents paraphrased per Anthropic's copyright policy, with exact figures preserved. Everything above is ⚠️ forum-post-level (competitor claims) except where a host or RSNA-staff reply is explicitly marked — treat competitor-derived numbers (macro AUC estimates, prevalence percentages, LB scores) as other teams' self-reported measurements on the same small gold set or on an unverified public leaderboard, not independently verified. Threads listed as "reviewed by title, not opened" in §1's meta/logistics list were judged unambiguously non-technical from their titles alone (team formation, eligibility, prize payment, individual submission errors) and were not read.
