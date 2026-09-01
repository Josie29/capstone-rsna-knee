# How We Combined Two AI Readings Into One Label

## The problem

We asked two different AI passes to read the same radiology reports and estimate,
for each of 12 knee conditions, how likely it is the condition is present. One pass
(`v2`) reads more like a cautious human: it commits to Yes, No, or "can't tell,"
and only gives a fine-grained number when it commits. The other (`gpt56sol`) always
gives a precise number, even when the report barely touches the topic.

They agree well most of the time. But averaging them blindly would be a mistake,
because "can't tell" isn't the same problem for every condition.

## The core issue: silence means different things for different findings

For some conditions — a torn ACL, meniscus damage — if it's there, radiologists
almost always say so. Silence is itself a meaningful signal that it's probably not
present.

For others — synovitis especially — the opposite is true. It's common (nearly half
our verified cases have it) but rarely gets written down explicitly, because it's
often incidental to whatever the scan was really ordered for. Here, silence tells
us almost nothing, and treating it as "probably absent" would be actively wrong.

## The fix: three tiers, not one rule

Instead of one blending formula for every cell, we sort each reading into one of
three buckets:

1. **The report says something explicit.** Use both AI passes' readings, weighted
   by how confident each one reported itself to be. This is the strong-signal
   case and gets full weight in training.

2. **The report is silent, but a related finding fills the gap.** We tested this
   directly for synovitis: effusion (fluid in the joint) is reported far more
   reliably than synovitis, and the two occur together often enough that
   effusion turns out to be a better predictor of synovitis than the AI's direct
   synovitis reading is. So when synovitis is unaddressed, we estimate it from
   the joint's effusion reading instead — but only ever to fill a gap, never to
   overrule an explicit statement. This gets partial weight.

3. **The report is silent, and no reliable stand-in exists.** We fall back to the
   more precise AI pass's own best guess, but the model is told to trust this
   reading less during training, since it wasn't grounded in explicit language.

## Why not just guess "probably negative" everywhere?

Because that's what silently destroys a column like synovitis: nearly half of
those cases really are positive, so defaulting everything unaddressed to "no"
would train the model to systematically miss a condition it should be catching in
roughly half its true occurrences.

## What this produces

For each of the 4,407 studies and 12 conditions, we output three things:
a **probability estimate**, a **tier** (1, 2, or 3, so it's always visible which
kind of evidence produced the number), and a **training weight** (how much that
cell should influence the model, separate from what the number itself says).
Keeping the estimate and the confidence-in-that-estimate as separate columns,
rather than blending them into one softened number, is deliberate — it lets the
model learn from a confident guess without being punished as hard if that guess
turns out wrong.

This is a first pass. The proxy relationship (effusion → synovitis) is validated;
similar relationships for other under-reported conditions are flagged as
worth testing but not yet applied.
