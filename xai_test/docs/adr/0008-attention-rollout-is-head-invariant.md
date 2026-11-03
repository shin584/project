# Attention rollout is head-invariant, so per-head questions go to ISM and IG

9-Hydra v3's nine regression heads sit on **one shared encoder trunk**, and attention rollout is a
function of that encoder's attention weights alone. There is therefore exactly **one rollout per
row, not one per head**, and the v3 XAI analysis stores it as `(n_rows, 5, 5)` with no head axis.
Rollout answers which tokens the trunk routes information between; it **cannot** answer whether
`SpCas9` and `SpCas9-NG` use the PAM differently, which is the analysis' second question.

A per-head variant is constructible — weight the attention by each head's gradient, grad-CAM
style — and is deliberately **out of scope**. It is a different quantity from the one ADR 0006
fixes (Integrated Gradients on the pre-encoder input embedding), so adopting it would add a second
attribution definition to reconcile, and the per-head question already has two independent
answers: ISM at base resolution and IG at the 5-token resolution, whose agreement ADR 0007 makes
a reported check rather than an assumption.

Recording this because the analysis plan lists attention rollout among the techniques for an
explicitly multi-head model, so a reader who expects a per-head rollout figure will look for one
and find none. Discovering the invariance late is cheap in compute and expensive in interpretation:
nine identical rollout matrices read naturally as a bug or as "the heads attend identically", when
in fact the quantity has no head to vary over. Unfolding the rollout to 30x30 base coordinates is
likewise refused, per ADR 0007 — attention is a quantity between tokens, and the last token binds
the PAM to the right flank, so unfolding re-mixes exactly what that ADR separates. The SaCas9
pipeline unfolded its rollout to 36x36; that choice is not inherited.
