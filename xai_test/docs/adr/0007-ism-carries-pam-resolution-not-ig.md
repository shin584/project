# ISM carries base-level PAM resolution, not Integrated Gradients

NT tokenises DNA as non-overlapping 6-mers, so a 30bp SpCas9 target window is exactly five tokens
with boundaries `0-5 | 6-11 | 12-17 | 18-23 | 24-29` — which puts the 3nt PAM (24-26) and the right
flank (27-29) inside **one** token. Integrated Gradients yields one attribution per token, and
dividing it evenly across the token's six bases (the SaCas9 pipeline's Deterministic Projection
Rule) cannot recover which half the signal came from, so any per-base "PAM attribution" derived
that way is really a PAM-plus-flank average. Because per-head PAM behaviour is a primary question
for 9-Hydra v3, IG is therefore reported at five-token resolution only, with that token named
`pam_plus_right_flank` rather than `pam`, and the base-level PAM result comes from in-silico
saturation mutagenesis, whose single-base substitutions are indifferent to token boundaries.
The rejected alternatives were reading positions 24-26 off the projected attribution anyway, which
answers the central question incorrectly while looking correct, and tokenising six 1bp-shifted
copies and averaging, which costs six times the inference and averages over six different token
vocabularies with no clear meaning. SaCas9's 36bp window masked this problem rather than avoiding
it: its PAM window at 25-30 straddles two tokens, so the contamination was diffuse instead of a
clean PAM/flank merge.
