# MC_project

Cleavage-efficiency prediction for CRISPR nucleases (SpCas9, SaCas9, Cas12a) and
the scanner that ranks candidate target sites with them.

This glossary holds repo-wide vocabulary. The SaCas9 XAI comparison pipeline
keeps its own, larger glossary in `xai_test/CONTEXT.md`.

## Language

**protospacer**:
The 23 bp stretch a Cas12a guide RNA pairs with, at 0-indexed positions 8-30 of
a 34 bp target-context sequence — immediately 3′ of the PAM. It is what the
GC and dinucleotide features are computed over, never the full 34 bp.
_Avoid_: "guide", "spacer", "target sequence" (the last one is the whole 34 bp
row, PAM and flanks included)

**PAM**:
The 4 bp motif Cas12a requires immediately 5′ of the protospacer, at positions
4-7 of the 34 bp sequence. Cas12a's PAM is 5′ of its protospacer, unlike
SpCas9's and SaCas9's, which sit 3′. Written as an IUPAC motif: the scanner
queries `TTTV`, while the training library is the broader `TTTN` — see
ADR 0001.
_Avoid_: calling the Cas12a PAM "downstream" or "3′"; naming a 3 bp PAM
(`TTT` is only the invariant part of a 4 bp motif)

**target efficiency**:
The quantity every model in this repo predicts: a **0-1 ratio**, not a
percentage. For Cas12a it is the measured indel frequency after background
subtraction, divided by 100 and clipped to `[0, 1]`. Percent is a display unit
and the conversion belongs to the display layer.
_Avoid_: "score" unqualified when the unit matters; mixing ratios and
percentages in one comparison

**Cas12a library**:
The Kim 2018 dataset the Cas12a model is trained on: 15,000 usable 34 bp
synthetic targets measured in HEK293T, one row per unique protospacer. Its PAM
is `TTTN`, not `TTTV`.
_Avoid_: "DeepCpf1 dataset" when referring to our loaded form of it (the paper's
model and splits are not ours); "15,002 rows" (the raw sheet's count, two of
which have no target value)
