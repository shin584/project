# Compute NT attribution at the encoder input embedding

Integrated Gradients against a Nucleotide Transformer checkpoint must take its gradients on the
**pre-encoder input-embedding tensor**, with a zero-embedding baseline, replaying the full encoder
and classification head — not on any post-encoder representation. `EsmClassificationHead` pools the
`<cls>` position alone, so attributing a post-encoder tensor yields gradients that are identically
zero at every sequence position: the SaCas9 pipeline shipped exactly that defect, produced an
all-zero Model B attribution, and needed a dedicated Colab notebook to verify the fix. The same
`EsmForSequenceClassification` shape is used by 9-Hydra v3, so the trap is unchanged; the cost of
discovering it late is regenerating every attribution artifact, and an all-zero attribution is easy
to mistake for "the model ignores the sequence" rather than for a wiring bug. Implementations
therefore carry a completeness check and a non-zero check as unit tests, and the `transformers`
version is pinned in the Colab driver and recorded in each artifact's metadata sidecar, because
this failure originates in library internals rather than in our own code.
