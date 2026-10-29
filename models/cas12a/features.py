"""Canonical Cas12a sequence features.

Pure NumPy: no TensorFlow, no pandas, no `cas12a.data` import. This is the
canonical definition, because a copy that drifts produces predictions that look
fine and are wrong (#91).

`cas12a/model/Cas12a_model_final.ipynb` reads from here.
`e_system/predict.py:60,68` does **not** yet: it still carries its own
byte-identical copy, left alone because `e_system` is outside #91's scope — so
the drift risk above is live, and `cas12a/docs/cas12a-model-review.md` §5
tracks it. Wiring it up points the dependency `e_system` -> `cas12a`, never
back; `cas12a` must not import `e_system`.

The 34 bp row layout comes from the Kim 2018 column header,
"4 bp + PAM + 23 bp protospacer + 3 bp", and is verified against the data in
`cas12a/docs/cas12a-model-review.md` §1.2.
"""

import numpy as np

SEQUENCE_LENGTH = 34
PAM_SLICE = slice(4, 8)
PROTOSPACER_SLICE = slice(8, 31)
PROTOSPACER_LENGTH = PROTOSPACER_SLICE.stop - PROTOSPACER_SLICE.start
#: Length of the PAM-distal protospacer tail that the `gc_seed` feature covers.
#: The notebook named that feature "seed" even though Cas12a's seed region is
#: PAM-proximal — the opposite end. The name is kept to match the feature it
#: produces; it is not a claim about which end matters biologically.
SEED_LENGTH = 8
#: Overlapping dinucleotide positions in a 23 bp protospacer.
DINUCLEOTIDE_POSITIONS = PROTOSPACER_LENGTH - 1

BASES = "ACGT"
ONE_HOT = {
    "A": [1, 0, 0, 0],
    "C": [0, 1, 0, 0],
    "G": [0, 0, 1, 0],
    "T": [0, 0, 0, 1],
}
UNKNOWN_BASE = [0, 0, 0, 0]
DINUCLEOTIDE_INDEX = {
    first + second: index
    for index, (first, second) in enumerate(
        (first, second) for first in BASES for second in BASES
    )
}

#: Width of `extract_seq_features` concatenated: 2 GC + 22x16 dinucleotide.
FEATURE_DIM = 2 + DINUCLEOTIDE_POSITIONS * len(DINUCLEOTIDE_INDEX)


def one_hot_encode(sequence):
    """One-hot a DNA string as `(len(sequence), 4)` over ACGT.

    Any other character - including the `N` padding `cas12a.data` adds to a
    short sequence - encodes as all zeros rather than raising.
    """
    sequence = sequence.upper()
    return np.array([ONE_HOT.get(base, UNKNOWN_BASE) for base in sequence])


def protospacer_of(sequence):
    """The 23 bp protospacer of a 34 bp target-context sequence."""
    return sequence[PROTOSPACER_SLICE]


def extract_seq_features(seq):
    """GC and dinucleotide features of a 34 bp sequence's protospacer.

    Returns `([gc_total, gc_seed], dinucleotide)`, where `dinucleotide` is a
    flat 22x16 one-hot of the overlapping pairs: position `i` occupies
    `[i*16, (i+1)*16)`. A pair containing a non-ACGT base leaves its block at
    zero, so the vector sums to 22 only for a clean protospacer.

    The name and the abbreviated `seq` parameter are the notebook's and
    `e_system`'s, kept so existing call sites move here unchanged.
    """
    protospacer = protospacer_of(seq)
    gc_total = (protospacer.count("G") + protospacer.count("C")) / PROTOSPACER_LENGTH
    seed = protospacer[-SEED_LENGTH:]
    gc_seed = (seed.count("G") + seed.count("C")) / SEED_LENGTH

    dinuc_features = np.zeros(
        DINUCLEOTIDE_POSITIONS * len(DINUCLEOTIDE_INDEX), dtype=np.float32
    )
    for position in range(DINUCLEOTIDE_POSITIONS):
        pair = protospacer[position : position + 2]
        if pair in DINUCLEOTIDE_INDEX:
            offset = position * len(DINUCLEOTIDE_INDEX)
            dinuc_features[offset + DINUCLEOTIDE_INDEX[pair]] = 1.0

    return np.array([gc_total, gc_seed], dtype=np.float32), dinuc_features
