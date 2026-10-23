# 9-Hydra v2 honors Kim 2020's own Use column instead of a grouped split

The SaCas9 pipeline fixed its replicate leak by building a sequence-grouped split from scratch (issues #33-#35), so the obvious move for the SpCas9 multi-variant model was a protospacer-grouped split in the same style. We did not do that. `dataset/kim_2020_table_s8.xlsx` ships a `Use (Training or Test)` column that 9-Hydra v1's notebook ignored in favour of `train_test_split(test_size=0.1, random_state=42)`, and that column turns out to be clean on both leak axes we care about: zero protospacers and zero exact 30-mers are shared between its Training and Test rows, and it keeps all 30 PAM-library protospacers intact on one side or the other (25 train / 5 test). v1's own split, by contrast, put 740 of its 2,931 test rows in a protospacer it had trained on and 41 as literal duplicates.

So 9-Hydra v2 uses the published column verbatim for train/test, and carves only its validation set itself (protospacer-grouped, seed 42, 10% of Training) because the file offers no validation column and v1's Keras `validation_split=0.2` leaked protospacers into early stopping. Building our own train/test split would have bought no additional cleanliness while destroying the one thing the published column gives us for free: comparability to the numbers in the paper.

## Consequences

- Our test set is fixed by the dataset, so we cannot resize it, rebalance it, or regenerate it under a different seed. If a future dataset re-export changes that column, the split changes with it — hence the `dataset_sha256` pin in `split_checksum.json`. `SpCas9/spcas9_split.py` enforces this by refusing a non-default `--seed` that would overwrite the published split in place.
- "10% of Training" above is 10% of the training **protospacers**, not of the training rows. Grouping moves whole protospacers together, and the PAM library's heavy protospacers (up to 268 rows each) make the row share land near but not on 10%: 2,075 of 20,759 protospacers, which is 3,676 of 32,810 rows, or 11.2%. `split_checksum.json` is the authority on the exact figures.
- v1 cannot be fairly compared to the published model at all: of the 3,847 `Use = Test` rows inside v1's all-9-heads frame, 2,781 were trained on and 665 validated on, leaving 401 clean. That is reported as a finding, not worked around.
- v2 differs from v1 on two axes, not one: the split *and* the output activation (sigmoid → linear). The activation change is not cosmetic — v1's labels span negative values that a sigmoid head cannot represent, so v1 was mis-specified as well as leaky — but it does mean a v1↔v2 delta cannot be attributed to the dataset fix alone. A third run (v1's architecture including sigmoid, on the clean split) would isolate it, and was deliberately not built; architecture work is deferred to v3.

## Considered Options

- **Protospacer-grouped split of our own, in the style of #34**: rejected. It fixes a leak the published column does not have, and forfeits comparability to Kim 2020.
- **Both, with the published column as headline and a grouped split as a secondary check**: rejected as unjustified cost — two sets of numbers per head across nine heads, to validate a column we had already tested directly.
