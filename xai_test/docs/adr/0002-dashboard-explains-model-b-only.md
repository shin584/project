# Dashboard explains Model B alone, not the Model A vs Model B comparison

The XAI pipeline (Steps 2–5) is built around comparing Model A and Model B: Case Study classes (Concordant / Discordant) are defined by the two models' relative errors, and Integrated Gradients is exported for both. The visualization dashboard (`visualization/`) nonetheless shows only Model B, because its purpose is to let an end user understand why the deployed predictor produced a given score, not to argue that Model B beats Model A. Consequently the dashboard never shows Model A predictions or attributions, and never surfaces Concordant/Discordant labels, since those cannot be explained without Model A. The 15 Case Studies are reused only as the samples that have Integrated Gradients available.

## Considered Options

- Show Model A and Model B side by side: rejected. Comparison is what the analysis report is for, and side-by-side views would dilute the single-model explanation the dashboard is meant to give.

## Scope

This decision applies to the dashboard only. The demo CLI deliberately compares both models; see ADR 0003.
