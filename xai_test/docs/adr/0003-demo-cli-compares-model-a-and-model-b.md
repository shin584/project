# Demo CLI shows Model A next to Model B, unlike the dashboard

ADR 0002 restricts the visualization dashboard (`visualization/`) to Model B: no Model A output and no Concordant/Discordant labels. That restriction is scoped to the dashboard only. The demo CLI (`xai_demo.py`, backed by `xai_session.py`) deliberately shows Model A predictions, errors and attributions next to Model B's and surfaces the Concordant / Discordant case types, because the presentation it serves argues a comparative claim — whether Model B's gains over Model A reflect mechanism learning — and that claim cannot be made without both models and the case types defined by their relative errors.

The two surfaces still consume the same export contract (`final_analysis_result/`), so they never disagree about a shared number such as Model B's prediction; they differ only in which numbers they choose to show.

## Considered Options

- Extend ADR 0002's Model-B-only rule to the CLI: rejected. The CLI would then be unable to show the Primary Discordant story (A misses, B hits) or the Reverse Discordant counter-examples that the presentation relies on.
