"""Output-unit contract for cleavage-efficiency scores.

Every model in this repo - SpCas9 9-Hydra, Cas12a, SaCas9 - reports efficiency
as a **0-1 ratio**. The factor 100 belongs to the display layer and appears
here exactly once, in `to_percent`.

Two of the three heads are sigmoid and therefore already bounded; SaCas9's is a
plain `nn.Linear` and is not, so a score coming from a model goes through
`clip_score` before anything else consumes it (#89).

This module lives in `cas12a/` because that is the only place in the repo where
the convention can be tested without importing TensorFlow (#91). `e_system`
still carries its own `MODEL_SCORE_SCALE` multiplication and has not been wired
to this module yet - that connection is a separate issue.
"""

SCORE_MIN = 0.0
SCORE_MAX = 1.0
PERCENT_PER_UNIT = 100.0


def clip_score(raw):
    """Bring a model's raw output into `[0, 1]`."""
    return min(max(float(raw), SCORE_MIN), SCORE_MAX)


def to_percent(score):
    """Convert a 0-1 score to a percentage, for display only."""
    return float(score) * PERCENT_PER_UNIT


def final_score(raw, penalty):
    """Apply a `<= 1` penalty to a raw score, clipping first.

    Clipping before the multiplication is what keeps the result inside
    `[0, 1]`: an unbounded raw score multiplied by, say, a 0.5 PAM-distance
    penalty would still land above 1.

    Nothing calls this yet — `e_system` still does the multiplication itself.
    It exists so that the ordering decision above is held by a test rather than
    by a review comment, which is what #89 asked for.
    """
    if not SCORE_MIN <= penalty <= SCORE_MAX:
        raise ValueError(
            f"penalty must be within [{SCORE_MIN}, {SCORE_MAX}], got {penalty}"
        )
    return clip_score(raw) * float(penalty)
