"""Plot retained #203 observations; run after all measurements have ended."""

import argparse
import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(output):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), layout="constrained")
    for axis, mode, title in zip(axes, ("legacy", "v3"),
                                 ("Preserved NT source (production disk SHAP hits)",
                                  "v3 source: non-NT / NT / non-NT")):
        folder = output / mode
        with gzip.open(folder / "samples.jsonl.gz", "rt", encoding="utf-8") as stream:
            samples = [json.loads(line) for line in stream]
        x = [s["elapsed"] for s in samples]
        for metric, label, color in (("rss", "Working set sum", "#235daa"),
                                     ("private", "Private commit sum", "#bd5c1e")):
            axis.plot(x, [s[metric] / 1024**3 for s in samples], label=label, color=color, linewidth=1.4)
        phases = [json.loads(line) for line in (folder / "phases.jsonl").read_text(encoding="utf-8").splitlines()]
        origin = samples[0]["time"] - samples[0]["elapsed"]
        for phase in phases:
            if phase["label"] in ("select_family", "return_stable"):
                offset = phase["time"] - origin
                axis.axvline(offset, color="#666666", linestyle=":", linewidth=1)
                label = "to " + phase["family"] if phase["label"] == "select_family" else "stable"
                axis.text(offset + 2, 5.05, label, fontsize=8, rotation=90, va="top")
        axis.set(title=title, xlabel="Seconds since process launch", ylabel="GiB", ylim=(0, 5.3))
        axis.grid(alpha=.2)
        axis.legend(loc="upper left", fontsize=8)
    fig.suptitle("#203 source measurements — same PC, input and defaults; TensorFlow import cost included", fontsize=11)
    fig.savefig(output / "ram-timeline.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    plot(parser.parse_args().output.resolve())
