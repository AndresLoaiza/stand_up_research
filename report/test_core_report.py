"""Self-checks with synthetic data for the report functions in analysis/core.py.

    python report/test_core_report.py
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis import (callbacks, dry_chains, fightin_words, laugh_gaps, memory_permutation_p, percentile_of,  # noqa: E402
                      style_from_blocks, text_blocks)

# percentile: ties count half, NaN ignored
assert percentile_of(5, [1, 2, 3, 4]) == 100
assert percentile_of(2, [1, 2, 3, np.nan]) == 50

# style blocks: 1st vs 2nd person share and questions
b = text_blocks("I think I am right. Are you sure? " * 300)
s = style_from_blocks(b)
assert abs(s["yo_vs_tu"] - 100 * 2 / 3) < 1e-6, s
assert abs(s["preguntas_1000"] - 1000 * 300 / (300 * 8)) < 1e-6, s
es = style_from_blocks(text_blocks("Yo te dije que su perro me mordió. " * 200, "es"))
assert abs(es["yo_vs_tu"] - 100 * 2 / 3) < 1e-6, es  # yo, me vs te; "su" is not 2nd person

# callbacks: a phrase said again far later counts; a refrain repeated many times does not
filler = " ".join(f"word{i}" for i in range(2000))
t = "the purple elephant dances tonight " + filler + " remember the purple elephant dances"
assert callbacks(t)["n"] == 1, callbacks(t)
refrain = " ".join(["purple elephant dances tonight"] * 10 + [filler])
assert callbacks(refrain)["n"] == 0

# laugh gaps: words between markers
assert laugh_gaps("a b [laughter] c d e [laughter] f [laughs] g", min_markers=3) == [3, 1]

# Fightin' Words: a word A uses far more than B gets a high positive z
a, bb = Counter({"pizza": 50, "the": 500}), Counter({"pizza": 2, "the": 500, "soup": 48})  # same totals
fw = fightin_words(a, bb, a + bb).set_index("palabra")
assert fw.loc["pizza", "z"] > 3 and abs(fw.loc["the", "z"]) < 1

# dry chains: two stretches less than 60 s apart form one chain
sec = pd.DataFrame({"inicio_min": [1.0, 2.5, 10.0], "duracion_s": [45, 60, 45]})
ch = dry_chains(sec)
assert len(ch) == 1 and ch[0]["tramos"] == 2 and ch[0]["seco_s"] == 105, ch

# room memory: a smooth series has memory, white noise does not
rng = np.random.default_rng(1)
smooth = np.convolve(rng.normal(size=400), np.ones(8) / 8, mode="same")
assert memory_permutation_p(smooth, n_perm=300)[1] < 0.01
assert memory_permutation_p(rng.normal(size=400), n_perm=300)[1] > 0.01

print("OK: funciones del informe")
