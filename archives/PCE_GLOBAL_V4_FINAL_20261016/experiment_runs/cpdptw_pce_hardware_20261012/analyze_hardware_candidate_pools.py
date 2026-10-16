import json
from pathlib import Path

import numpy as np

from capacity_core import correlations_from_counts, readout_parameters
from run_live import normalize


here = Path(__file__).parent
protocol = json.loads((here / "protocol.json").read_text(encoding="utf-8"))
slots = [tuple(slot) for slot in protocol["slots"]]

ideal_values = np.asarray(protocol["ideal_correlations"])
ideal_top3 = []
ideal_ranks = []
for block in range(4):
    order = np.argsort(-ideal_values[block * 8:(block + 1) * 8], kind="stable")
    ideal_top3.append(order[:3].tolist())
    ideal_ranks.append(int(np.where(order == 0)[0][0]) + 1)
print("ideal", "top3", ideal_top3, "optimum_action_0_ranks", ideal_ranks)

for chip in ("Baihua", "Dongling"):
    raw = {
        name: normalize(json.loads((here / chip / "raw" / f"{name}.json").read_text(encoding="utf-8")))
        for name in ("zero", "one", "alternating", "pairs", "pce_Z", "pce_X", "pce_Y")
    }
    readout = readout_parameters(raw["zero"], raw["one"], 8)
    for mode in ("raw", "corrected"):
        values = np.zeros(len(slots))
        for basis in "ZXY":
            indexes = [i for i, slot in enumerate(slots) if slot[0] == basis]
            subset = [slots[i] for i in indexes]
            values[indexes] = correlations_from_counts(
                raw[f"pce_{basis}"], subset, None if mode == "raw" else readout
            )
        top3 = []
        ranks_of_optimum = []
        for block in range(4):
            block_values = values[block * 8:(block + 1) * 8]
            order = np.argsort(-block_values, kind="stable")
            top3.append(order[:3].tolist())
            ranks_of_optimum.append(int(np.where(order == 0)[0][0]) + 1)
        print(chip, mode, "top3", top3, "optimum_action_0_ranks", ranks_of_optimum)
