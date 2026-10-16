import json

import run_live


manifests = json.load(open("manifests.json", encoding="utf-8"))
protocol = json.load(open("protocol.json", encoding="utf-8"))
counts = run_live.submit_and_collect("Dongling", manifests["Dongling"])
run_live.analyze("Dongling", counts, protocol)
