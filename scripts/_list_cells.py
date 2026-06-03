"""List cells in notebook 10."""
import json

NB = "notebooks/analysis/10_cooling_thermal_calibration.ipynb"
with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, c in enumerate(nb["cells"]):
    ct = c["cell_type"]
    cid = c.get("id", "?")
    src = "".join(c["source"])[:80].replace("\n", "\\n")
    print(f"{i}: [{ct}] id={cid} => {src}")
