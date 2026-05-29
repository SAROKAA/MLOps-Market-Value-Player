import json

THRESHOLD_R2 = 0.60


with open("metrics.json") as f:
    metrics = json.load(f)

r2 = metrics["r2"]

print(f"R² = {r2}")

if r2 >= THRESHOLD_R2:
    print("Model validation PASSED")
else:
    raise Exception(
        f"Model validation FAILED. "
        f"R2 {r2} < {THRESHOLD_R2}"
    )