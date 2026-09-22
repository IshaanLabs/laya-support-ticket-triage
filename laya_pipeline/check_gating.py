"""Runnable checks for gating + temperature scaling math.
Run: python -m laya_pipeline.check_gating   (no Laya / no GPU needed)
"""
import math

from laya_pipeline.gating import (
    apply_temperature,
    calibrated_confidence,
    fit_temperature,
    gate,
)

# --- gate() threshold behaviour ---
assert gate(0.90, threshold=0.85) == "auto_route"
assert gate(0.85, threshold=0.85) == "auto_route"      # boundary is inclusive
assert gate(0.84, threshold=0.85) == "escalate_to_human"

# --- apply_temperature invariants ---
p = {"a": 0.7, "b": 0.2, "c": 0.1}
# T=1.0 is a no-op.
same = apply_temperature(p, 1.0)
assert all(abs(same[k] - p[k]) < 1e-9 for k in p)
# Output is always a valid distribution.
hot = apply_temperature(p, 0.5)   # sharpen
cold = apply_temperature(p, 3.0)  # soften
for d in (hot, cold):
    assert abs(sum(d.values()) - 1.0) < 1e-6
# Sharpening raises the top prob; softening lowers it. Argmax never changes.
assert hot["a"] > p["a"] > cold["a"]
assert max(hot, key=hot.get) == max(cold, key=cold.get) == "a"

# --- calibrated_confidence returns argmax + its scaled prob ---
label, conf = calibrated_confidence(p, 2.0)
assert label == "a"
assert 0.0 < conf < 1.0

# --- fit_temperature recovers a softening T when the model is over-confident ---
# Build fake records where the model is 0.95 on the WRONG label half the time:
# a good calibrator should pick T>1 (soften) to reduce NLL.
records = []
for i in range(100):
    if i % 2 == 0:
        records.append({"probs": {"x": 0.95, "y": 0.05}, "truth": "x"})  # right
    else:
        records.append({"probs": {"x": 0.95, "y": 0.05}, "truth": "y"})  # wrong, over-confident
T = fit_temperature(records, "probs", "truth")
assert T > 1.0, f"expected softening T>1 for over-confident model, got {T}"

print("OK: gating + temperature scaling checks passed")
