"""Step 1b/1c — focused unit test of the EKO pricing-model dispatch.

Exercises core.eko_pricing.price_eko_dispatch (the shared helper used
by both the live pricer page and the single-leg backtester) on a known
case from Step 1a and verifies that all three modes match the expected
prices to 4 decimal places, and that the VV branch returns a complete
detail dict.

Originally Step 1b's first sanity check; updated for Step 1c when the
dispatcher moved out of the streamlit page into core/eko_pricing.py so
both callers could share it.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from core.smile import smile_vol_at_strike
from core.eko_pricing import price_eko_dispatch, PRICING_MODELS

# Headline test case from Step 1a
S, T = 155.00, 1/12
sigma_atm, rr_25, bf_25 = 0.07468, -0.01435, 0.00208
r_d, r_f = 0.005, 0.045
K = 154.4842   # ATMF from Step 1a output
H = 159.1850   # spot+2.7%

sigma_smile = smile_vol_at_strike(S, K, T, sigma_atm, rr_25, bf_25, r_d, r_f)

print("Dispatch sanity — USDJPY 1M ATMF UO call, H = spot+2.7%")
print(f"  σ_smile(K) computed by helper: {sigma_smile*100:.4f}%")
print()

# Expected from Step 1a:
expected = {
    "flat_atm":      0.822907,
    "vol_at_strike": 0.822723,
    "vanna_volga":   0.957444,
}

print(f"{'model':<14s}  {'dispatch':>10s}  {'expected':>10s}  {'diff':>10s}  {'status':>6s}")
print(f"{'-'*14}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*6}")
all_ok = True
for model in PRICING_MODELS:
    p, det = price_eko_dispatch(
        "call", "up_and_out", S, K, H, T,
        sigma_atm, sigma_smile, rr_25, bf_25, r_d, r_f, model,
    )
    diff = p - expected[model]
    ok = abs(diff) < 1e-4
    flag = "OK" if ok else "FAIL"
    if not ok:
        all_ok = False
    print(f"{model:<14s}  {p:10.6f}  {expected[model]:10.6f}  {diff:+10.2e}  {flag:>6}")
    assert det.get("model") == model, f"model field mismatch: {det}"

# Check VV-specific detail keys
print()
p, det = price_eko_dispatch(
    "call", "up_and_out", S, K, H, T,
    sigma_atm, sigma_smile, rr_25, bf_25, r_d, r_f, "vanna_volga",
)
required_keys = {"model", "vol_used", "correction", "price_bs", "vv_detail"}
got_keys = set(det.keys())
assert required_keys <= got_keys, f"missing VV keys: {required_keys - got_keys}"
print(f"VV detail keys present: {sorted(got_keys & required_keys)}")
print(f"VV correction         : {det['correction']:+.6f}")
print(f"VV.price_bs           : {det['price_bs']:.6f}")
print(f"VV.detail.weights     : {tuple(round(w, 3) for w in det['vv_detail']['weights'])}")
print(f"VV.detail.cond. number: {det['vv_detail']['condition_number']:.2e}")

# Error path
try:
    price_eko_dispatch("call", "up_and_out", S, K, H, T,
                        sigma_atm, sigma_smile, rr_25, bf_25,
                        r_d, r_f, "bogus")
    print("FAIL: bogus model should have raised")
    all_ok = False
except ValueError as e:
    print(f"bogus model correctly raised ValueError: {e}")

print()
print("ALL OK" if all_ok else "SOME FAILED")
