import sys
import inspect

sys.path.insert(0, r".\src")

import production_inference_windows_apt_v4_1_aligned as aligned


print("=" * 80)
print("SmartSOC v4.1 — EXACT FUNCTION SIGNATURES")
print("=" * 80)

functions = [
    "normalize_wazuh_event",
    "derive_features",
    "build_behavior_text",
    "build_feature_matrix",
    "predict",
    "build_result_json",
]

for name in functions:

    fn = getattr(aligned, name, None)

    print(f"\n{name}")

    if fn is None:
        print("  NOT FOUND")
        continue

    print("  signature:", inspect.signature(fn))
    print("  module   :", fn.__module__)
    print("  source   :", inspect.getsourcefile(fn))


print("\n" + "=" * 80)
print("SmartSOCInference constructor / methods")
print("=" * 80)

from production_inference_windows_apt_v4_1 import SmartSOCInference

print("\nConstructor:")
print(inspect.signature(SmartSOCInference))

print("\nMethods:")

for name in [
    "__init__",
    "infer",
]:

    fn = getattr(SmartSOCInference, name, None)

    if fn:
        print(f"  {name}: {inspect.signature(fn)}")


print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)