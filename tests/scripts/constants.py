"""Shared test constants across CI scripts.

This module centralizes configuration lists used by:
- ci_smoke_test.py
- mock_simulation_runner.py
- record_baselines.py

When adding new fast configs, update only this file.
"""

# Configs under 2 minutes runtime for smoke tests and baseline recording
# Sorted by runtime (fastest first) based on timing database
FAST_CONFIGS = [
    "breastmnist_krum_vs_labelflip20.json",
    "femnist_pidstdscore_baseline.json",
    "femnist_bulyan_baseline.json",
    "femnist_mkrum_baseline.json",
    "femnist_trust_baseline.json",
    "femnist_rfa_baseline.json",
    "femnist_pidstd_baseline.json",
    "femnist_trimmean_baseline.json",
    "femnist_pid_baseline.json",
    "femnist_pidscaled_baseline.json",
    "femnist_krum_baseline.json",
    "femnist_rfa_vs_labelflip20.json",
    "femnist_mkrum_vs_labelflip20.json",
    "femnist_krum_multi_overlapping.json",
    "femnist_trust_vs_labelflip20.json",
    "femnist_pidstd_vs_labelflip20.json",
    "femnist_krum_vs_labelflip20.json",
    "femnist_mkrum_vs_labelflip50.json",
    "femnist_pidstdscore_vs_labelflip20.json",
    "femnist_bulyan_vs_labelflip50.json",
    "femnist_mkrum_multi_concurrent.json",
    "femnist_krum_vs_labelflip50.json",
    "femnist_mkrum_multi_showcase.json",
    "femnist_mkrum_vs_gaussnoise25.json",
]

# Derived config categories for selective testing
BASELINE_CONFIGS = [c for c in FAST_CONFIGS if "baseline" in c]
ATTACK_CONFIGS = [c for c in FAST_CONFIGS if "vs_" in c or "multi_" in c]

# Version for baseline format compatibility checking
BASELINE_FORMAT_VERSION = "1.0"
