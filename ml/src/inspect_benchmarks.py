import json
import os

with open('ml/results/isot/benchmark.json') as f:
    isot = json.load(f)

with open('ml/results/liar/benchmark.json') as f:
    liar = json.load(f)

print("=== ISOT SUMMARY ===")
print("Dataset:", isot["dataset"]["name"], "Total:", isot["dataset"]["totalRecords"])
print("Class Dist:", isot["dataset"]["classDistribution"])
print("Splits:", isot["splits"])
for m in isot["models"]:
    print(f"\nModel: {m['name']} ({m['id']})")
    print(f"  Train Time: {m.get('trainingTime')}")
    print(f"  Test Metrics: {m['test']['metrics']}")
    if "crossValidation" in m:
        print(f"  CV Mean: {m['crossValidation']['mean']}")
        print(f"  CV Std: {m['crossValidation']['std']}")
    if "paperReference" in m:
        print(f"  Paper Ref: {m['paperReference']}")

print("\n=== LIAR SUMMARY ===")
print("Dataset:", liar["dataset"]["name"], "Total:", liar["dataset"]["totalRecords"])
print("Class Dist:", liar["dataset"]["classDistribution"])
print("Splits:", liar["splits"])
for m in liar["models"]:
    print(f"\nModel: {m['name']} ({m['id']})")
    print(f"  Train Time: {m.get('trainingTime')}")
    print(f"  Test Metrics (0.50): {m['test']['metrics']}")
    if "metricsAtTunedThreshold" in m['test']:
        print(f"  Test Metrics (0.55): {m['test']['metricsAtTunedThreshold']}")
    if "crossValidation" in m:
        print(f"  CV Mean: {m['crossValidation']['mean']}")
        print(f"  CV Std: {m['crossValidation']['std']}")
    if "paperReference" in m:
        print(f"  Paper Ref: {m['paperReference']}")
