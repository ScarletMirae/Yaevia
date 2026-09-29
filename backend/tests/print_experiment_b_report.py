import json

p_main = r"D:\Yaevia\backend\tests\evaluation_results\roi_experiment\ab_comparison_report.json"
p_ff   = r"D:\Yaevia\backend\tests\evaluation_results\roi_experiment\fahim_fathur_analysis.json"

with open(p_main, encoding="utf-8") as f:
    r = json.load(f)
with open(p_ff, encoding="utf-8") as f:
    ff = json.load(f)

B = r["baseline"]
E = r["experiment"]
C = r["comparison_summary"]

print("=" * 72)
print("EXPERIMENT B FINAL REPORT  --  BASELINE vs TEXT-DENSITY ROI")
print("=" * 72)
fmt = "{:<40} {:>10}  {:>10}  {:>8}"
print(fmt.format("Metric", "Baseline", "Experiment", "Delta"))
print("-" * 72)

rows = [
    ("Overall Accuracy (%)",
        B["accuracy"]["overall_accuracy"], E["accuracy"]["overall_accuracy"]),
    ("Page-1 Accuracy (%)",
        B["accuracy"]["page1_accuracy"], E["accuracy"]["page1_accuracy"]),
    ("Page 2-20 Accuracy (%)",
        B["accuracy"]["page2_20_accuracy"], E["accuracy"]["page2_20_accuracy"]),
    ("Intra-class Mean Dist",
        B["distance_metrics"]["overall_intra_mean"], E["distance_metrics"]["overall_intra_mean"]),
    ("Inter-class Mean Dist",
        B["distance_metrics"]["overall_inter_mean"], E["distance_metrics"]["overall_inter_mean"]),
    ("Separability Ratio",
        B["distance_metrics"]["separability_ratio"], E["distance_metrics"]["separability_ratio"]),
    ("ROI Coverage Mean",
        B["roi_coverage"]["mean"], E["roi_coverage"]["mean"]),
    ("ROI Coverage pct>=85pct",
        B["roi_coverage"]["pct_ge85"], E["roi_coverage"]["pct_ge85"]),
    ("FG Ratio @128x128 Mean",
        B["foreground_ratio_128"]["mean"], E["foreground_ratio_128"]["mean"]),
    ("Mean Winning Vote Share (%)",
        B["vote_distribution"]["mean_winning_vote_share"], E["vote_distribution"]["mean_winning_vote_share"]),
    ("Mean Margin (#1-#2 %)",
        B["vote_distribution"]["mean_margin"], E["vote_distribution"]["mean_margin"]),
    ("Vote 1/5 (%)",
        B["vote_distribution"]["pct_1_of_5"], E["vote_distribution"]["pct_1_of_5"]),
    ("Vote 2/5 (%)",
        B["vote_distribution"]["pct_2_of_5"], E["vote_distribution"]["pct_2_of_5"]),
    ("Vote >=3/5 (%)",
        B["vote_distribution"]["pct_ge3_of_5"], E["vote_distribution"]["pct_ge3_of_5"]),
]

for label, bv, ev in rows:
    d    = ev - bv
    sign = "+" if d >= 0 else ""
    print(fmt.format(label, f"{bv:.3f}", f"{ev:.3f}", f"{sign}{d:.3f}"))

n = 360
print()
print(f"Fallback count (exp)            : {E['fallback_count']}/{n} ({E['fallback_pct']}%)")
print(f"Page-1 baseline failures        : {C['page1_baseline_failures']}/18")
print(f"Page-1 experiment fixes         : {C['page1_experiment_fixes']}")
print(f"Page-1 regressions              : {C['page1_experiment_regressions']}")
print(f"Total regressions (all pages)   : {C['total_regressions_all_pages']}")

print()
print("=" * 72)
print("FAHIM vs FATHUR CASE STUDY")
print("=" * 72)
ff_fmt = "{:<45} {:>12}  {:>12}"
print(ff_fmt.format("Metric", "Baseline", "Experiment"))
print("-" * 72)

fb = ff["baseline"]
fe = ff["experiment"]

case_rows = [
    ("Fahim intra_mean dist",        "fahim_intra_mean"),
    ("Fathur intra_mean dist",       "fathur_intra_mean"),
    ("Fahim-Fathur cross mean",      "cross_mean"),
    ("Fahim-Fathur cross min",       "cross_min"),
    ("Cross < Fahim intra mean (%)", "cross_below_fahim_intra_mean_pct"),
    ("Cross < Fathur intra mean (%)","cross_below_fathur_intra_mean_pct"),
    ("Fahim->Fathur errors",         "fahim_misclassified_as_fathur"),
    ("Fathur->Fahim errors",         "fathur_misclassified_as_fahim"),
    ("Fahim total misclassified",    "fahim_total_misclassified"),
    ("Fathur total misclassified",   "fathur_total_misclassified"),
]
for label, key in case_rows:
    bv = fb[key]
    ev = fe[key]
    print(ff_fmt.format(label, str(bv), str(ev)))

if C["regression_samples"]:
    print()
    print("REGRESSION SAMPLES (baseline correct -> experiment wrong):")
    for s in C["regression_samples"]:
        print(f"  {s['student_name']}  sample#{s['sample_num']}  ({s['filename']})")
else:
    print()
    print("No additional regressions beyond page-1.")

print("=" * 72)
print("Output files:")
import os
out = r"D:\Yaevia\backend\tests\evaluation_results\roi_experiment"
for fname in ["ab_comparison_report.json","page1_comparison.csv",
              "misclassification_comparison.csv","fahim_fathur_analysis.json"]:
    fpath = os.path.join(out, fname)
    size  = os.path.getsize(fpath) if os.path.exists(fpath) else -1
    print(f"  {fname}  ({size:,} bytes)")
print("  visualizations/  (360 x 7-panel PNG)")
print("=" * 72)
