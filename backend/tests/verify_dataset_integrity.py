"""
Verification of dataset integrity:
  - exactly 360 active records
  - exactly 18 students
  - exactly 20 samples per student
  - all raw files exist
  - orphan files check
"""
import os
import sqlite3

db_path = r"D:\Yaevia\backend\database.db"
raw_dir = r"D:\Yaevia\backend\dataset\raw"
proc_dir = r"D:\Yaevia\backend\dataset\processed"

conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
rows = conn.execute("SELECT * FROM dataset ORDER BY student_name ASC, original_filename ASC").fetchall()
conn.close()

print(f"Total database records: {len(rows)}")
students = sorted(list(set(r["student_name"] for r in rows)))
print(f"Total students: {len(students)}")

counts = {}
missing_raw = []
for r in rows:
    st = r["student_name"]
    counts[st] = counts.get(st, 0) + 1
    raw_name = r["saved_filename"] or os.path.basename(r["file_path"])
    raw_p = os.path.join(raw_dir, raw_name)
    if not os.path.exists(raw_p):
        missing_raw.append((r["id"], st, raw_name))

print("\nSamples per student:")
all_exact_20 = True
for st, cnt in counts.items():
    status = "OK (20)" if cnt == 20 else f"ANOMALY ({cnt})"
    if cnt != 20:
        all_exact_20 = False
    print(f"  {st:<32}: {cnt} samples [{status}]")

print(f"\nMissing raw files: {len(missing_raw)}")
if missing_raw:
    for m in missing_raw:
        print("   Missing:", m)

raw_files = os.listdir(raw_dir)
db_saved_names = set(r["saved_filename"] for r in rows)
orphan_raw = [f for f in raw_files if f not in db_saved_names]
print(f"\nTotal files in raw directory: {len(raw_files)}")
print(f"Orphan raw files: {len(orphan_raw)}")
if orphan_raw:
    print("   Orphan raw files:", orphan_raw[:5])

# Check duplicate records
fn_counts = {}
for r in rows:
    key = (r["student_name"], r["saved_filename"])
    fn_counts[key] = fn_counts.get(key, 0) + 1
duplicates = [k for k, v in fn_counts.items() if v > 1]
print(f"Duplicate database entries: {len(duplicates)}")

print(f"\nSummary Integrity Status:")
print(f"  - 360 records: {'PASS' if len(rows) == 360 else 'FAIL'}")
print(f"  - 18 students: {'PASS' if len(students) == 18 else 'FAIL'}")
print(f"  - 20 samples/student: {'PASS' if all_exact_20 else 'FAIL'}")
print(f"  - 0 missing raw files: {'PASS' if len(missing_raw) == 0 else 'FAIL'}")
print(f"  - 0 duplicates: {'PASS' if len(duplicates) == 0 else 'FAIL'}")
