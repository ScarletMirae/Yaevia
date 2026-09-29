"""
Phase 0: Backup all relevant source files, database, and record current active model.
"""
import os
import shutil
import sqlite3

backup_dir = r"D:\Yaevia\backend\backups\pre_frozen_production_256"
os.makedirs(backup_dir, exist_ok=True)

# 1. Backup DB
db_src = r"D:\Yaevia\backend\database.db"
db_dst = os.path.join(backup_dir, "database_pre_freeze.db")
shutil.copy2(db_src, db_dst)
print(f"Database backed up to: {db_dst}")

# 2. Check current active model in DB
conn = sqlite3.connect(db_src)
conn.row_factory = sqlite3.Row
active_models = conn.execute("SELECT * FROM model_meta WHERE is_active = 1").fetchall()
all_models = conn.execute("SELECT * FROM model_meta ORDER BY id DESC").fetchall()
conn.close()

print(f"\nTotal models in DB: {len(all_models)}")
print(f"Active models in DB: {len(active_models)}")
for m in active_models:
    print(f"  Active Model ID={m['id']}, File={m['model_filename']}, TestAcc={m['test_accuracy']}, Created={m['train_timestamp']}, Features={m['feature_vector_size']}, K={m['knn_k']}")

# 3. Backup source files
src_files = [
    r"D:\Yaevia\backend\config.py",
    r"D:\Yaevia\backend\model\classifier.py",
    r"D:\Yaevia\backend\model\trainer.py",
    r"D:\Yaevia\backend\features\hog_extractor.py",
    r"D:\Yaevia\backend\routes\verify.py",
    r"D:\Yaevia\backend\routes\train.py",
    r"D:\Yaevia\backend\routes\dataset.py",
    r"D:\Yaevia\frontend\upload.html",
    r"D:\Yaevia\frontend\js\upload.js",
    r"D:\Yaevia\frontend\verify.html",
    r"D:\Yaevia\frontend\js\verify.js",
]

for sf in src_files:
    if os.path.exists(sf):
        dst = os.path.join(backup_dir, os.path.basename(sf))
        shutil.copy2(sf, dst)
        print(f"Backed up {os.path.basename(sf)} -> {dst}")
