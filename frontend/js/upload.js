/**
 * upload.js — Logika Halaman Upload & Training
 * Yae Miko Theme | Sistem Verifikasi Tulisan Tangan
 * ==============================================
 */

const API = API_BASE;
let allDataset = [];


// ─────────────────────────────────────────────────────────
// DRAG & DROP FILE UPLOAD
// ─────────────────────────────────────────────────────────
const dropZone  = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
let selectedFiles = [];

dropZone.addEventListener("dragover",  (e) => { e.preventDefault(); dropZone.classList.add("dragover"); });
dropZone.addEventListener("dragleave", ()  => dropZone.classList.remove("dragover"));
dropZone.addEventListener("click",     ()  => fileInput.click());
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  handleFiles(e.dataTransfer.files);
});
fileInput.addEventListener("change", () => handleFiles(fileInput.files));

// Prevent form submit on drop zone click from bubbling
dropZone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
});

function handleFiles(fileList) {
  selectedFiles = Array.from(fileList);
  const queue = document.getElementById("file-queue");
  const list  = document.getElementById("file-queue-list");
  if (!selectedFiles.length) { queue.style.display = "none"; return; }
  queue.style.display = "block";
  list.innerHTML = selectedFiles.map((f) => `
    <div style="display:flex;align-items:center;gap:0.5rem;padding:0.38rem 0.7rem;background:var(--soft);border-radius:var(--radius-sm);margin-bottom:0.3rem;font-size:0.82rem;border:1px solid var(--pink);">
      <i data-lucide="file-image" style="width:14px;height:14px;color:var(--rose-gold);flex-shrink:0;"></i>
      <span style="flex:1;color:var(--text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">${f.name}</span>
      <span style="color:var(--text-muted);flex-shrink:0;">${(f.size/1024).toFixed(1)} KB</span>
    </div>`).join("");
  if (window.lucide) lucide.createIcons({ nodes: [list] });
}

// ─────────────────────────────────────────────────────────
// FORM SUBMIT — UPLOAD
// ─────────────────────────────────────────────────────────
document.getElementById("upload-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!selectedFiles.length) { showToast("Pilih file gambar terlebih dahulu", "error"); return; }

  const studentName = document.getElementById("student-name").value.trim();
  const studentId   = document.getElementById("student-id").value.trim();
  const mataKuliah  = document.getElementById("mata-kuliah").value;
  const notes       = document.getElementById("notes").value.trim();

  if (!studentName) { showToast("Nama mahasiswa wajib diisi", "error"); return; }
  if (!mataKuliah)  { showToast("Pilih mata kuliah terlebih dahulu", "error"); return; }

  const btn     = document.getElementById("upload-btn");
  const btnText = document.getElementById("upload-btn-text");
  btn.disabled  = true;
  btn.innerHTML = `<span class="spinner" style="width:16px;height:16px;border-width:2px;"></span> Mengupload...`;

  let success = 0, failed = 0;
  for (const file of selectedFiles) {
    const formData = new FormData();
    formData.append("file",         file);
    formData.append("student_name", studentName);
    formData.append("student_id",   studentId);
    formData.append("mata_kuliah",  mataKuliah);
    formData.append("notes",        notes);
    try {
      const res  = await fetch(API + "/api/dataset/upload", { method: "POST", body: formData });
      const data = await res.json();
      if (data.success) success++;
      else { failed++; console.warn("Upload gagal:", data.message); }
    } catch (err) { failed++; console.error(err); }
  }

  btn.disabled = false;
  btn.innerHTML = `<i data-lucide="upload-cloud"></i> Upload Citra`;
  if (window.lucide) lucide.createIcons({ nodes: [btn] });

  if (success > 0) {
    showToast(`${success} file berhasil diupload!`, "success");
    selectedFiles = [];
    fileInput.value = "";
    document.getElementById("file-queue").style.display = "none";
    document.getElementById("upload-form").reset();
    loadDataset();
    loadDatasetStats();
  }
  if (failed > 0) showToast(`${failed} file gagal diupload`, "error");
});

// ─────────────────────────────────────────────────────────
// TRAINING — ANIMATED STEPS
// ─────────────────────────────────────────────────────────

/* Step timing (ms until each step activates after training starts).
   The actual training is synchronous on server; we simulate progress. */
const STEP_DELAYS  = [0, 1500, 3500, 6000, 9000, 12000];
const STEP_PCTS    = [8,  25,   45,   65,   85,   96  ];
const STEP_LABELS  = [
  "Menyiapkan Dataset",
  "Mengekstraksi Fitur HOG",
  "Membagi Dataset (80:20)",
  "Melatih Model KNN",
  "Menghitung Akurasi",
  "Menyimpan Model",
];

let stepTimers = [];

function resetTrainingUI() {
  stepTimers.forEach(clearTimeout);
  stepTimers = [];
  const steps = document.querySelectorAll(".training-step");
  steps.forEach((s) => s.className = "training-step");
  setProgress(0, "");
}

function setProgress(pct, label) {
  const bar  = document.getElementById("train-progress-bar");
  const text = document.getElementById("train-progress-text");
  const pctEl= document.getElementById("train-progress-pct");
  if (bar)   bar.style.width = pct + "%";
  if (text && label) text.textContent = label;
  if (pctEl) pctEl.textContent = pct + "%";
}

function activateStep(idx) {
  // Mark previous as done
  if (idx > 0) {
    const prev = document.getElementById(`step-${idx - 1}`);
    if (prev) {
      prev.classList.remove("active");
      prev.classList.add("done");
      const ind = prev.querySelector(".step-indicator");
      if (ind) ind.innerHTML = `<i data-lucide="check" style="width:13px;height:13px;"></i>`;
      if (window.lucide) lucide.createIcons({ nodes: [prev] });
    }
  }
  // Activate current
  const cur = document.getElementById(`step-${idx}`);
  if (cur) {
    cur.classList.add("active");
    setProgress(STEP_PCTS[idx], STEP_LABELS[idx] + "...");
  }
}

function finishAllSteps() {
  for (let i = 0; i < 6; i++) {
    const el = document.getElementById(`step-${i}`);
    if (el) {
      el.classList.remove("active");
      el.classList.add("done");
      const ind = el.querySelector(".step-indicator");
      if (ind) ind.innerHTML = `<i data-lucide="check" style="width:13px;height:13px;"></i>`;
    }
  }
  if (window.lucide) lucide.createIcons();
  setProgress(100, "Training selesai!");
}

async function startTraining() {
  const btn = document.getElementById("train-btn");
  const wrap = document.getElementById("training-steps-wrap");
  const result = document.getElementById("train-result");

  const knnK   = parseInt(document.getElementById("param-k").value) || 5;
  const metric = document.getElementById("param-metric").value;
  const orient = parseInt(document.getElementById("param-orient").value) || 9;
  const ppc    = parseInt(document.getElementById("param-ppc").value) || 8;

  // Show steps panel
  resetTrainingUI();
  wrap.style.display = "block";
  result.innerHTML = "";
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner" style="width:16px;height:16px;border-width:2px;"></span> Training berlangsung...`;

  // Animate steps with fake delays WHILE real request runs
  STEP_DELAYS.forEach((delay, i) => {
    const t = setTimeout(() => activateStep(i), delay);
    stepTimers.push(t);
  });

  try {
    const res = await fetch(API + "/api/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        knn_k:           knnK,
        metric:          metric,
        orientations:    orient,
        pixels_per_cell: [ppc, ppc],
        cells_per_block: [2, 2],
        test_size:       0.2,
      }),
    });
    const data = await res.json();

    // Clear pending step timers & finalize
    stepTimers.forEach(clearTimeout);
    stepTimers = [];
    finishAllSteps();

    if (data.success) {
      showToast("Training model berhasil!", "success");
      result.innerHTML = `
        <div class="card" style="margin-top:0;background:linear-gradient(135deg,#d4f5e9,#f0fff8);border-color:rgba(76,175,80,0.35);">
          <div class="card-title" style="color:#1a5c3a;">
            <i data-lucide="check-circle-2" style="color:#4caf50;"></i>
            Training Berhasil!
          </div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.6rem;margin-top:0.25rem;">
            ${resultChip("Akurasi Uji",    (data.test_accuracy  || 0) + "%",  "target")}
            ${resultChip("Precision",      (data.precision      || 0) + "%",  "crosshair")}
            ${resultChip("Recall",         (data.recall         || 0) + "%",  "zap")}
            ${resultChip("F1 Score",       (data.f1_score       || 0) + "%",  "award")}
            ${resultChip("Data Latih",     data.n_train,                       "database")}
            ${resultChip("Data Uji",       data.n_test,                        "flask-conical")}
            ${resultChip("Kelas",          (data.n_classes || 0) + " mhs",    "users")}
            ${resultChip("Nilai K",        data.knn_k,                         "git-merge")}
          </div>
          <div style="margin-top:0.75rem;text-align:center;">
            <a href="evaluate.html" class="btn btn-secondary btn-sm" style="font-size:0.8rem;">
              <i data-lucide="bar-chart-2"></i> Lihat Evaluasi Detail
            </a>
          </div>
        </div>`;
      if (window.lucide) lucide.createIcons({ nodes: [result] });
    } else {
      // Tampilkan pesan gagal — termasuk daftar mahasiswa jika gagal validasi
      stepTimers.forEach(clearTimeout);
      const insuf = data.insufficient_students || [];
      const insufHtml = insuf.length
        ? `<div style="margin-top:0.65rem;">
            <p style="font-size:0.78rem;font-weight:700;color:var(--text);margin-bottom:0.4rem;">
              Mahasiswa yang perlu dilengkapi datanya:
            </p>
            <div style="display:flex;flex-wrap:wrap;gap:0.35rem;">
              ${insuf.map(n => `<span class="badge badge-red"><i data-lucide="user-x"></i>&nbsp;${n}</span>`).join("")}
            </div>
          </div>`
        : "";
      showToast("Training gagal: " + (data.message || ""), "error");
      result.innerHTML = `
        <div class="alert alert-error">
          <i data-lucide="circle-x"></i>
          <div>
            <strong>Training Dibatalkan</strong><br>
            <span style="font-size:0.85rem;">${data.message || "Terjadi kesalahan."}</span>
            ${insufHtml}
          </div>
        </div>`;
      if (window.lucide) lucide.createIcons({ nodes: [result] });

  } catch {
    stepTimers.forEach(clearTimeout);
    finishAllSteps();
    showToast("Tidak dapat terhubung ke server Flask", "error");
    result.innerHTML = `<div class="alert alert-error"><i data-lucide="wifi-off"></i><span>Tidak dapat terhubung ke server. Pastikan <code>python app.py</code> sudah berjalan.</span></div>`;
    if (window.lucide) lucide.createIcons({ nodes: [result] });
  }

  btn.disabled = false;
  btn.innerHTML = `<i data-lucide="zap"></i> Mulai Training`;
  if (window.lucide) lucide.createIcons({ nodes: [btn] });

  // Auto-hide steps after 4s
  setTimeout(() => {
    if (wrap) wrap.style.display = "none";
  }, 4500);
}

function resultChip(label, val, icon) {
  return `
    <div style="background:rgba(255,255,255,0.7);border-radius:10px;padding:0.55rem 0.75rem;display:flex;align-items:center;gap:0.5rem;">
      <i data-lucide="${icon}" style="width:14px;height:14px;color:var(--rose-gold);flex-shrink:0;"></i>
      <div>
        <div style="font-size:0.68rem;color:#555;font-weight:600;letter-spacing:0.03em;">${label}</div>
        <div style="font-size:0.92rem;font-weight:800;color:#1a5c3a;">${val}</div>
      </div>
    </div>`;
}

// ─────────────────────────────────────────────────────────
// DUMMY INFO
// ─────────────────────────────────────────────────────────
function showDummyInfo() {
  showToast("Buka terminal → cd backend → python generate_dummy_data.py", "info", 6000);
}

// ─────────────────────────────────────────────────────────
// DATASET TABLE
// ─────────────────────────────────────────────────────────
async function loadDataset() {
  const tbody = document.getElementById("dataset-tbody");
  tbody.innerHTML = `<tr><td colspan="8" style="text-align:center;padding:1.5rem;color:var(--text-muted);">
    <span class="spinner"></span>&nbsp; Memuat data...
  </td></tr>`;
  try {
    const res = await fetch(API + "/api/dataset/list");
    const d   = await res.json();
    allDataset = d.data || [];
    renderDatasetTable(allDataset);
  } catch {
    tbody.innerHTML = `<tr><td colspan="8" style="text-align:center;padding:2rem;color:var(--text-muted);">
      <i data-lucide="wifi-off" style="width:18px;height:18px;"></i>&nbsp; Tidak dapat terhubung ke server
    </td></tr>`;
    if (window.lucide) lucide.createIcons({ nodes: [tbody] });
  }
}

function renderDatasetTable(rows) {
  const tbody = document.getElementById("dataset-tbody");
  if (!rows.length) {
    tbody.innerHTML = `<tr><td colspan="8" style="text-align:center;padding:2.5rem;color:var(--text-muted);">
      Dataset masih kosong. Upload citra tulisan tangan di atas.
    </td></tr>`;
    return;
  }
  tbody.innerHTML = rows.map((row, i) => {
    const fname = row.original_filename.length > 24
      ? row.original_filename.substring(0, 24) + "…"
      : row.original_filename;
    const processed = row.is_processed
      ? `<span class="badge badge-green"><i data-lucide="check-circle-2"></i>&nbsp;Selesai</span>`
      : `<span class="badge badge-yellow">Pending</span>`;
    return `
    <tr>
      <td style="font-weight:700;color:var(--text);">${i + 1}</td>
      <td style="font-weight:600;color:var(--text);">${row.student_name}</td>
      <td><span class="badge badge-pink">${row.student_id || "—"}</span></td>
      <td style="font-size:0.81rem;">${row.mata_kuliah}</td>
      <td style="font-size:0.79rem;color:var(--text-muted);" title="${row.original_filename}">${fname}</td>
      <td>${processed}</td>
      <td style="font-size:0.76rem;color:var(--text-muted);">${formatDate(row.upload_timestamp)}</td>
      <td>
        <button class="btn btn-danger btn-sm" onclick="deleteDatasetItem(${row.id})" title="Hapus">
          <i data-lucide="trash-2"></i>
        </button>
      </td>
    </tr>`;
  }).join("");
  if (window.lucide) lucide.createIcons({ nodes: [tbody] });
}

function filterTable() {
  const nameQ = document.getElementById("filter-name").value.toLowerCase();
  const mkQ   = document.getElementById("filter-mk").value;
  const filtered = allDataset.filter(r => {
    return (!nameQ || r.student_name.toLowerCase().includes(nameQ)) &&
           (!mkQ   || r.mata_kuliah === mkQ);
  });
  renderDatasetTable(filtered);
}

async function deleteDatasetItem(id) {
  if (!confirm("Hapus data ini dari dataset?")) return;
  try {
    const res = await fetch(API + `/api/dataset/${id}`, { method: "DELETE" });
    const d   = await res.json();
    if (d.success) { showToast("Data berhasil dihapus", "success"); loadDataset(); loadDatasetStats(); }
    else showToast(d.message, "error");
  } catch { showToast("Gagal menghapus", "error"); }
}

async function deleteAllDataset() {
  if (!confirm("Hapus SEMUA dataset? Tindakan ini tidak dapat dibatalkan.")) return;
  try {
    const res = await fetch(API + "/api/dataset/all", { method: "DELETE" });
    const d   = await res.json();
    if (d.success) { showToast(d.message, "success"); loadDataset(); loadDatasetStats(); }
    else showToast(d.message, "error");
  } catch { showToast("Gagal menghapus semua data", "error"); }
}

// ─────────────────────────────────────────────────────────
// DATASET STATS
// ─────────────────────────────────────────────────────────
async function loadDatasetStats() {
  const el = document.getElementById("dataset-stats-content");
  try {
    const res = await fetch(API + "/api/dataset/stats");
    const d   = await res.json();
    if (!d.success) throw new Error();

    el.innerHTML = `
      <div class="stat-row" style="margin:0 0 0.75rem;">
        <div class="stat-chip" style="min-width:90px;">
          <div class="stat-chip-icon"><i data-lucide="image"></i></div>
          <div class="stat-chip-value">${d.total}</div>
          <div class="stat-chip-label">Total Citra</div>
        </div>
        <div class="stat-chip" style="min-width:90px;">
          <div class="stat-chip-icon"><i data-lucide="users"></i></div>
          <div class="stat-chip-value">${d.classes_count}</div>
          <div class="stat-chip-label">Mahasiswa</div>
        </div>
      </div>
      ${d.total === 0 ? `<div class="alert alert-info"><i data-lucide="info"></i><span>Dataset kosong. Upload minimal 2 citra dari 2 mahasiswa berbeda untuk mulai training.</span></div>` : ""}
      ${d.total > 0 && d.classes_count < 2 ? `<div class="alert alert-warning"><i data-lucide="triangle-alert"></i><span>Diperlukan minimal 2 kelas untuk training. Tambahkan data mahasiswa lain.</span></div>` : ""}
      ${d.total > 0 && d.classes_count >= 2 ? `<div class="alert alert-info"><i data-lucide="check-circle-2"></i><span>Dataset siap digunakan untuk training!</span></div>` : ""}`;
    if (window.lucide) lucide.createIcons({ nodes: [el] });
  } catch {
    el.innerHTML = `<span style="color:var(--text-muted);font-size:0.86rem;">Tidak dapat terhubung ke server.</span>`;
  }
}

// ─────────────────────────────────────────────────────────
// DATASET SUMMARY CARD
// ─────────────────────────────────────────────────────────
async function loadDatasetSummary() {
  try {
    const res = await fetch(API + "/api/dataset/summary");
    const d   = await res.json();
    if (!d.success) return;

    document.getElementById("ds-n-students").textContent = d.n_students ?? "0";
    document.getElementById("ds-total").textContent      = d.total ?? "0";
    document.getElementById("ds-train").textContent      = d.n_train_estimate ?? "0";
    document.getElementById("ds-test").textContent       = d.n_test_estimate ?? "0";
    document.getElementById("ds-avg").textContent        = d.avg_per_student ?? "0";

    // Tabel per-mahasiswa
    if (d.students && d.students.length > 0) {
      const tbody = document.getElementById("ds-student-tbody");
      const wrap  = document.getElementById("ds-student-table-wrap");
      if (tbody && wrap) {
        wrap.style.display = "block";
        tbody.innerHTML = d.students.map((s, i) => {
          const ok = s.count >= 2;
          return `
          <tr>
            <td style="font-weight:700;color:var(--text);">${i + 1}</td>
            <td style="font-weight:600;color:var(--text);">${s.name}</td>
            <td><span class="badge badge-pink">${s.count} sampel</span></td>
            <td>${ok
              ? '<span class="badge badge-green"><i data-lucide="check-circle-2"></i>&nbsp;Memenuhi Syarat</span>'
              : '<span class="badge badge-red"><i data-lucide="alert-circle"></i>&nbsp;Kurang Sampel (min. 2)</span>'}
            </td>
          </tr>`;
        }).join("");
        if (window.lucide) lucide.createIcons({ nodes: [tbody] });
      }
    }
  } catch {}
}

// ─────────────────────────────────────────────────────────
// INIT
// ─────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  loadDataset();
  loadDatasetStats();
  loadDatasetSummary();
});

