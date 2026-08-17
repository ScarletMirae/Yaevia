/**
 * verify.js — Logika Halaman Verifikasi
 * Yae Miko Theme | Sistem Verifikasi Tulisan Tangan
 * =======================================
 * Update v2.0: Euclidean Distance similarity, status badge per threshold,
 *              analysis time, feature vector length, K info.
 */

const API = API_BASE;
let selectedFile = null;

// ─────────────────────────────────────────────────────────
// FILE DROP & PREVIEW
// ─────────────────────────────────────────────────────────
const dropZone  = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");

dropZone.addEventListener("dragover",  (e) => { e.preventDefault(); dropZone.classList.add("dragover"); });
dropZone.addEventListener("dragleave", ()  => dropZone.classList.remove("dragover"));
dropZone.addEventListener("click",     ()  => fileInput.click());
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

function handleFile(file) {
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    document.getElementById("preview-img").src = e.target.result;
    document.getElementById("preview-filename").textContent =
      `${file.name}  (${(file.size / 1024).toFixed(1)} KB)`;
    document.getElementById("preview-container").style.display = "block";
    document.getElementById("verify-btn").disabled = false;
    dropZone.style.display = "none";
    resetSteps();
  };
  reader.readAsDataURL(file);
  if (window.lucide) lucide.createIcons();
}

function removePreview() {
  selectedFile = null;
  fileInput.value = "";
  document.getElementById("preview-container").style.display = "none";
  document.getElementById("verify-btn").disabled = true;
  dropZone.style.display = "block";
  resetResult();
  resetSteps();
}

// ─────────────────────────────────────────────────────────
// PROCESS STEPS
// ─────────────────────────────────────────────────────────
const STEP_IDS = ["step-upload", "step-preprocess", "step-hog", "step-knn", "step-result"];

function resetSteps() {
  STEP_IDS.forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.className = "process-step";
  });
}

async function animateSteps() {
  for (let i = 0; i < STEP_IDS.length - 1; i++) {
    const el = document.getElementById(STEP_IDS[i]);
    if (el) el.className = "process-step active";
    await new Promise((r) => setTimeout(r, 620));
    if (el) el.className = "process-step done";
  }
}

// ─────────────────────────────────────────────────────────
// STATUS BADGE HELPER (Euclidean Distance Based)
// ─────────────────────────────────────────────────────────
function getStatusMeta(status) {
  const s = (status || "").toUpperCase();
  if (s.includes("SANGAT MIRIP"))  return { cls: "badge-verified",   icon: "shield-check",   color: "#1a5c3a" };
  if (s.includes("MIRIP"))         return { cls: "badge-mirip",      icon: "check-circle-2", color: "#7A5500" };
  if (s.includes("KURANG MIRIP"))  return { cls: "badge-uncertain",  icon: "help-circle",    color: "#7A3A00" };
  return                                  { cls: "badge-unverified", icon: "x-circle",       color: "#7A2E45" };
}

// ─────────────────────────────────────────────────────────
// VERIFICATION
// ─────────────────────────────────────────────────────────
async function doVerify() {
  if (!selectedFile) { showToast("Pilih gambar terlebih dahulu", "error"); return; }

  const btn     = document.getElementById("verify-btn");
  btn.disabled  = true;
  btn.innerHTML = `<span class="spinner" style="width:16px;height:16px;border-width:2px;"></span> Memverifikasi...`;

  // Show result panel in loading state
  document.getElementById("result-placeholder").style.display = "none";
  document.getElementById("result-panel").style.display = "block";
  document.getElementById("result-badge").className = "result-status-badge badge-uncertain";
  document.getElementById("result-badge").innerHTML = `<i data-lucide="loader-2" class="spin-icon"></i> Memproses...`;
  document.getElementById("result-name").textContent = "—";
  document.getElementById("result-score-text").textContent = "0%";
  document.getElementById("result-bar").style.width = "0%";
  document.getElementById("top-matches-list").innerHTML = "";

  // Clear meta info if present
  const metaEl = document.getElementById("result-meta-info");
  if (metaEl) metaEl.innerHTML = "";

  if (window.lucide) lucide.createIcons();

  // Animate process steps while waiting
  animateSteps();

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const res  = await fetch(API + "/api/verify", { method: "POST", body: formData });
    const data = await res.json();

    // Complete last step
    const last = document.getElementById(STEP_IDS[STEP_IDS.length - 1]);
    if (last) last.className = "process-step done";

    if (!data.success) {
      showToast(data.message || "Verifikasi gagal", "error");
      document.getElementById("result-badge").className = "result-status-badge badge-unverified";
      document.getElementById("result-badge").innerHTML =
        `<i data-lucide="circle-x"></i> ${data.message || "Gagal"}`;
      if (window.lucide) lucide.createIcons();
      resetBtn();
      return;
    }

    renderResult(data);
    showToast("Verifikasi selesai!", "success");

  } catch {
    const last = document.getElementById(STEP_IDS[STEP_IDS.length - 1]);
    if (last) last.className = "process-step done";
    showToast("Tidak dapat terhubung ke server Flask", "error");
    document.getElementById("result-badge").className = "result-status-badge badge-unverified";
    document.getElementById("result-badge").innerHTML =
      `<i data-lucide="wifi-off"></i> Server tidak aktif`;
    if (window.lucide) lucide.createIcons();
  }

  resetBtn();
}

function resetBtn() {
  const btn = document.getElementById("verify-btn");
  btn.disabled = false;
  btn.innerHTML = `<i data-lucide="search"></i><span id="verify-btn-text">Verifikasi Sekarang</span>`;
  if (window.lucide) lucide.createIcons({ nodes: [btn] });
}

// ─────────────────────────────────────────────────────────
// RENDER RESULT (Updated v2.0)
// ─────────────────────────────────────────────────────────
function renderResult(data) {
  const pct      = parseFloat(data.similarity_percent || 0);
  const dist     = parseFloat(data.euclidean_distance || 0);
  const status   = data.similarity_status || data.verification_status || "TIDAK MIRIP";
  const time     = parseFloat(data.analysis_time_seconds || 0);
  const featLen  = data.feature_vector_length || 0;
  const kVal     = data.k_neighbors || 5;

  // --- Status badge (Euclidean Distance based) ---
  const meta  = getStatusMeta(status);
  const badge = document.getElementById("result-badge");
  badge.className = `result-status-badge ${meta.cls}`;
  badge.innerHTML = `<i data-lucide="${meta.icon}"></i> ${status}`;

  // --- Name & score ---
  document.getElementById("result-name").textContent = data.predicted_name || "—";
  document.getElementById("result-score-text").textContent = pct.toFixed(1) + "%";

  // --- Animate similarity bar ---
  setTimeout(() => {
    document.getElementById("result-bar").style.width = Math.min(pct, 100) + "%";
  }, 300);

  // --- Extra meta info row (Euclidean Distance, analysis time, etc.) ---
  const metaEl = document.getElementById("result-meta-info");
  if (metaEl) {
    metaEl.innerHTML = `
      <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:0.5rem;margin-top:0.85rem;">
        ${metaChip("Euclidean Distance", dist.toFixed(4), "ruler")}
        ${metaChip("Similarity Score",   pct.toFixed(2) + "%", "percent")}
        ${metaChip("Nilai K",            kVal, "git-merge")}
        ${metaChip("Feature Vector",     featLen.toLocaleString() + " dim", "bar-chart-2")}
        ${metaChip("Waktu Analisis",     time.toFixed(3) + " detik", "clock")}
      </div>`;
    if (window.lucide) lucide.createIcons({ nodes: [metaEl] });
  }

  // --- Top matches (with Euclidean Distance) ---
  const matches = data.top_matches || [];
  document.getElementById("top-matches-list").innerHTML = matches.map((m, i) => {
    const isTop  = i === 0;
    const pctVal = typeof m.percent === "number" ? m.percent.toFixed(1) : "—";
    const dVal   = typeof m.distance === "number" ? m.distance.toFixed(4) : "—";
    return `
    <div style="display:flex;align-items:center;gap:0.75rem;padding:0.55rem 0.8rem;
         background:${isTop ? "linear-gradient(135deg,var(--soft),var(--cream))" : "transparent"};
         border-radius:var(--radius-sm);margin-bottom:0.35rem;
         border:1px solid ${isTop ? "var(--pink)" : "transparent"};
         transition:all var(--t-fast);">
      <span style="font-size:0.78rem;font-weight:800;color:var(--text-muted);min-width:20px;text-align:center;">${i + 1}</span>
      <i data-lucide="${isTop ? "user-check" : "user"}" style="width:14px;height:14px;color:${isTop ? "var(--rose-gold)" : "var(--text-muted)"};flex-shrink:0;"></i>
      <span style="flex:1;font-weight:${isTop ? "700" : "500"};color:var(--text);font-size:0.87rem;">${m.name}</span>
      <div style="text-align:right;">
        <div style="font-weight:800;color:${isTop ? "var(--text)" : "var(--text-muted)"};font-size:${isTop ? "0.95rem" : "0.85rem"};">${pctVal}%</div>
        <div style="font-size:0.68rem;color:var(--text-muted);">d=${dVal}</div>
      </div>
    </div>`;
  }).join("");

  if (window.lucide) lucide.createIcons();
}

function metaChip(label, val, icon) {
  return `
    <div style="background:var(--white);border:1.5px solid rgba(200,155,110,0.18);border-radius:var(--radius-sm);
                padding:0.55rem 0.75rem;display:flex;align-items:center;gap:0.5rem;">
      <i data-lucide="${icon}" style="width:13px;height:13px;color:var(--rose-gold);flex-shrink:0;"></i>
      <div>
        <div style="font-size:0.65rem;color:var(--text-muted);font-weight:600;">${label}</div>
        <div style="font-size:0.88rem;font-weight:800;color:var(--text);">${val}</div>
      </div>
    </div>`;
}

// ─────────────────────────────────────────────────────────
// RESET
// ─────────────────────────────────────────────────────────
function resetResult() {
  document.getElementById("result-panel").style.display = "none";
  document.getElementById("result-placeholder").style.display = "block";
}

function resetVerify() {
  removePreview();
  resetResult();
}
