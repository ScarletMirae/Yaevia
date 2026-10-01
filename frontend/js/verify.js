console.log("[VERIFY] script loaded");

const getApiBase = () => {
  if (typeof API_BASE !== 'undefined' && API_BASE) return API_BASE;
  if (location.protocol === 'file:') return 'http://localhost:5000';
  return location.origin;
};

let selectedFile = null;
let selectedWriter = null;
let registeredWriters = [];

// Entry point initialization
document.addEventListener("DOMContentLoaded", () => {
  console.log("[VERIFY] DOMContentLoaded fired");
  
  initUploadEvents();
  loadClaimedWriters();
  checkFormValidity();
});

// ── Upload event wiring ──────────────────────────────────────────────────────
function initUploadEvents() {
  const dropZone  = document.getElementById("drop-zone");
  const fileInput = document.getElementById("file-input");

  if (!dropZone || !fileInput) {
    console.warn("[VERIFY] dropZone or fileInput not found in DOM!");
    return;
  }

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });

  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));

  dropZone.addEventListener("click", (e) => {
    if (e.target !== fileInput) fileInput.click();
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  });
}

// ── Claimed Writers Dropdown Loader ──────────────────────────────────────────
async function loadClaimedWriters() {
  const selectEl = document.getElementById("claimed-writer-select");
  if (!selectEl) {
    console.error("[VERIFY] claimed-writer-select element not found!");
    return;
  }

  // Show loading state in dropdown
  selectEl.innerHTML = '<option value="">⏳ Memuat daftar penulis...</option>';
  selectEl.disabled = true;

  try {
    const baseUrl = getApiBase();
    const url = baseUrl + "/api/verify/claimed-writers";
    console.log("[VERIFY] Fetching claimed writers from:", url);

    const res = await fetch(url);
    console.log("[VERIFY] claimed-writers HTTP status:", res.status);

    if (!res.ok) {
      throw new Error(`HTTP ${res.status} ${res.statusText}`);
    }

    const data = await res.json();
    console.log("[VERIFY] claimed-writers response:", data);

    if (data.success && Array.isArray(data.writers) && data.writers.length > 0) {
      registeredWriters = data.writers;
      selectEl.innerHTML = '<option value="">-- Pilih Penulis Terdaftar --</option>';
      data.writers.forEach(w => {
        const opt = document.createElement("option");
        opt.value = w.name;
        opt.textContent = w.nim ? `${w.name} (${w.nim})` : w.name;
        opt.dataset.nim = w.nim || "";
        selectEl.appendChild(opt);
      });
      selectEl.disabled = false;
      console.log(`[VERIFY] ✓ Loaded ${data.writers.length} registered writers into dropdown.`);
    } else {
      throw new Error(`Respons tidak valid: success=${data.success}, writers=${JSON.stringify(data.writers)}`);
    }
  } catch (err) {
    console.error("[VERIFY] ✗ Failed to load claimed writers:", err);
    selectEl.innerHTML = '<option value="">⚠ Gagal memuat. Periksa koneksi ke server.</option>';
    selectEl.disabled = false;
    // Show user-visible warning inline
    const checker = document.getElementById("identity-status-checker");
    if (checker) {
      checker.style.color = "#c0392b";
      checker.innerHTML = `<i data-lucide="wifi-off" style="width:14px;height:14px;flex-shrink:0;"></i>
        <span>Gagal memuat daftar penulis terdaftar. Pastikan server Flask berjalan dan refresh halaman.</span>`;
      if (window.lucide) lucide.createIcons({ nodes: [checker] });
    }
    if (typeof showToast === "function") {
      showToast("Gagal memuat daftar penulis. Periksa koneksi ke server.", "error");
    }
  }
}

function onClaimedWriterSelected() {
  const selectEl = document.getElementById("claimed-writer-select");
  const nimGroup = document.getElementById("claimed-nim-group");
  const nimDisplay = document.getElementById("claimed-nim-display");
  const checker = document.getElementById("identity-status-checker");

  if (!selectEl || !selectEl.value) {
    selectedWriter = null;
    if (nimGroup) nimGroup.style.display = "none";
    updateIdentityCheckerUI("default", "Pilih identitas mahasiswa yang diklaim sebagai pemilik tulisan.");
    checkFormValidity();
    return;
  }

  const selectedOpt = selectEl.options[selectEl.selectedIndex];
  const name = selectEl.value;
  const nim = selectedOpt.dataset.nim || "";

  selectedWriter = { name, nim };

  if (nimGroup && nimDisplay) {
    nimDisplay.value = nim || "-";
    nimGroup.style.display = "block";
  }

  updateIdentityCheckerUI("valid", `✓ Klaim identitas dipilih: ${name}`);
  checkFormValidity();
}

function handleFile(file) {
  console.log("[VERIFY] handleFile start", file ? file.name : null);
  if (!file) return;

  const dropZone = document.getElementById("drop-zone");
  const previewContainer = document.getElementById("preview-container");
  const previewImg = document.getElementById("preview-img");
  const previewFilename = document.getElementById("preview-filename");

  console.log("[VERIFY] preview elements:", {
    dropZone: !!dropZone,
    previewContainer: !!previewContainer,
    previewImg: !!previewImg,
    previewFilename: !!previewFilename
  });

  // Validasi format file — mendukung multiple extensions seperti .jpg.jpeg
  const fileName = file.name || "";
  const parts = fileName.split('.');
  const ext = parts.length > 1 ? parts[parts.length - 1].toLowerCase() : '';
  const allowed = ["jpg", "jpeg", "png", "bmp", "tiff"];
  if (!allowed.includes(ext) && (!file.type || !file.type.startsWith("image/"))) {
    showToast(`Format file .${ext} tidak didukung. Gunakan JPG, PNG, BMP, TIFF`, "error");
    return;
  }

  // Validasi ukuran (maks 16 MB)
  if (file.size > 16 * 1024 * 1024) {
    showToast("Ukuran file melebihi 16 MB", "error");
    return;
  }

  selectedFile = file;
  console.log("[VERIFY] selectedFile assigned:", selectedFile.name, `${(selectedFile.size / 1024).toFixed(1)} KB`);

  const reader = new FileReader();
  reader.onload = (e) => {
    console.log("[VERIFY] FileReader result ready");
    if (previewImg) previewImg.src = e.target.result;
    if (previewFilename) {
      previewFilename.textContent = `${file.name}  (${(file.size / 1024).toFixed(1)} KB)`;
    }
    if (previewContainer) previewContainer.style.display = "block";
    if (dropZone) dropZone.style.display = "none";
    console.log("[VERIFY] preview rendered successfully");
    checkFormValidity();
    resetSteps();
  };
  console.log("[VERIFY] FileReader start");
  reader.readAsDataURL(file);
  if (window.lucide) lucide.createIcons();
}

function removePreview() {
  selectedFile = null;
  const fileInput = document.getElementById("file-input");
  const dropZone = document.getElementById("drop-zone");
  const previewContainer = document.getElementById("preview-container");

  if (fileInput) fileInput.value = "";
  if (previewContainer) previewContainer.style.display = "none";
  if (dropZone) dropZone.style.display = "block";

  console.log("[VERIFY] removePreview executed. selectedFile reset to null.");
  checkFormValidity();
  resetResult();
  resetSteps();
}

function updateIdentityCheckerUI(state, text) {
  const checker = document.getElementById("identity-status-checker");
  if (!checker) return;

  let icon = "info";
  let color = "var(--text-muted)";

  if (state === "checking") {
    icon = "loader-2";
    color = "var(--purple)";
  } else if (state === "valid") {
    icon = "check-circle-2";
    color = "#1e8449";
  } else if (state === "invalid" || state === "error") {
    icon = "x-circle";
    color = "#c0392b";
  }

  checker.style.color = color;
  checker.innerHTML = `<i data-lucide="${icon}" style="width:14px;height:14px;flex-shrink:0;${state === 'checking' ? 'animation:spin 1s linear infinite;' : ''}"></i><span>${text}</span>`;
  if (window.lucide) lucide.createIcons({ nodes: [checker] });
}

function checkFormValidity() {
  const verifyBtn = document.getElementById("verify-btn");
  if (!verifyBtn) return;

  const canVerify = selectedFile !== null && selectedWriter !== null;
  verifyBtn.disabled = !canVerify;
}

async function doVerify() {
  if (!selectedFile || !selectedWriter) {
    showToast("Silakan pilih gambar dan identitas mahasiswa yang diklaim.", "warning");
    return;
  }

  const verifyBtn = document.getElementById("verify-btn");
  if (verifyBtn) {
    verifyBtn.disabled = true;
    verifyBtn.innerHTML = `<i data-lucide="loader-2" class="spin-icon"></i> <span>Memverifikasi Klaim...</span>`;
    if (window.lucide) lucide.createIcons({ nodes: [verifyBtn] });
  }

  document.getElementById("result-placeholder").style.display = "none";
  const resultPanel = document.getElementById("result-panel");
  resultPanel.style.display = "block";

  const heroCard = document.getElementById("verification-hero-card");
  if (heroCard) {
    heroCard.innerHTML = `
      <div style="background:var(--white);border:1.5px solid rgba(200,155,110,0.25);border-radius:var(--radius-md);padding:1.5rem;text-align:center;">
        <i data-lucide="loader-2" class="spin-icon" style="width:28px;height:28px;color:var(--rose-gold);margin-bottom:0.5rem;"></i>
        <div style="font-weight:700;color:var(--text);">Memproses Ekstraksi HOG dan Verifikasi Jarak...</div>
      </div>`;
    if (window.lucide) lucide.createIcons({ nodes: [heroCard] });
  }

  const STEP_IDS = ["step-upload", "step-preprocess", "step-hog", "step-knn", "step-result"];
  resetSteps();

  const setStepActive = (idx) => {
    for (let i = 0; i < STEP_IDS.length; i++) {
      const el = document.getElementById(STEP_IDS[i]);
      if (!el) continue;
      if (i < idx) el.className = "process-step done";
      else if (i === idx) el.className = "process-step active";
      else el.className = "process-step";
    }
  };

  setStepActive(0);
  await new Promise(r => setTimeout(r, 120));
  setStepActive(1);
  await new Promise(r => setTimeout(r, 120));
  setStepActive(2);
  await new Promise(r => setTimeout(r, 120));
  setStepActive(3);

  const formData = new FormData();
  formData.append("file", selectedFile);
  formData.append("claimed_writer", selectedWriter.name);
  formData.append("ground_truth_name", selectedWriter.name);
  formData.append("ground_truth_nim", selectedWriter.nim || "");

  try {
    const baseUrl = getApiBase();
    console.log("[VERIFY] Submitting payload with claimed_writer:", selectedWriter.name);
    const res = await fetch(baseUrl + "/api/verify", {
      method: "POST",
      body: formData
    });
    const data = await res.json();
    console.log("[VERIFY] Verification response received:", data);

    setStepActive(4);
    const lastStep = document.getElementById(STEP_IDS[STEP_IDS.length - 1]);
    if (lastStep) lastStep.className = "process-step done";

    if (!data.success) {
      showToast(data.message || "Verifikasi gagal", "error");
      if (heroCard) {
        heroCard.innerHTML = `
          <div class="result-status-badge badge-unverified" style="margin:0 auto;">
            <i data-lucide="circle-x"></i> ${data.message || "Gagal"}
          </div>`;
        if (window.lucide) lucide.createIcons();
      }
      resetBtn();
      return;
    }

    renderResult(data);
    showToast("Verifikasi klaim selesai!", "success");

  } catch (err) {
    console.error("[VERIFY] Verify Network Error:", err);
    const lastStep = document.getElementById(STEP_IDS[STEP_IDS.length - 1]);
    if (lastStep) lastStep.className = "process-step done";
    showToast("Tidak dapat terhubung ke server Flask", "error");
    if (heroCard) {
      heroCard.innerHTML = `
        <div class="result-status-badge badge-unverified" style="margin:0 auto;">
          <i data-lucide="wifi-off"></i> Server tidak aktif
        </div>`;
      if (window.lucide) lucide.createIcons();
    }
  }

  resetBtn();
}

function resetBtn() {
  checkFormValidity();
  const btn = document.getElementById("verify-btn");
  if (btn && !btn.disabled) {
    btn.innerHTML = `<i data-lucide="search"></i><span id="verify-btn-text">Verifikasi Klaim Identitas</span>`;
    if (window.lucide) lucide.createIcons({ nodes: [btn] });
  }
}

function renderResult(data) {
  const verif = data.verification || {};
  const isAccepted = verif.decision === "ACCEPT" || verif.status === "VALID";
  const verifScore = typeof verif.score === "number" ? verif.score.toFixed(4) : "—";
  const verifThreshold = typeof verif.threshold === "number" ? verif.threshold.toFixed(4) : "25.1291";
  const claimedName = verif.claimed_writer || data.ground_truth_name || "—";
  const claimedNim = data.ground_truth_nim || (selectedWriter ? selectedWriter.nim : "—");
  const neighborDists = Array.isArray(verif.neighbor_distances) ? verif.neighbor_distances : [];

  const heroCard = document.getElementById("verification-hero-card");
  if (heroCard) {
    heroCard.innerHTML = `
      <div style="background:${isAccepted ? 'linear-gradient(135deg,rgba(212,245,233,0.45),rgba(255,255,255,0.98))' : 'linear-gradient(135deg,rgba(255,229,229,0.5),rgba(255,255,255,0.98))'};
                  border:2px solid ${isAccepted ? '#2ecc71' : '#e74c3c'};
                  border-radius:var(--radius-md);padding:1.4rem;box-shadow:0 6px 18px rgba(0,0,0,0.06);margin-bottom:1.25rem;">
        
        <!-- Header & Badge -->
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:1.1rem;padding-bottom:0.85rem;border-bottom:1px solid rgba(0,0,0,0.08);flex-wrap:wrap;gap:0.75rem;">
          <div style="display:flex;align-items:center;gap:0.65rem;">
            <div style="width:42px;height:42px;border-radius:50%;background:${isAccepted ? '#2ecc71' : '#e74c3c'};color:white;display:flex;align-items:center;justify-content:center;font-weight:bold;font-size:1.35rem;box-shadow:0 3px 8px ${isAccepted ? 'rgba(46,204,113,0.35)' : 'rgba(231,76,60,0.35)'};">
              ${isAccepted ? '✓' : '✕'}
            </div>
            <div>
              <div style="font-size:0.72rem;font-weight:700;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.06em;">Status Verifikasi Klaim (1-to-1 Mode)</div>
              <div style="font-size:1.25rem;font-weight:900;color:${isAccepted ? '#1e8449' : '#c0392b'};letter-spacing:0.02em;">
                ${isAccepted ? 'VALID (ACCEPT)' : 'TIDAK VALID (REJECT)'}
              </div>
            </div>
          </div>
          <span style="font-size:0.78rem;font-weight:800;padding:0.35rem 0.85rem;border-radius:20px;background:${isAccepted ? '#d4f5e9' : '#ffe5e5'};color:${isAccepted ? '#1a5c3a' : '#7a2e45'};border:1px solid ${isAccepted ? 'rgba(46,204,113,0.4)' : 'rgba(231,76,60,0.4)'};">
            ${isAccepted ? 'Claim Verified' : 'Claim Rejected'}
          </span>
        </div>

        <!-- Identity & Comparison Grid -->
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.9rem;text-align:left;margin-bottom:1rem;">
          <!-- Claimed Identity Card -->
          <div style="background:var(--white);padding:0.85rem 1rem;border-radius:var(--radius-sm);border:1px solid rgba(0,0,0,0.08);">
            <div style="font-size:0.7rem;font-weight:700;color:var(--rose-gold);text-transform:uppercase;margin-bottom:0.35rem;display:flex;align-items:center;gap:0.35rem;">
              <i data-lucide="user-check" style="width:13px;height:13px;"></i> Identitas yang Diklaim
            </div>
            <div style="font-size:1rem;font-weight:800;color:var(--text);">${claimedName}</div>
            <div style="font-size:0.78rem;color:var(--text-muted);margin-top:2px;">NIM: ${claimedNim || '—'}</div>
          </div>

          <!-- Score & Threshold Card -->
          <div style="background:var(--white);padding:0.85rem 1rem;border-radius:var(--radius-sm);border:1px solid rgba(0,0,0,0.08);">
            <div style="font-size:0.7rem;font-weight:700;color:var(--purple);text-transform:uppercase;margin-bottom:0.35rem;display:flex;align-items:center;gap:0.35rem;">
              <i data-lucide="gauge" style="width:13px;height:13px;"></i> Jarak vs Batas Ambang (EER)
            </div>
            <div style="font-size:0.88rem;font-weight:700;color:var(--text);">
              Skor Jarak: <span style="font-size:1.05rem;font-weight:900;color:${isAccepted ? '#1e8449' : '#c0392b'};">${verifScore}</span>
            </div>
            <div style="font-size:0.76rem;color:var(--text-muted);margin-top:2px;">
              Threshold: <strong>${verifThreshold}</strong> &bull; Rule: <em>d &le; ${verifThreshold}</em>
            </div>
          </div>
        </div>

        <!-- 5-Neighbor Breakdown to Claimed Writer -->
        ${neighborDists.length > 0 ? `
        <div style="background:rgba(255,255,255,0.7);padding:0.75rem 0.9rem;border-radius:var(--radius-sm);border:1px solid rgba(0,0,0,0.06);text-align:left;">
          <div style="font-size:0.72rem;font-weight:700;color:var(--text-soft);margin-bottom:0.4rem;display:flex;align-items:center;gap:0.35rem;">
            <i data-lucide="list-ordered" style="width:12px;height:12px;color:var(--rose-gold);"></i> 5 Jarak Euclidean Terdekat ke Sampel Penulis yang Diklaim:
          </div>
          <div style="display:flex;gap:0.45rem;flex-wrap:wrap;">
            ${neighborDists.map((d, idx) => `
              <span style="font-size:0.75rem;font-weight:700;font-family:monospace;background:var(--white);padding:0.25rem 0.55rem;border-radius:4px;border:1px solid rgba(200,155,110,0.25);color:var(--text);">
                #${idx + 1}: ${typeof d === "number" ? d.toFixed(4) : d}
              </span>
            `).join("")}
            <span style="font-size:0.72rem;font-weight:600;color:var(--text-muted);align-self:center;margin-left:auto;">
              Rata-rata Top-5: <strong>${verifScore}</strong> (Lower = Closer)
            </span>
          </div>
        </div>` : ''}

        <!-- Method Attribution Note -->
        <div style="font-size:0.68rem;color:var(--text-muted);margin-top:0.65rem;text-align:center;">
          Metode: <em>Mean Top-5 Claimed Euclidean Distance</em> &bull; Sumber Threshold: <em>Experiment K1 estimated EER operating point</em>
        </div>
      </div>
    `;
  }

  // Supporting 1-to-N Identification
  const pct          = parseFloat(data.similarity_percent || 0);
  const votePct      = parseFloat(
    data.predicted_vote_weight ?? 
    (data.top_matches && data.top_matches[0] && (data.top_matches[0].vote_percent ?? (data.top_matches[0].vote_weight ? data.top_matches[0].vote_weight * 100 : 0))) ?? 
    0
  );
  const dist         = parseFloat(data.euclidean_distance || 0);
  const simStatus    = data.similarity_status || "TIDAK MIRIP";
  const time         = parseFloat(data.analysis_time_seconds || 0);
  const featLen      = data.feature_vector_length || 0;
  const kVal         = data.k_neighbors || 5;

  const scoreText = document.getElementById("result-score-text");
  if (scoreText) scoreText.textContent = pct.toFixed(1) + "%";

  setTimeout(() => {
    const bar = document.getElementById("result-bar");
    if (bar) bar.style.width = Math.min(pct, 100) + "%";
  }, 300);

  const metaEl = document.getElementById("result-meta-info");
  if (metaEl) {
    metaEl.innerHTML = `
      <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:0.5rem;margin-top:0.85rem;">
        ${metaChip("Prediksi KNN (Top-1)", data.predicted_name || "—", "award")}
        ${metaChip("KNN Vote Share",       votePct.toFixed(2) + "%", "git-merge")}
        ${metaChip("Sample Similarity",    pct.toFixed(2) + "%", "percent")}
        ${metaChip("Min Euclidean Dist",   dist.toFixed(4), "ruler")}
        ${metaChip("Nilai K",              kVal, "users")}
        ${metaChip("Waktu Analisis",       time.toFixed(3) + " detik", "clock")}
      </div>`;
  }

  const matches = data.top_matches || [];
  let html = matches.map((m, i) => {
    const isTop   = i === 0;
    const simVal  = typeof m.percent === "number" ? m.percent.toFixed(1) : "—";
    const dVal    = typeof m.distance === "number" ? m.distance.toFixed(4) : "—";
    const vVal    = typeof m.vote_percent === "number" 
                    ? m.vote_percent.toFixed(1) 
                    : (typeof m.vote_weight === "number" ? (m.vote_weight * 100).toFixed(1) : "0.0");
    const kCount  = m.neighbor_count !== undefined ? `${m.neighbor_count}/${kVal} tetangga` : '';
    
    return `
    <div style="display:flex;align-items:center;gap:0.75rem;padding:0.6rem 0.85rem;
         background:${isTop ? "linear-gradient(135deg,rgba(242,167,195,0.18),rgba(255,248,240,0.9))" : "var(--white)"};
         border-radius:var(--radius-sm);margin-bottom:0.45rem;
         border:1.5px solid ${isTop ? "var(--pink)" : "rgba(200,155,110,0.15)"};
         box-shadow:${isTop ? "0 2px 8px rgba(184,80,110,0.08)" : "none"};
         transition:all var(--t-fast);">
      <span style="font-size:0.8rem;font-weight:800;color:${isTop ? "var(--rose-gold)" : "var(--text-muted)"};min-width:20px;text-align:center;">${i + 1}</span>
      <div style="width:28px;height:28px;border-radius:50%;background:${isTop ? "var(--soft)" : "rgba(240,230,220,0.4)"};display:flex;align-items:center;justify-content:center;flex-shrink:0;">
        <i data-lucide="${isTop ? "user-check" : "user"}" style="width:14px;height:14px;color:${isTop ? "var(--rose-gold)" : "var(--text-muted)"};"></i>
      </div>
      <div style="flex:1;min-width:0;">
        <div style="font-weight:${isTop ? "800" : "600"};color:var(--text);font-size:0.88rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
          ${m.name}
        </div>
        <div style="display:flex;align-items:center;gap:0.4rem;margin-top:0.2rem;flex-wrap:wrap;">
          <span style="font-size:0.7rem;font-weight:700;color:var(--purple);background:rgba(107,63,160,0.08);padding:0.1rem 0.4rem;border-radius:4px;display:inline-flex;align-items:center;gap:0.25rem;">
            <i data-lucide="git-merge" style="width:10px;height:10px;"></i> KNN Vote: ${vVal}%
          </span>
          ${kCount ? `<span style="font-size:0.7rem;font-weight:600;color:var(--text-muted);background:var(--soft);padding:0.1rem 0.4rem;border-radius:4px;">${kCount}</span>` : ''}
        </div>
      </div>
      <div style="text-align:right;flex-shrink:0;">
        <div style="font-size:0.65rem;color:var(--text-muted);font-weight:600;letter-spacing:0.02em;">Kemiripan Sampel</div>
        <div style="font-weight:${isTop ? "var(--text)" : "var(--text-muted)"};font-size:0.92rem;">${simVal}%</div>
        <div style="font-size:0.68rem;color:var(--text-muted);font-family:monospace;">d=${dVal}</div>
      </div>
    </div>`;
  }).join("");

  const kDetails = data.k_neighbors_detail || [];
  if (kDetails.length > 0) {
    html += `
    <div style="margin-top:1rem;padding:0.75rem;background:rgba(255,248,240,0.7);border:1px solid rgba(200,155,110,0.2);border-radius:var(--radius-sm);">
      <div style="font-size:0.75rem;font-weight:700;color:var(--purple);margin-bottom:0.4rem;display:flex;align-items:center;gap:0.35rem;">
        <i data-lucide="git-commit" style="width:12px;height:12px;"></i> Rincian ${kVal} Tetangga Terdekat (K-NN Breakdown)
      </div>
      <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:0.4rem;">
        ${kDetails.map(n => `
          <div style="background:var(--white);padding:0.35rem 0.55rem;border-radius:4px;border:1px solid rgba(200,155,110,0.15);font-size:0.73rem;">
            <div style="font-weight:700;color:var(--text);display:flex;justify-content:space-between;">
              <span>#${n.rank} ${n.name}</span>
              <span style="color:var(--purple);">${n.vote_contrib_pct}%</span>
            </div>
            <div style="color:var(--text-muted);display:flex;justify-content:space-between;margin-top:2px;">
              <span>d = ${n.distance}</span>
              <span>sim: ${n.similarity_percent}%</span>
            </div>
          </div>
        `).join("")}
      </div>
    </div>`;
  }

  const topList = document.getElementById("top-matches-list");
  if (topList) topList.innerHTML = html;
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

function resetSteps() {
  const STEP_IDS = ["step-upload", "step-preprocess", "step-hog", "step-knn", "step-result"];
  STEP_IDS.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.className = "process-step";
  });
}

function resetResult() {
  const panel = document.getElementById("result-panel");
  const holder = document.getElementById("result-placeholder");
  if (panel) panel.style.display = "none";
  if (holder) holder.style.display = "block";
}

function resetVerify() {
  removePreview();
  resetResult();
}
