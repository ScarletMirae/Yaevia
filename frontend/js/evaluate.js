/**
 * evaluate.js — Logika Halaman Evaluasi Model
 * Yae Miko Theme | Sistem Verifikasi Tulisan Tangan
 * =============================================
 * Menampilkan:
 *   - Accuracy, Precision, Recall, F1 Score
 *   - Chart: Training vs Testing per mahasiswa
 *   - Chart: Distribusi sampel per mahasiswa
 *   - Confusion Matrix
 *   - Tabel detail per kelas
 */

const API = API_BASE;

// Warna Yae Miko untuk chart
const COLORS = {
  pink:    "rgba(247, 198, 217, 0.85)",
  pinkBorder: "#C89B6E",
  purple:  "rgba(107, 63, 160, 0.75)",
  purpleBorder: "#6B3FA0",
  gold:    "rgba(212, 163, 115, 0.8)",
  goldBorder: "#B8864E",
  cream:   "#FFF8F3",
};

// Chart instances (untuk destroy sebelum re-render)
let chartTrainTest = null;
let chartSamples   = null;

// ─────────────────────────────────────────────────────────
// MAIN LOAD
// ─────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", loadEvaluation);

async function loadEvaluation() {
  const alertEl = document.getElementById("no-model-alert");
  const section  = document.getElementById("metrics-section");

  // Show loading
  alertEl.innerHTML = `
    <div class="alert alert-info animate-fade-in">
      <span class="spinner"></span>
      <span>Memuat data evaluasi model...</span>
    </div>`;

  try {
    const res  = await fetch(API + "/api/evaluate");
    const data = await res.json();

    if (!data.success) {
      alertEl.innerHTML = `
        <div class="alert alert-warning animate-fade-in">
          <i data-lucide="triangle-alert"></i>
          <span>${data.message || "Belum ada model terlatih."} <a href="upload.html" style="color:var(--text);font-weight:700;">Mulai training →</a></span>
        </div>`;
      if (window.lucide) lucide.createIcons({ nodes: [alertEl] });
      return;
    }

    alertEl.innerHTML = "";
    section.style.display = "block";

    renderLiveEvalBanner(data.live_verification_eval);
    renderMetrics(data.loocv_metrics || data.metrics);
    renderModelParams(data.model_info);
    renderArchitectureInfo(data.reference_dataset, data.loocv_metrics);
    renderCharts(data.per_class_chart);
    renderPerClassTable(data.per_class_chart);

    if (data.confusion_matrix) {
      renderConfusionMatrix(data.confusion_matrix);
    }

    if (window.lucide) lucide.createIcons();

  } catch (err) {
    alertEl.innerHTML = `
      <div class="alert alert-error animate-fade-in">
        <i data-lucide="wifi-off"></i>
        <span>Tidak dapat terhubung ke server. Pastikan <code>python app.py</code> sudah berjalan.</span>
      </div>`;
    if (window.lucide) lucide.createIcons({ nodes: [alertEl] });
  }
}

// ─────────────────────────────────────────────────────────
// RENDER METRICS CHIPS (LOOCV VALIDATION)
// ─────────────────────────────────────────────────────────
function renderMetrics(m) {
  const fmt = (v) => v != null ? parseFloat(v).toFixed(2) + "%" : "—";
  const accEl = document.getElementById("val-accuracy");
  const precEl = document.getElementById("val-precision");
  const recEl = document.getElementById("val-recall");
  const f1El = document.getElementById("val-f1");

  if (accEl)  accEl.textContent  = fmt(m.accuracy ?? m.test_accuracy);
  if (precEl) precEl.textContent = fmt(m.precision_macro);
  if (recEl)  recEl.textContent  = fmt(m.recall_macro);
  if (f1El)   f1El.textContent   = fmt(m.f1_macro);
}

// ─────────────────────────────────────────────────────────
// RENDER MODEL PARAMETERS
// ─────────────────────────────────────────────────────────
function renderModelParams(info) {
  const el = document.getElementById("model-params-content");
  if (!info) { el.textContent = "Tidak tersedia."; return; }

  const ppc = Array.isArray(info.hog_pixels_per_cell) ? info.hog_pixels_per_cell.join("×") : info.hog_pixels_per_cell;
  const cpb = Array.isArray(info.hog_cells_per_block) ? info.hog_cells_per_block.join("×") : info.hog_cells_per_block;
  const ts  = info.train_timestamp ? new Date(info.train_timestamp).toLocaleString("id-ID") : "—";
  const ttm = info.training_time ? info.training_time.toFixed(2) + " detik" : "—";

  el.innerHTML = `
    <div style="margin-bottom:0.75rem;">
      <p style="font-size:0.7rem;font-weight:700;color:var(--purple);letter-spacing:0.06em;text-transform:uppercase;margin-bottom:0.35rem;">
        <i data-lucide="git-merge" style="width:11px;height:11px;"></i> KNN Classifier (Frozen)
      </p>
      ${paramRow("Nilai K", info.knn_k || 5)}
      ${paramRow("Metric Jarak", info.knn_metric || "euclidean")}
      ${paramRow("Bobot Voting", info.knn_weights || "distance")}
      ${paramRow("Panjang Vektor Fitur", (info.feature_vector_size || 34596).toLocaleString() + " dimensi")}
    </div>
    <div>
      <p style="font-size:0.7rem;font-weight:700;color:var(--purple);letter-spacing:0.06em;text-transform:uppercase;margin-bottom:0.35rem;">
        <i data-lucide="bar-chart-2" style="width:11px;height:11px;"></i> HOG Feature Extractor (Frozen)
      </p>
      ${paramRow("Resolusi Citra", "256 × 256 piksel (Letterbox)")}
      ${paramRow("Orientations", info.hog_orientations || 9)}
      ${paramRow("Pixels per Cell", ppc || "8×8")}
      ${paramRow("Cells per Block", cpb || "2×2")}
      ${paramRow("Normalisasi Blok", info.hog_block_norm || "L2-Hys")}
    </div>
    <div style="margin-top:0.75rem;padding-top:0.75rem;border-top:1px solid var(--pink);">
      ${paramRow("Waktu Training", ttm)}
      ${paramRow("Status Model", "Dibekukan (H0 Produksi)")}
    </div>`;
}

function paramRow(label, val) {
  return `
    <div style="display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid var(--cream-alt);">
      <span style="color:var(--text-muted);font-size:0.82rem;">${label}</span>
      <span style="font-weight:700;color:var(--text);font-size:0.85rem;">${val ?? "—"}</span>
    </div>`;
}

// ─────────────────────────────────────────────────────────
// RENDER ARCHITECTURE & DATASET DETAILS
// ─────────────────────────────────────────────────────────
function renderArchitectureInfo(refInfo, loocvInfo) {
  const el = document.getElementById("split-content");
  if (!el) return;

  const nResp   = refInfo ? refInfo.n_respondents : 20;
  const nTotal  = refInfo ? refInfo.n_total_dataset : 400;
  const nCorr   = loocvInfo ? loocvInfo.correct_samples : 244;
  const nWrong  = loocvInfo ? loocvInfo.wrong_samples : 156;

  el.innerHTML = `
    <div style="display:flex;gap:1rem;margin-bottom:1rem;">
      <div style="flex:1;text-align:center;background:linear-gradient(135deg,var(--soft),var(--cream));border-radius:var(--radius-sm);padding:0.85rem;border:1px solid var(--pink);">
        <div style="font-family:'Quicksand',sans-serif;font-size:1.8rem;font-weight:800;color:var(--text);">${nResp}</div>
        <div style="font-size:0.72rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:.04em;">Responden</div>
      </div>
      <div style="flex:1;text-align:center;background:linear-gradient(135deg,var(--soft),var(--cream));border-radius:var(--radius-sm);padding:0.85rem;border:1px solid var(--pink);">
        <div style="font-family:'Quicksand',sans-serif;font-size:1.8rem;font-weight:800;color:var(--text);">${nTotal}</div>
        <div style="font-size:0.72rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:.04em;">Basis Data Referensi</div>
      </div>
    </div>
    <div style="display:flex;gap:1rem;margin-bottom:0.75rem;">
      <div style="flex:1;text-align:center;background:linear-gradient(135deg,rgba(107,63,160,0.08),rgba(107,63,160,0.03));border-radius:var(--radius-sm);padding:0.75rem;border:1px solid rgba(107,63,160,0.2);">
        <div style="font-family:'Quicksand',sans-serif;font-size:1.4rem;font-weight:800;color:var(--purple);">${nCorr} / ${nTotal}</div>
        <div style="font-size:0.72rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:.04em;">Prediksi Benar (LOOCV)</div>
      </div>
      <div style="flex:1;text-align:center;background:linear-gradient(135deg,rgba(212,163,115,0.12),rgba(212,163,115,0.04));border-radius:var(--radius-sm);padding:0.75rem;border:1px solid rgba(212,163,115,0.3);">
        <div style="font-family:'Quicksand',sans-serif;font-size:1.4rem;font-weight:800;color:var(--rose-gold);">${nWrong} / ${nTotal}</div>
        <div style="font-size:0.72rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:.04em;">Prediksi Salah (LOOCV)</div>
      </div>
    </div>
    <div style="padding:0.6rem 0.8rem;background:var(--white);border-radius:var(--radius-sm);border:1px solid rgba(200,155,110,0.2);font-size:0.78rem;line-height:1.5;">
      <div style="display:flex;justify-content:space-between;margin-bottom:2px;">
        <span style="color:var(--text-muted);">Metodologi Validasi:</span>
        <strong style="color:var(--purple);">LOOCV (400 Folds, Bebas Leakage)</strong>
      </div>
      <div style="display:flex;justify-content:space-between;">
        <span style="color:var(--text-muted);">Held-Out Test Set H0:</span>
        <strong style="color:var(--rose-gold);">61.25% (49/80 sampel)</strong>
      </div>
    </div>`;
}

// ─────────────────────────────────────────────────────────
// RENDER CHARTS (Chart.js)
// ─────────────────────────────────────────────────────────
function renderCharts(perClass) {
  if (!perClass || !perClass.length) return;

  const labels     = perClass.map(c => truncateLabel(c.name, 14));
  const accData    = perClass.map(c => c.accuracy);
  const totalData  = perClass.map(c => c.total);

  const chartFont = { family: "'Poppins', sans-serif", size: 11 };
  const gridColor = "rgba(247,198,217,0.4)";

  // Destroy existing
  if (chartTrainTest) { chartTrainTest.destroy(); chartTrainTest = null; }
  if (chartSamples)   { chartSamples.destroy();   chartSamples   = null; }

  // Chart 1: Akurasi LOOCV per Mahasiswa (%)
  const ctx1 = document.getElementById("chart-train-test").getContext("2d");
  chartTrainTest = new Chart(ctx1, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label:           "Akurasi LOOCV (%)",
          data:            accData,
          backgroundColor: accData.map(v => v >= 70 ? COLORS.purple : (v >= 50 ? COLORS.gold : "rgba(244,67,54,0.65)")),
          borderColor:     accData.map(v => v >= 70 ? COLORS.purpleBorder : (v >= 50 ? COLORS.goldBorder : "#c0392b")),
          borderWidth:     1.5,
          borderRadius:    6,
        },
      ],
    },
    options: {
      responsive:         true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { font: chartFont, color: "#7A2E45", padding: 16 },
        },
        tooltip: {
          callbacks: {
            label: (ctx) => ` Akurasi LOOCV: ${ctx.parsed.y}% (${perClass[ctx.dataIndex].correct}/${perClass[ctx.dataIndex].total} benar)`,
          },
        },
      },
      scales: {
        x: {
          ticks: { font: chartFont, color: "#B5607A", maxRotation: 45 },
          grid:  { color: gridColor },
        },
        y: {
          beginAtZero: true,
          max: 100,
          ticks: {
            font: chartFont,
            color: "#B5607A",
            callback: (v) => v + "%",
          },
          grid:  { color: gridColor },
        },
      },
    },
  });

  // Chart 2: Distribusi Sampel Dataset Referensi per Mahasiswa
  const ctx2 = document.getElementById("chart-samples").getContext("2d");
  chartSamples = new Chart(ctx2, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label:           "Jumlah Sampel Referensi",
          data:            totalData,
          backgroundColor: COLORS.pink,
          borderColor:     COLORS.pinkBorder,
          borderWidth:     1.5,
          borderRadius:    6,
        },
      ],
    },
    options: {
      responsive:          true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => ` ${ctx.parsed.y} sampel (Seimbang 20/mhs)`,
          },
        },
      },
      scales: {
        x: {
          ticks: { font: chartFont, color: "#B5607A", maxRotation: 45 },
          grid:  { color: gridColor },
        },
        y: {
          beginAtZero: true,
          ticks: { font: chartFont, color: "#B5607A", precision: 0 },
          grid:  { color: gridColor },
        },
      },
    },
  });
}

// ─────────────────────────────────────────────────────────
// RENDER PER-CLASS TABLE
// ─────────────────────────────────────────────────────────
function renderPerClassTable(perClass) {
  const tbody = document.getElementById("per-class-tbody");
  if (!perClass || !perClass.length) {
    tbody.innerHTML = `<tr><td colspan="6" style="text-align:center;padding:2rem;color:var(--text-muted);">Tidak ada data.</td></tr>`;
    return;
  }

  tbody.innerHTML = perClass.map((c, i) => {
    const acc = parseFloat(c.accuracy || 0);
    const badgeClass = acc >= 70 ? "badge-green" : (acc >= 50 ? "badge-yellow" : "badge-red");
    return `
    <tr>
      <td style="font-weight:700;color:var(--text);">${i + 1}</td>
      <td style="font-weight:600;color:var(--text);">${c.name}</td>
      <td><span class="badge badge-pink">${c.total} sampel</span></td>
      <td><span class="badge badge-green">${c.correct} benar</span></td>
      <td><span class="badge badge-red">${c.wrong} salah</span></td>
      <td><span class="badge ${badgeClass}" style="font-weight:800;">${acc.toFixed(1)}%</span></td>
    </tr>`;
  }).join("");
}

// ─────────────────────────────────────────────────────────
// RENDER CONFUSION MATRIX
// ─────────────────────────────────────────────────────────
function renderConfusionMatrix(cmData) {
  const card  = document.getElementById("cm-card");
  const wrap  = document.getElementById("cm-table-wrap");
  const matrix = cmData.matrix;
  const labels = cmData.labels;

  if (!matrix || !labels || !matrix.length) return;
  card.style.display = "block";

  // Max value for color scaling
  const maxVal = Math.max(...matrix.flat().filter(v => v > 0));

  let html = `<table style="border-collapse:collapse;font-size:0.7rem;min-width:100%;">
    <thead><tr>
      <th style="padding:4px 6px;background:var(--soft);text-align:right;font-size:0.65rem;color:var(--text-muted);min-width:80px;">Aktual \\ Prediksi</th>
      ${labels.map(l => `<th style="padding:4px 6px;background:var(--soft);white-space:nowrap;font-size:0.68rem;color:var(--text);transform:rotate(-35deg);min-width:60px;text-align:center;" title="${l}">${truncateLabel(l, 10)}</th>`).join("")}
    </tr></thead>
    <tbody>`;

  matrix.forEach((row, r) => {
    html += `<tr>
      <td style="padding:4px 6px;background:var(--soft);font-weight:700;color:var(--text);white-space:nowrap;font-size:0.72rem;" title="${labels[r]}">${truncateLabel(labels[r], 12)}</td>
      ${row.map((val, c) => {
        const isDiag   = r === c;
        const opacity  = val > 0 ? 0.15 + (val / maxVal) * 0.75 : 0;
        const bgColor  = isDiag
          ? `rgba(107,63,160,${opacity})`
          : `rgba(247,198,217,${opacity})`;
        const txtColor = isDiag && val > 0 ? "var(--purple)" : val > 0 ? "var(--text)" : "var(--text-muted)";
        return `<td style="text-align:center;padding:4px 5px;background:${bgColor};color:${txtColor};font-weight:${isDiag && val > 0 ? "800" : "500"};font-size:0.8rem;">${val > 0 ? val : "·"}</td>`;
      }).join("")}
    </tr>`;
  });

  html += `</tbody></table>`;
  wrap.innerHTML = html;
}

// ─────────────────────────────────────────────────────────
// HELPERS
// ─────────────────────────────────────────────────────────
function truncateLabel(str, maxLen) {
  if (!str) return "";
  const parts = str.trim().split(" ");
  if (parts.length === 1) return str.substring(0, maxLen);
  // Nama depan saja
  return parts[0];
}

// ─────────────────────────────────────────────────────────
// LIVE EVALUATION BANNER
// ─────────────────────────────────────────────────────────
function renderLiveEvalBanner(liveEval) {
  const container = document.getElementById("live-eval-container");
  if (!container) return;

  if (!liveEval || !liveEval.has_records || liveEval.n_samples === 0) {
    container.innerHTML = `
      <div style="display:flex;align-items:center;gap:1rem;">
        <div style="width:42px;height:42px;border-radius:50%;background:rgba(212,163,115,0.15);color:var(--gold);display:flex;align-items:center;justify-content:center;flex-shrink:0;">
          <i data-lucide="info" style="width:22px;height:22px;"></i>
        </div>
        <div>
          <div style="font-weight:700;color:var(--text);font-size:0.95rem;margin-bottom:2px;">
            Status Data Pengujian Riil (Live Verification Records)
          </div>
          <div style="font-size:0.83rem;color:var(--text-muted);">
            Belum tersedia data pengujian berlabel yang cukup dari pengujian verifikasi pengguna.
            <a href="verify.html" style="color:var(--rose-gold);font-weight:700;">Lakukan Verifikasi Berlabel →</a>
          </div>
        </div>
      </div>`;
    if (window.lucide) lucide.createIcons({ nodes: [container] });
    return;
  }

  const acc = liveEval.accuracy;
  container.innerHTML = `
    <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:1rem;">
      <div style="display:flex;align-items:center;gap:1rem;">
        <div style="width:46px;height:46px;border-radius:50%;background:linear-gradient(135deg,var(--soft),var(--pink));color:var(--purple);display:flex;align-items:center;justify-content:center;flex-shrink:0;box-shadow:var(--shadow-gold);">
          <i data-lucide="check-square" style="width:24px;height:24px;"></i>
        </div>
        <div>
          <div style="font-size:0.75rem;font-weight:700;color:var(--purple);text-transform:uppercase;letter-spacing:0.05em;">Hasil Evaluasi Verifikasi Riwayat Berlabel</div>
          <div style="font-size:1.1rem;font-weight:800;color:var(--text);">${liveEval.description}</div>
        </div>
      </div>
      <div style="display:flex;gap:0.75rem;align-items:center;flex-wrap:wrap;">
        <div style="background:var(--white);padding:0.5rem 0.85rem;border-radius:var(--radius-sm);border:1px solid rgba(200,155,110,0.2);text-align:center;">
          <div style="font-size:0.68rem;color:var(--text-muted);font-weight:600;">Total Pengujian</div>
          <div style="font-size:1.05rem;font-weight:800;color:var(--text);">${liveEval.n_samples} citra</div>
        </div>
        <div style="background:var(--white);padding:0.5rem 0.85rem;border-radius:var(--radius-sm);border:1px solid rgba(200,155,110,0.2);text-align:center;">
          <div style="font-size:0.68rem;color:var(--text-muted);font-weight:600;">Benar / Salah</div>
          <div style="font-size:1.05rem;font-weight:800;color:var(--purple);">${liveEval.correct_count} / ${liveEval.wrong_count}</div>
        </div>
        <div style="background:linear-gradient(135deg,var(--soft),var(--cream));padding:0.5rem 1rem;border-radius:var(--radius-sm);border:1.5px solid var(--pink);text-align:center;">
          <div style="font-size:0.68rem;color:var(--text-muted);font-weight:700;text-transform:uppercase;">Live Accuracy</div>
          <div style="font-size:1.25rem;font-weight:800;color:var(--text);">${acc.toFixed(2)}%</div>
        </div>
      </div>
    </div>`;
  if (window.lucide) lucide.createIcons({ nodes: [container] });
}
