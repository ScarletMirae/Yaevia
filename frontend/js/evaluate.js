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

    renderMetrics(data.metrics);
    renderModelParams(data.model_info);
    renderSplitStats(data.model_info);
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
// RENDER METRICS CHIPS
// ─────────────────────────────────────────────────────────
function renderMetrics(m) {
  const fmt = (v) => v != null ? parseFloat(v).toFixed(2) + "%" : "—";
  document.getElementById("val-accuracy").textContent  = fmt(m.test_accuracy);
  document.getElementById("val-precision").textContent = fmt(m.precision_macro);
  document.getElementById("val-recall").textContent    = fmt(m.recall_macro);
  document.getElementById("val-f1").textContent        = fmt(m.f1_macro);
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
        <i data-lucide="git-merge" style="width:11px;height:11px;"></i> KNN
      </p>
      ${paramRow("Nilai K", info.knn_k)}
      ${paramRow("Metric Jarak", info.knn_metric || "euclidean")}
      ${paramRow("Feature Vector Length", (info.feature_vector_size || 0).toLocaleString() + " dimensi")}
    </div>
    <div>
      <p style="font-size:0.7rem;font-weight:700;color:var(--purple);letter-spacing:0.06em;text-transform:uppercase;margin-bottom:0.35rem;">
        <i data-lucide="bar-chart-2" style="width:11px;height:11px;"></i> HOG
      </p>
      ${paramRow("Orientations", info.hog_orientations)}
      ${paramRow("Pixels per Cell", ppc)}
      ${paramRow("Cells per Block", cpb)}
    </div>
    <div style="margin-top:0.75rem;padding-top:0.75rem;border-top:1px solid var(--pink);">
      ${paramRow("Waktu Training", ttm)}
      ${paramRow("Tanggal Training", ts)}
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
// RENDER SPLIT STATS
// ─────────────────────────────────────────────────────────
function renderSplitStats(info) {
  const el = document.getElementById("split-content");
  if (!info) { el.textContent = "Tidak tersedia."; return; }

  const train_pct = Math.round((1 - (info.test_size || 0.2)) * 100);
  const test_pct  = Math.round((info.test_size || 0.2) * 100);

  el.innerHTML = `
    <div style="display:flex;gap:1rem;margin-bottom:1.25rem;">
      <div style="flex:1;text-align:center;background:linear-gradient(135deg,var(--soft),var(--cream));border-radius:var(--radius-sm);padding:1rem;border:1px solid var(--pink);">
        <div style="font-family:'Quicksand',sans-serif;font-size:2rem;font-weight:800;color:var(--text);">${info.n_respondents ?? "—"}</div>
        <div style="font-size:0.72rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:.04em;">Responden</div>
      </div>
      <div style="flex:1;text-align:center;background:linear-gradient(135deg,var(--soft),var(--cream));border-radius:var(--radius-sm);padding:1rem;border:1px solid var(--pink);">
        <div style="font-family:'Quicksand',sans-serif;font-size:2rem;font-weight:800;color:var(--text);">${info.n_total_dataset ?? "—"}</div>
        <div style="font-size:0.72rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:.04em;">Total Dataset</div>
      </div>
    </div>
    <div style="display:flex;gap:1rem;">
      <div style="flex:1;text-align:center;background:linear-gradient(135deg,rgba(107,63,160,0.08),rgba(107,63,160,0.03));border-radius:var(--radius-sm);padding:0.85rem;border:1px solid rgba(107,63,160,0.2);">
        <div style="font-family:'Quicksand',sans-serif;font-size:1.6rem;font-weight:800;color:var(--purple);">${info.n_train_samples ?? "—"}</div>
        <div style="font-size:0.72rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:.04em;">Training (${train_pct}%)</div>
      </div>
      <div style="flex:1;text-align:center;background:linear-gradient(135deg,rgba(212,163,115,0.12),rgba(212,163,115,0.04));border-radius:var(--radius-sm);padding:0.85rem;border:1px solid rgba(212,163,115,0.3);">
        <div style="font-family:'Quicksand',sans-serif;font-size:1.6rem;font-weight:800;color:var(--rose-gold);">${info.n_test_samples ?? "—"}</div>
        <div style="font-size:0.72rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:.04em;">Testing (${test_pct}%)</div>
      </div>
    </div>`;
}

// ─────────────────────────────────────────────────────────
// RENDER CHARTS (Chart.js)
// ─────────────────────────────────────────────────────────
function renderCharts(perClass) {
  if (!perClass || !perClass.length) return;

  const labels     = perClass.map(c => truncateLabel(c.name, 14));
  const trainData  = perClass.map(c => c.train);
  const testData   = perClass.map(c => c.test);
  const totalData  = perClass.map(c => c.total);

  const chartFont = { family: "'Poppins', sans-serif", size: 11 };
  const gridColor = "rgba(247,198,217,0.4)";

  // Destroy existing
  if (chartTrainTest) { chartTrainTest.destroy(); chartTrainTest = null; }
  if (chartSamples)   { chartSamples.destroy();   chartSamples   = null; }

  // Chart 1: Training vs Testing per Mahasiswa
  const ctx1 = document.getElementById("chart-train-test").getContext("2d");
  chartTrainTest = new Chart(ctx1, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label:           "Data Training",
          data:            trainData,
          backgroundColor: COLORS.purple,
          borderColor:     COLORS.purpleBorder,
          borderWidth:     1.5,
          borderRadius:    6,
        },
        {
          label:           "Data Testing",
          data:            testData,
          backgroundColor: COLORS.gold,
          borderColor:     COLORS.goldBorder,
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

  // Chart 2: Distribusi Sampel per Mahasiswa
  const ctx2 = document.getElementById("chart-samples").getContext("2d");
  chartSamples = new Chart(ctx2, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label:           "Jumlah Sampel",
          data:            totalData,
          backgroundColor: perClass.map((_, i) =>
            i % 2 === 0 ? COLORS.pink : COLORS.gold),
          borderColor:     perClass.map((_, i) =>
            i % 2 === 0 ? COLORS.pinkBorder : COLORS.goldBorder),
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
            label: (ctx) => ` ${ctx.parsed.y} sampel`,
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
    tbody.innerHTML = `<tr><td colspan="5" style="text-align:center;padding:2rem;color:var(--text-muted);">Tidak ada data.</td></tr>`;
    return;
  }

  tbody.innerHTML = perClass.map((c, i) => `
    <tr>
      <td style="font-weight:700;color:var(--text);">${i + 1}</td>
      <td style="font-weight:600;color:var(--text);">${c.name}</td>
      <td><span class="badge badge-pink">${c.total}</span></td>
      <td><span class="badge badge-purple">${c.train}</span></td>
      <td><span class="badge badge-gold">${c.test}</span></td>
    </tr>`).join("");
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
