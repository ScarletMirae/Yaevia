/**
 * history.js — Logika Halaman Riwayat Verifikasi
 * Yae Miko Theme | Sistem Verifikasi Tulisan Tangan
 * ================================================
 */

const API = API_BASE;
const PAGE_SIZE  = 20;
let currentPage  = 0;
let totalRecords = 0;
let allHistory   = [];

// ─────────────────────────────────────────────────────────
// LOAD HISTORY
// ─────────────────────────────────────────────────────────
async function loadHistory() {
  const tbody = document.getElementById("history-tbody");
  tbody.innerHTML = `<tr><td colspan="8" style="text-align:center;padding:2rem;color:var(--text-muted);">
    <span class="spinner"></span>&nbsp; Memuat riwayat...
  </td></tr>`;

  try {
    const res  = await fetch(`${API}/api/verify/history?limit=${PAGE_SIZE}&offset=${currentPage * PAGE_SIZE}`);
    const data = await res.json();
    totalRecords = data.total || 0;
    allHistory   = data.data  || [];
    renderTable(allHistory);
    updatePagination();
    computeStats(allHistory);
  } catch {
    tbody.innerHTML = `<tr><td colspan="8" style="text-align:center;padding:2rem;color:var(--text-muted);">
      Tidak dapat terhubung ke server
    </td></tr>`;
  }
}

// ─────────────────────────────────────────────────────────
// RENDER TABLE
// ─────────────────────────────────────────────────────────
function renderTable(rows) {
  const tbody = document.getElementById("history-tbody");
  if (!rows.length) {
    tbody.innerHTML = `<tr><td colspan="8" style="text-align:center;padding:3rem;color:var(--text-muted);">
      Belum ada riwayat verifikasi.
      <a href="verify.html" style="color:var(--rose-gold);font-weight:600;"> Mulai verifikasi →</a>
    </td></tr>`;
    return;
  }

  tbody.innerHTML = rows.map((row, i) => {
    const fname = (row.query_filename || "").length > 22
      ? row.query_filename.substring(0, 22) + "…"
      : row.query_filename;
    return `
    <tr>
      <td style="font-weight:700;color:var(--text);">${currentPage * PAGE_SIZE + i + 1}</td>
      <td style="font-size:0.78rem;color:var(--text-muted);">${formatDate(row.verification_timestamp)}</td>
      <td style="font-size:0.8rem;" title="${row.query_filename}">${fname}</td>
      <td style="font-weight:600;color:var(--text);">${row.predicted_name || "—"}</td>
      <td>${getSimilarityBadge(row.similarity_percent)}</td>
      <td>${getStatusBadge(row.verification_status)}</td>
      <td style="font-size:0.73rem;color:var(--text-muted);">${row.model_version ? row.model_version.substring(0, 14) : "—"}</td>
      <td style="display:flex;gap:0.3rem;">
        <button class="btn btn-secondary btn-sm" onclick="showDetail(${row.id})" title="Detail">
          <i data-lucide="eye"></i>
        </button>
        <button class="btn btn-danger btn-sm" onclick="deleteRecord(${row.id})" title="Hapus">
          <i data-lucide="trash-2"></i>
        </button>
      </td>
    </tr>`;
  }).join("");

  if (window.lucide) lucide.createIcons({ nodes: [tbody] });
}

// ─────────────────────────────────────────────────────────
// PAGINATION
// ─────────────────────────────────────────────────────────
function updatePagination() {
  const totalPages = Math.ceil(totalRecords / PAGE_SIZE);
  const start = currentPage * PAGE_SIZE + 1;
  const end   = Math.min((currentPage + 1) * PAGE_SIZE, totalRecords);

  document.getElementById("pagination-info").textContent =
    totalRecords > 0
      ? `Menampilkan ${start}–${end} dari ${totalRecords} data`
      : "Tidak ada data";

  document.getElementById("prev-btn").disabled = currentPage === 0;
  document.getElementById("next-btn").disabled = currentPage >= totalPages - 1;
}

function changePage(dir) {
  currentPage = Math.max(0, currentPage + dir);
  loadHistory();
}

// ─────────────────────────────────────────────────────────
// STATS
// ─────────────────────────────────────────────────────────
function computeStats(rows) {
  document.getElementById("stat-total-verify").textContent = totalRecords;

  const verified  = rows.filter(r => r.verification_status && r.verification_status.includes("TERIDENTIFIKASI")).length;
  const uncertain = rows.filter(r => r.verification_status && r.verification_status.includes("TIDAK PASTI")).length;
  const scores    = rows.map(r => parseFloat(r.similarity_percent)).filter(v => !isNaN(v));
  const avgScore  = scores.length
    ? (scores.reduce((a, b) => a + b, 0) / scores.length).toFixed(1) + "%"
    : "—";

  document.getElementById("stat-verified").textContent  = verified;
  document.getElementById("stat-uncertain").textContent = uncertain;
  document.getElementById("stat-avg-score").textContent = avgScore;
}

// ─────────────────────────────────────────────────────────
// FILTER
// ─────────────────────────────────────────────────────────
function applyFilters() {
  const nameQ   = document.getElementById("filter-name").value.toLowerCase();
  const statusQ = document.getElementById("filter-status").value;
  const filtered = allHistory.filter(r => {
    const matchName   = !nameQ   || (r.predicted_name || "").toLowerCase().includes(nameQ);
    const matchStatus = !statusQ || (r.verification_status || "").includes(statusQ);
    return matchName && matchStatus;
  });
  renderTable(filtered);
}

// ─────────────────────────────────────────────────────────
// DETAIL MODAL
// ─────────────────────────────────────────────────────────
async function showDetail(id) {
  const modal   = document.getElementById("detail-modal");
  const content = document.getElementById("modal-content");
  modal.classList.add("open");
  content.innerHTML = `<div style="text-align:center;padding:1.5rem;"><span class="spinner spinner-lg"></span></div>`;

  try {
    const res  = await fetch(`${API}/api/verify/${id}`);
    const data = await res.json();
    if (!data.success) throw new Error(data.message || "Gagal");
    const r = data.data;
    const matches = Array.isArray(r.top_matches) ? r.top_matches : [];

    content.innerHTML = `
      <table style="width:100%;font-size:0.86rem;border-collapse:collapse;">
        ${detailRow("ID", "#" + r.id, "hash")}
        ${detailRow("Waktu", formatDate(r.verification_timestamp), "clock")}
        ${detailRow("File", `<span style="word-break:break-all;">${r.query_filename}</span>`, "file-image")}
        ${detailRow("Prediksi", `<strong style="color:var(--text);font-size:0.95rem;">${r.predicted_name || "—"}</strong>`, "user-check")}
        ${detailRow("Similarity", getSimilarityBadge(r.similarity_percent), "star")}
        ${detailRow("Status", getStatusBadge(r.verification_status), "shield")}
        ${detailRow("Model", `<span style="font-size:0.78rem;">${r.model_version || "—"}</span>`, "cpu")}
      </table>

      ${matches.length ? `
        <div style="margin-top:1.1rem;">
          <p style="font-size:0.72rem;font-weight:700;color:var(--purple);letter-spacing:0.06em;text-transform:uppercase;margin-bottom:0.5rem;display:flex;align-items:center;gap:0.3rem;">
            <i data-lucide="list-ordered" style="width:12px;height:12px;"></i> Top Kandidat
          </p>
          ${matches.map((m, i) => `
            <div style="display:flex;justify-content:space-between;padding:0.42rem 0.6rem;background:${i===0?"var(--soft)":"transparent"};border-radius:var(--radius-sm);margin-bottom:0.25rem;font-size:0.84rem;border:1px solid ${i===0?"var(--pink)":"transparent"};">
              <span style="color:var(--text);display:flex;align-items:center;gap:0.35rem;">
                <i data-lucide="${i===0?"star":"circle"}" style="width:12px;height:12px;color:${i===0?"var(--gold)":"var(--text-muted)"};"></i>
                ${i + 1}. ${m.name}
              </span>
              <span style="font-weight:700;color:${i===0?"var(--text)":"var(--text-muted)"};">${m.percent.toFixed(1)}%</span>
            </div>`).join("")}
        </div>` : ""}

      <div style="margin-top:1.25rem;text-align:center;">
        <button class="btn btn-danger btn-sm" onclick="deleteRecord(${r.id}, true)">
          <i data-lucide="trash-2"></i> Hapus Record
        </button>
      </div>`;
    if (window.lucide) lucide.createIcons({ nodes: [content] });
  } catch (err) {
    content.innerHTML = `<div class="alert alert-error"><i data-lucide="circle-x"></i><span>${err.message}</span></div>`;
    if (window.lucide) lucide.createIcons({ nodes: [content] });
  }
}

function detailRow(label, val, icon) {
  return `
    <tr>
      <td style="padding:6px 10px 6px 0;color:var(--text-muted);white-space:nowrap;vertical-align:middle;display:flex;align-items:center;gap:0.35rem;">
        <i data-lucide="${icon}" style="width:12px;height:12px;"></i> ${label}
      </td>
      <td style="padding:6px 0;font-weight:500;">${val}</td>
    </tr>`;
}

function closeModal() {
  document.getElementById("detail-modal").classList.remove("open");
}

// Close on backdrop click
document.getElementById("detail-modal").addEventListener("click", (e) => {
  if (e.target === document.getElementById("detail-modal")) closeModal();
});

// ─────────────────────────────────────────────────────────
// DELETE
// ─────────────────────────────────────────────────────────
async function deleteRecord(id, closeAfter = false) {
  if (!confirm("Hapus riwayat verifikasi ini?")) return;
  try {
    const res  = await fetch(`${API}/api/verify/${id}`, { method: "DELETE" });
    const data = await res.json();
    if (data.success) {
      showToast("Record berhasil dihapus", "success");
      if (closeAfter) closeModal();
      loadHistory();
    } else showToast(data.message, "error");
  } catch { showToast("Gagal menghapus", "error"); }
}

// ─────────────────────────────────────────────────────────
// EXPORT CSV
// ─────────────────────────────────────────────────────────
function exportCSV() {
  if (!allHistory.length) { showToast("Tidak ada data untuk diexport", "warning"); return; }

  const headers = ["ID", "Waktu Verifikasi", "File", "Prediksi Penulis", "Similarity (%)", "Status", "Model"];
  const rows = allHistory.map(r => [
    r.id,
    r.verification_timestamp,
    r.query_filename,
    r.predicted_name    || "",
    r.similarity_percent ? parseFloat(r.similarity_percent).toFixed(2) : "",
    r.verification_status || "",
    r.model_version    || "",
  ]);

  const csvContent = [headers, ...rows]
    .map(row => row.map(v => `"${String(v).replace(/"/g, '""')}"`).join(","))
    .join("\n");

  const blob = new Blob(["\uFEFF" + csvContent], { type: "text/csv;charset=utf-8;" });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement("a");
  a.href     = url;
  a.download = `riwayat_verifikasi_${new Date().toISOString().slice(0, 10)}.csv`;
  a.click();
  URL.revokeObjectURL(url);
  showToast("CSV berhasil diexport!", "success");
}

// ─────────────────────────────────────────────────────────
// INIT
// ─────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", loadHistory);
