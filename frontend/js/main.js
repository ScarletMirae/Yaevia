/**
 * main.js — Shared Utilities + Sakura Animation Engine
 * ======================================================
 * Yae Miko Theme | Sistem Verifikasi Tulisan Tangan
 */

// Jika diakses via http (Flask serve), gunakan URL relatif.
// Jika dibuka langsung sebagai file://, fallback ke localhost:5000.
const API_BASE = (location.protocol === 'file:')
  ? 'http://localhost:5000'
  : (location.origin);  // e.g. http://localhost:5000


// ── Init on DOM ready ─────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  initNavbar();
  initLucide();
  initSakura();
  initLoadingOverlay();
});

// ── Loading Overlay (Navigasi Halaman) ────────────────────
// [PNG LOADING] — Ganti file: images/icons/loading.png
function initLoadingOverlay() {
  // Buat overlay element jika belum ada
  if (!document.getElementById("yaevia-loading-overlay")) {
    const overlay = document.createElement("div");
    overlay.id = "yaevia-loading-overlay";
    overlay.innerHTML = `
      <div class="loading-card">
        <div class="loading-img-frame">
          <img src="images/icons/loading.png" alt="Loading...">
        </div>
        <div class="loading-content">
          <div class="loading-text">Memuat...</div>
          <div class="loading-progress-track">
            <div class="loading-progress-bar" id="loading-progress-bar"></div>
          </div>
        </div>
        <div class="loading-spinner"></div>
      </div>`;
    document.body.appendChild(overlay);
  }

  // Intercept semua link navigasi (nav-link & tombol pindah halaman)
  document.addEventListener("click", (e) => {
    const link = e.target.closest("a[href]");
    if (!link) return;

    const href = link.getAttribute("href");

    // Skip: anchor (#), javascript:, external link, link kosong, link ke halaman sama
    if (!href || href === "#" || href.startsWith("javascript:") || href.startsWith("http")) return;
    if (href === location.pathname.split("/").pop()) return;

    // Skip: link yang membuka di tab baru
    if (link.target === "_blank") return;

    // Tampilkan overlay + animasi progress bar selama 1500ms
    e.preventDefault();
    const overlay = document.getElementById("yaevia-loading-overlay");
    const bar = document.getElementById("loading-progress-bar");
    if (overlay) {
      // Reset progress bar
      if (bar) bar.style.width = "0%";
      overlay.classList.add("active");

      // Animate progress bar dari 0% ke 100% dalam 1500ms
      let start = null;
      const duration = 1500;
      function animateBar(timestamp) {
        if (!start) start = timestamp;
        const progress = Math.min((timestamp - start) / duration, 1);
        if (bar) bar.style.width = (progress * 100) + "%";
        if (progress < 1) {
          requestAnimationFrame(animateBar);
        }
      }
      requestAnimationFrame(animateBar);

      // Navigasi setelah 1500ms
      setTimeout(() => {
        window.location.href = href;
      }, 1500);
    } else {
      window.location.href = href;
    }
  });
}


// ── Lucide Icons Init ─────────────────────────────────────
function initLucide() {
  if (window.lucide) {
    lucide.createIcons();
  }
}

// ── Navbar toggle ─────────────────────────────────────────
function initNavbar() {
  const toggle = document.getElementById("nav-toggle");
  const nav    = document.getElementById("navbar-nav");
  if (!toggle || !nav) return;

  toggle.addEventListener("click", (e) => {
    e.stopPropagation();
    nav.classList.toggle("open");
  });

  document.addEventListener("click", (e) => {
    if (!toggle.contains(e.target) && !nav.contains(e.target)) {
      nav.classList.remove("open");
    }
  });
}

// =========================================================
// SAKURA PETAL ANIMATION ENGINE
// =========================================================
/**
 * Sakura petal shape as inline SVG data URI.
 * Three variants for visual variety.
 */
const PETAL_VARIANTS = [
  // Soft round petal
  `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 28">
     <path d="M10 2 C16 2, 20 8, 18 16 C16 22, 12 26, 10 26 C8 26, 4 22, 2 16 C0 8, 4 2, 10 2Z"
           fill="rgba(247,198,217,VAR_OPACITY)"/>
     <path d="M10 2 C10 8, 10 16, 10 26" stroke="rgba(235,168,195,0.4)" stroke-width="0.5" fill="none"/>
   </svg>`,
  // Heart petal
  `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 22 28">
     <path d="M11 4 C11 4, 5 1, 3 7 C1 13, 6 19, 11 26 C16 19, 21 13, 19 7 C17 1, 11 4, 11 4Z"
           fill="rgba(250,220,232,VAR_OPACITY)"/>
     <path d="M11 5 C11 10, 11 18, 11 26" stroke="rgba(235,168,195,0.3)" stroke-width="0.5" fill="none"/>
   </svg>`,
  // Elongated petal
  `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 30">
     <ellipse cx="8" cy="15" rx="7" ry="13"
              fill="rgba(212,163,115,VAR_OPACITY)"
              transform="rotate(-8 8 15)"/>
   </svg>`,
];

/**
 * Creates & animates sakura petals as background layer.
 * Uses pure CSS animations for compositor-thread performance.
 */
function initSakura() {
  // Container
  const container = document.createElement("div");
  container.id = "sakura-container";
  document.body.insertBefore(container, document.body.firstChild);

  const PETAL_COUNT    = 65;
  const MIN_DURATION   = 10;  // seconds
  const MAX_DURATION   = 26;
  const MIN_SIZE       = 7;
  const MAX_SIZE       = 22;

  for (let i = 0; i < PETAL_COUNT; i++) {
    spawnPetal(container, i, PETAL_COUNT, MIN_DURATION, MAX_DURATION, MIN_SIZE, MAX_SIZE);
  }
}

function spawnPetal(container, index, total, minDur, maxDur, minSz, maxSz) {
  const petal = document.createElement("div");
  petal.className = "sakura-petal";

  // Random properties
  const size     = minSz + Math.random() * (maxSz - minSz);
  const duration = minDur + Math.random() * (maxDur - minDur);
  const delay    = -(Math.random() * duration);       // negative = already in-flight on load
  const startX   = Math.random() * 105;               // % from left (a bit beyond edges)
  const opacity  = 0.55 + Math.random() * 0.45;      // 0.55 – 1.0
  const swayAmt  = 40 + Math.random() * 80;           // px sway amplitude
  const swayDir  = Math.random() > 0.5 ? 1 : -1;
  const blur     = size < 10 ? (Math.random() * 1.5) : 0;  // small petals get blur (depth)
  const glow     = Math.random() > 0.6;               // some petals get soft glow

  // Pick petal variant
  const variant  = PETAL_VARIANTS[Math.floor(Math.random() * PETAL_VARIANTS.length)];
  const svgStr   = variant.replace(/VAR_OPACITY/g, opacity.toFixed(2));
  const svgUrl   = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svgStr)}`;

  // Sway amounts for three quarter-points
  const swayA = `${swayAmt * swayDir}px`;
  const swayB = `${-swayAmt * swayDir * 0.6}px`;
  const swayC = `${swayAmt * swayDir * 0.8}px`;

  petal.style.cssText = `
    left: ${startX}%;
    width: ${size}px;
    height: ${size * 1.4}px;
    background-image: url("${svgUrl}");
    background-size: contain;
    background-repeat: no-repeat;
    animation-duration: ${duration.toFixed(1)}s;
    animation-delay: ${delay.toFixed(1)}s;
    animation-timing-function: linear;
    animation-iteration-count: infinite;
    --sway-a: ${swayA};
    --sway-b: ${swayB};
    --sway-c: ${swayC};
    filter: blur(${blur.toFixed(1)}px) ${glow ? `drop-shadow(0 0 3px rgba(247,198,217,0.5))` : ''};
    border-radius: 0;
  `;

  container.appendChild(petal);
}

// =========================================================
// TOAST NOTIFICATIONS
// =========================================================
const TOAST_ICONS = {
  success: "circle-check",
  error:   "circle-x",
  info:    "info",
  warning: "triangle-alert",
};

function showToast(message, type = "info", duration = 4200) {
  const container = document.getElementById("toast-container");
  if (!container) return;

  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  toast.innerHTML = `<i data-lucide="${TOAST_ICONS[type] || 'info'}"></i><span>${message}</span>`;
  container.appendChild(toast);

  if (window.lucide) lucide.createIcons({ nodes: [toast] });

  setTimeout(() => {
    toast.style.animation = "toast-out 0.3s var(--ease) both";
    setTimeout(() => toast.remove(), 350);
  }, duration);
}

// =========================================================
// ALERT IN PAGE
// =========================================================
const ALERT_ICONS = { success:"circle-check", error:"circle-x", info:"info", warning:"triangle-alert" };

function showAlert(containerId, message, type = "info", autoDismiss = 0) {
  const el = document.getElementById(containerId);
  if (!el) return;

  el.innerHTML = `
    <div class="alert alert-${type}" role="alert">
      <i data-lucide="${ALERT_ICONS[type] || 'info'}"></i>
      <span>${message}</span>
    </div>`;

  if (window.lucide) lucide.createIcons({ nodes: [el] });
  if (autoDismiss > 0) setTimeout(() => { el.innerHTML = ""; }, autoDismiss);
}

// =========================================================
// FORMATTERS & BADGE HELPERS
// =========================================================
function formatDate(isoString) {
  if (!isoString) return "—";
  try {
    return new Date(isoString).toLocaleString("id-ID", {
      day: "2-digit", month: "short", year: "numeric",
      hour: "2-digit", minute: "2-digit",
    });
  } catch { return isoString; }
}

function getSimilarityBadge(percent) {
  if (percent === null || percent === undefined) return "—";
  const pct = parseFloat(percent);
  let cls = "badge-red";
  if (pct >= 75) cls = "badge-green";
  else if (pct >= 60) cls = "badge-gold";
  else if (pct >= 40) cls = "badge-yellow";
  return `<span class="badge ${cls}">${pct.toFixed(1)}%</span>`;
}

function getStatusBadge(status) {
  if (!status) return "—";
  if (status.includes("SANGAT YAKIN"))
    return `<span class="badge badge-green"><i data-lucide="shield-check"></i> Sangat Yakin</span>`;
  if (status.includes("TERIDENTIFIKASI"))
    return `<span class="badge badge-gold"><i data-lucide="check-circle"></i> Teridentifikasi</span>`;
  if (status.includes("TIDAK PASTI"))
    return `<span class="badge badge-yellow"><i data-lucide="help-circle"></i> Tidak Pasti</span>`;
  return `<span class="badge badge-red"><i data-lucide="x-circle"></i> Tidak Teridentifikasi</span>`;
}

// =========================================================
// GENERIC API FETCH
// =========================================================
async function apiFetch(url, options = {}) {
  const res  = await fetch(API_BASE + url, options);
  const data = await res.json().catch(() => ({}));
  return { ok: res.ok, status: res.status, data };
}
