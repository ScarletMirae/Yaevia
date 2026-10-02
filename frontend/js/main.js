/**
 * main.js — Shared Utilities + Sakura Animation Engine + Global Theme Switcher
 * ==============================================================================
 * Yae Miko Theme | Sistem Verifikasi Tulisan Tangan
 */

// Jika diakses via http (Flask serve), gunakan URL relatif.
// Jika dibuka langsung sebagai file://, fallback ke localhost:5000.
const API_BASE = (location.protocol === 'file:')
  ? 'http://localhost:5000'
  : (location.origin);  // e.g. http://localhost:5000


// ── Init on DOM ready ─────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  initNavbar();
  initLucide();
  initSakura();
  initLoadingOverlay();
  initPublicEvaluationBadge();
});

// =========================================================
// GLOBAL THEME SWITCHER (Light Mode / Night Mode)
// =========================================================
function initTheme() {
  const saved = localStorage.getItem('yaevia-theme') || 'light';
  document.documentElement.dataset.theme = saved;

  const btn = ensureThemeToggleBtn();
  if (btn) {
    updateThemeToggleBtnUI(btn, saved);
    btn.onclick = (e) => {
      e.stopPropagation();
      const current = document.documentElement.dataset.theme || 'light';
      const next = current === 'dark' ? 'light' : 'dark';
      document.documentElement.dataset.theme = next;
      localStorage.setItem('yaevia-theme', next);
      updateThemeToggleBtnUI(btn, next);
      if (typeof showToast === 'function') {
        showToast(`Mode tema: ${next === 'dark' ? '🌙 Night Mode' : '☀ Light Mode'}`, 'info', 1800);
      }
    };
  }
}

function ensureThemeToggleBtn() {
  let btn = document.getElementById("theme-toggle-btn");
  if (!btn) {
    const navInner = document.querySelector(".navbar-inner");
    if (navInner) {
      btn = document.createElement("button");
      btn.className = "theme-toggle-btn";
      btn.id = "theme-toggle-btn";
      btn.setAttribute("aria-label", "Switch Theme");
      navInner.appendChild(btn);
    }
  }
  return btn;
}

function updateThemeToggleBtnUI(btn, theme) {
  if (!btn) return;
  const isDark = theme === 'dark';
  btn.setAttribute('aria-label', isDark ? 'Aktifkan Light Mode' : 'Aktifkan Night Mode');
  btn.setAttribute('title', isDark ? 'Mode Terang (Light)' : 'Mode Gelap (Night)');
  btn.innerHTML = `<i data-lucide="${isDark ? 'sun' : 'moon'}"></i>`;
  if (window.lucide) lucide.createIcons({ nodes: [btn] });
}

// =========================================================
// PUBLIC EVALUATION MODE BADGE
// =========================================================
async function initPublicEvaluationBadge() {
  try {
    const res = await fetch(API_BASE + "/api/health");
    const data = await res.json();
    if (data && data.public_evaluation_mode) {
      window.IS_PUBLIC_EVALUATION_MODE = true;
      const brand = document.querySelector(".navbar-brand");
      if (brand && !document.getElementById("eval-mode-badge")) {
        const badge = document.createElement("span");
        badge.id = "eval-mode-badge";
        badge.className = "badge badge-pink";
        badge.style.cssText = "font-size:0.72rem;padding:2px 8px;margin-left:0.6rem;font-weight:700;display:inline-flex;align-items:center;gap:3px;";
        badge.innerHTML = `<i data-lucide="shield-check" style="width:12px;height:12px;"></i> Mode Evaluasi Media`;
        brand.parentNode.insertBefore(badge, brand.nextSibling);
        if (window.lucide) lucide.createIcons({ nodes: [badge] });
      }
    }
  } catch {}
}

// ── Loading Overlay (Navigasi Halaman) ────────────────────
function initLoadingOverlay() {
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

  document.addEventListener("click", (e) => {
    const link = e.target.closest("a[href]");
    if (!link) return;

    const href = link.getAttribute("href");

    if (!href || href === "#" || href.startsWith("javascript:") || href.startsWith("http")) return;
    if (href === location.pathname.split("/").pop()) return;
    if (link.target === "_blank") return;

    e.preventDefault();
    const overlay = document.getElementById("yaevia-loading-overlay");
    const bar = document.getElementById("loading-progress-bar");
    if (overlay) {
      if (bar) bar.style.width = "0%";
      overlay.classList.add("active");

      let start = null;
      const duration = 1200;
      function animateBar(timestamp) {
        if (!start) start = timestamp;
        const progress = Math.min((timestamp - start) / duration, 1);
        if (bar) bar.style.width = (progress * 100) + "%";
        if (progress < 1) {
          requestAnimationFrame(animateBar);
        }
      }
      requestAnimationFrame(animateBar);

      setTimeout(() => {
        window.location.href = href;
      }, 1200);
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
const PETAL_VARIANTS = [
  `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 28">
     <path d="M10 2 C16 2, 20 8, 18 16 C16 22, 12 26, 10 26 C8 26, 4 22, 2 16 C0 8, 4 2, 10 2Z"
           fill="rgba(247,198,217,VAR_OPACITY)"/>
     <path d="M10 2 C10 8, 10 16, 10 26" stroke="rgba(235,168,195,0.4)" stroke-width="0.5" fill="none"/>
   </svg>`,
  `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 22 28">
     <path d="M11 4 C11 4, 5 1, 3 7 C1 13, 6 19, 11 26 C16 19, 21 13, 19 7 C17 1, 11 4, 11 4Z"
           fill="rgba(250,220,232,VAR_OPACITY)"/>
     <path d="M11 5 C11 10, 11 18, 11 26" stroke="rgba(235,168,195,0.3)" stroke-width="0.5" fill="none"/>
   </svg>`,
  `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 30">
     <ellipse cx="8" cy="15" rx="7" ry="13"
              fill="rgba(212,163,115,VAR_OPACITY)"
              transform="rotate(-8 8 15)"/>
   </svg>`,
];

function initSakura() {
  let container = document.getElementById("sakura-container");
  if (container && container.children.length > 0) return;
  if (!container) {
    container = document.createElement("div");
    container.id = "sakura-container";
    document.body.insertBefore(container, document.body.firstChild);
  }

  const PETAL_COUNT    = 50;
  const MIN_DURATION   = 10;
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

  const size     = minSz + Math.random() * (maxSz - minSz);
  const duration = minDur + Math.random() * (maxDur - minDur);
  const delay    = -(Math.random() * duration);
  const startX   = Math.random() * 105;
  const opacity  = 0.55 + Math.random() * 0.45;
  const swayAmt  = 40 + Math.random() * 80;
  const swayDir  = Math.random() > 0.5 ? 1 : -1;
  const blur     = size < 10 ? (Math.random() * 1.5) : 0;
  const glow     = Math.random() > 0.6;

  const variant  = PETAL_VARIANTS[Math.floor(Math.random() * PETAL_VARIANTS.length)];
  const svgStr   = variant.replace(/VAR_OPACITY/g, opacity.toFixed(2));
  const svgUrl   = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svgStr)}`;

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
// TOAST NOTIFICATIONS & ALERTS
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
  if (pct >= 65) cls = "badge-green";
  else if (pct >= 50) cls = "badge-gold";
  else if (pct >= 40) cls = "badge-yellow";
  return `<span class="badge ${cls} numeric-value">${pct.toFixed(1)}%</span>`;
}

function getStatusBadge(status) {
  if (!status) return "—";
  const s = String(status).toUpperCase();
  if (s === "SANGAT MIRIP" || s === "SANGAT YAKIN")
    return `<span class="badge badge-green"><i data-lucide="shield-check"></i> ${status}</span>`;
  if (s === "TERIDENTIFIKASI")
    return `<span class="badge badge-green"><i data-lucide="check-circle"></i> Teridentifikasi</span>`;
  if (s === "MIRIP")
    return `<span class="badge badge-gold"><i data-lucide="check-circle"></i> Mirip</span>`;
  if (s === "TIDAK PASTI")
    return `<span class="badge badge-yellow"><i data-lucide="help-circle"></i> Tidak Pasti</span>`;
  if (s === "KURANG MIRIP")
    return `<span class="badge badge-yellow"><i data-lucide="help-circle"></i> Kurang Mirip</span>`;
  if (s === "TIDAK TERIDENTIFIKASI")
    return `<span class="badge badge-red"><i data-lucide="x-circle"></i> Tidak Teridentifikasi</span>`;
  if (s === "TIDAK MIRIP")
    return `<span class="badge badge-red"><i data-lucide="x-circle"></i> Tidak Mirip</span>`;
  return `<span class="badge badge-gold">${status}</span>`;
}

async function apiFetch(url, options = {}) {
  const res  = await fetch(API_BASE + url, options);
  const data = await res.json().catch(() => ({}));
  return { ok: res.ok, status: res.status, data };
}
