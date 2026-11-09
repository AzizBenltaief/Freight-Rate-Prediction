/* Freight Rate Predictor - frontend logic (no dependencies) */
"use strict";

const $ = (id) => document.getElementById(id);
const fmtUSD = (v) => "$" + Number(v).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });

/* ---------------- API health ---------------- */
async function checkHealth() {
  const dot = $("api-status");
  try {
    const res = await fetch("/api/health");
    dot.className = "status-dot " + (res.ok ? "ok" : "down");
    dot.title = res.ok ? "API online" : "API error";
  } catch {
    dot.className = "status-dot down";
    dot.title = "API offline";
  }
}

/* ---------------- autoload cities + equipment from meta ----------------
   (also trains the model on first call, so the later predict is instant) */
async function loadMeta() {
  try {
    const res = await fetch("/api/meta");
    if (!res.ok) throw new Error("backend not ready");
    const meta = await res.json();
    const dl = $("city-list");
    dl.innerHTML = meta.cities.map((c) => `<option value="${c.replace(/"/g, "&quot;")}">`).join("");
    $("equipment").innerHTML = meta.equipment_types
      .map((e) => `<option value="${e}">${e}</option>`)
      .join("");
  } catch {
    $("error").classList.remove("hidden");
    $("error").textContent = "Cannot reach the API. Start the server with: python -m src.app";
  } finally {
    checkHealth();
  }
}

/* ---------------- prediction form ---------------- */
$("predict-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const btn = $("predict-btn");
  const body = {
    date: $("date").value,
    distance_miles: parseFloat($("distance").value),
    weight_lbs: $("weight").value === "" ? null : parseFloat($("weight").value),
    equipment: $("equipment").value,
    pickup_city: $("pickup").value.trim(),
    delivery_city: $("delivery").value.trim(),
  };

  setBusy(true);
  hideError();
  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "prediction failed");
    showResult(data.predicted_rate, body);
  } catch (err) {
    $("result").classList.add("hidden");
    showError(String(err.detail || err.message || err));
  } finally {
    setBusy(false);
  }
});

function setBusy(busy) {
  $("predict-btn").disabled = busy;
  $("predict-btn").textContent = busy ? "Predicting\u2026" : "Predict rate";
}

function showResult(rate, inputs) {
  $("result").classList.remove("hidden");
  $("result-rate").textContent = fmtUSD(rate);
  $("result-detail").textContent =
    `${inputs.equipment} \u00b7 ${inputs.distance_miles} mi \u00b7 ` +
    `${inputs.pickup_city} \u2192 ${inputs.delivery_city} \u00b7 ${inputs.date}` +
    (inputs.weight_lbs ? ` \u00b7 ${inputs.weight_lbs.toLocaleString("en-US")} lbs` : " \u00b7 weight = median");
  $("result-detail").innerHTML +=
    ` \u00b7 <span class="per-mile">$${(rate / inputs.distance_miles).toFixed(2)}/mile</span>`;
}

function showError(msg) { $("error").textContent = msg; $("error").classList.remove("hidden"); }
function hideError() { $("error").classList.add("hidden"); }

/* ---------------- demo lane (the December chart lane) ---------------- */
$("fill-demo").addEventListener("click", () => {
  $("date").value = "2025-12-01";
  $("equipment").value = "Dry Van";
  $("pickup").value = "Lexington";
  $("delivery").value = "Fort Wayne";
  $("distance").value = "360";
  $("weight").value = "32000";
});

/* ---------------- December chart (hand-rolled SVG line chart) ---------------- */
$("load-chart-btn").addEventListener("click", async () => {
  const btn = $("load-chart-btn");
  btn.disabled = true;
  btn.textContent = "Loading\u2026";
  try {
    const res = await fetch("/api/december");
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "failed to load chart");
    $("lane-info").textContent =
      `${data.lane.pickup} \u2192 ${data.lane.delivery} \u00b7 Dry Van \u00b7 32,000 lbs \u00b7 360 mi`;
    drawChart(data.points);
  } catch (err) {
    $("chart").innerHTML = `<p class="hint">${err.detail || err.message}</p>`;
  } finally {
    btn.disabled = false;
    btn.textContent = "Load daily December forecast";
  }
});

function drawChart(points) {
  const W = 640, H = 300, PAD = { t: 18, r: 18, b: 46, l: 62 };
  const xs = points.map((_, i) => i);
  const ys = points.map((p) => p.predicted_rate);
  const yMin = Math.min(...ys) * 0.96, yMax = Math.max(...ys) * 1.04;

  const px = (i) => PAD.l + (xs[i] / (points.length - 1)) * (W - PAD.l - PAD.r);
  const py = (v) => PAD.t + (1 - (v - yMin) / (yMax - yMin)) * (H - PAD.t - PAD.b);

  const line = points.map((p, i) => `${i ? "L" : "M"}${px(i).toFixed(1)},${py(p.predicted_rate).toFixed(1)}`).join(" ");
  const area = line + ` L${px(points.length - 1).toFixed(1)},${H - PAD.b} L${px(0)},${H - PAD.b} Z`;

  const yTicks = 5;
  let grid = "";
  for (let t = 0; t <= yTicks; t++) {
    const v = yMin + (t / yTicks) * (yMax - yMin);
    const y = py(v);
    grid += `<line x1="${PAD.l}" y1="${y}" x2="${W - PAD.r}" y2="${y}" stroke="#2d3648" stroke-dasharray="3 3"/>` +
            `<text x="${PAD.l - 10}" y="${y + 4}" fill="#8b98a9" font-size="11" text-anchor="end">$${Math.round(v)}</text>`;
  }
  const dateTicks = [1, 8, 15, 22, 31];
  let labels = "";
  for (const d of dateTicks) {
    const x = px(d - 1);
    labels += `<text x="${x}" y="${H - PAD.b + 18}" fill="#8b98a9" font-size="11" text-anchor="middle">Dec ${d}</text>`;
  }

  const dots = points.map((p, i) =>
    `<circle data-i="${i}" cx="${px(i).toFixed(1)}" cy="${py(p.predicted_rate).toFixed(1)}" r="4" fill="#f59e0b" opacity="0"/>`
  ).join("");

  $("chart").innerHTML =
    `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg">
      ${grid}
      <path d="${area}" fill="rgba(245,158,11,0.12)"/>
      <path d="${line}" fill="none" stroke="#f59e0b" stroke-width="2.5" stroke-linejoin="round"/>
      ${dots}
      ${labels}
    </svg>`;

  // hover tooltip
  const tooltip = $("chart-tooltip");
  const chartBox = $("chart");
  chartBox.querySelectorAll("circle").forEach((c) => {
    c.addEventListener("mouseenter", () => {
      c.setAttribute("opacity", "1");
      const p = points[c.dataset.i];
      tooltip.innerHTML = `<b>${p.date}</b><br/>${fmtUSD(p.predicted_rate)}`;
      tooltip.classList.remove("hidden");
      const box = chartBox.getBoundingClientRect();
      const pt = c.getBoundingClientRect();
      tooltip.style.left = (pt.left - box.left + pt.width / 2) + "px";
      tooltip.style.top = (pt.top - box.top) + "px";
    });
    c.addEventListener("mouseleave", () => {
      c.setAttribute("opacity", "0");
      tooltip.classList.add("hidden");
    });
  });
}

/* ---------------- boot ---------------- */
loadMeta();
