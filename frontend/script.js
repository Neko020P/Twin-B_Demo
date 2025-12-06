// frontend/script.js
// Zone Comfort Heatmap - FIXED to match Python notebook calculation
// Formula: comfort_level = 10 × (1 - |current_temp - preferred_temp| / 5)

const SCENARIOS = ['conference','exam_period','normal_weekday','summer_break','weekend'];
const POLICIES = ['minimum_activation','setpoint_expansion_1c','setpoint_expansion_2c','setpoint_expansion_3c'];

// ---------- Utility ----------
function safeGet(obj, ...keys) {
  for (const k of keys) {
    if (obj && Object.prototype.hasOwnProperty.call(obj, k)) return obj[k];
  }
  return undefined;
}
function parseNumber(v) {
  if (v == null) return NaN;
  if (typeof v === 'number') return v;
  const n = parseFloat(String(v).replace(',', '.'));
  return isNaN(n) ? NaN : n;
}

// ✅ FIXED: Match Python notebook formula
function computeComfortFromTemperature(currentTemp, preferredTemp) {
  const curr = parseNumber(currentTemp);
  const pref = parseNumber(preferredTemp);
  
  if (isNaN(curr) || isNaN(pref)) return NaN;
  
  // Python formula: 10 × (1 - temp_deviation / 5)
  const tempDeviation = Math.abs(curr - pref);
  const maxDeviation = 5.0;
  const comfortScore = 10.0 * (1.0 - tempDeviation / maxDeviation);
  
  // Clamp to [0, 10]
  return Math.max(0.0, Math.min(10.0, comfortScore));
}

// Color interpolation: t in [0..1] -> red->yellow->green
function comfortToColor(value, min, max) {
  if (value == null || isNaN(value)) return '#efefef';
  let t = (value - min) / (max - min || 1);
  t = Math.max(0, Math.min(1, t));
  if (t <= 0.5) {
    const p = t / 0.5;
    const r = Math.round(220 + (250 - 220) * p);
    const g = Math.round(50 + (250 - 50) * p);
    const b = Math.round(50 + (120 - 50) * p);
    return `rgb(${r},${g},${b})`;
  } else {
    const p = (t - 0.5) / 0.5;
    const r = Math.round(250 + (20 - 250) * p);
    const g = Math.round(250 + (160 - 250) * p);
    const b = Math.round(120 + (50 - 120) * p);
    return `rgb(${r},${g},${b})`;
  }
}

// ---------- Fetch helpers ----------
async function fetchCsvJson(dataType, scenario, policy) {
  const url = `/api/csv/${dataType}/${scenario}/${policy}`;
  const r = await fetch(url);
  if (!r.ok) throw new Error(`Failed to fetch ${url}: ${r.status}`);
  const json = await r.json();
  return json.rows ?? json.data ?? json;
}
function getSelectedDataType() {
  const el = document.getElementById('data-type');
  return (el && el.value) ? el.value : 'zones';
}

// ---------- Row accessors ----------
function extractZoneFromRow(row) {
  if (!row) return '';
  return (row.zone ?? row.Zone ?? row.ZoneName ?? row.zone_name ?? row.Room ?? row.room ?? row['room_name'] ?? '').toString();
}
function extractHourFromRow(row) {
  let hourVal = safeGet(row, 'hour', 'Hour', 'hour_of_day', 'h');
  if (hourVal == null) {
    const t = safeGet(row, 'timestamp', 'time', 'datetime', 'DateTime', 'date');
    if (t) {
      const d = new Date(t);
      if (!isNaN(d)) return d.getHours();
      const m = String(t).match(/(\d{1,2}):\d{2}/);
      if (m) return parseInt(m[1], 10);
    }
    return NaN;
  }
  if (typeof hourVal === 'string') {
    const m = hourVal.match(/(\d{1,2})/);
    if (m) return parseInt(m[1], 10);
    const n = parseInt(hourVal, 10);
    return isNaN(n) ? NaN : n;
  }
  const n = Number(hourVal);
  return isNaN(n) ? NaN : n;
}

// ---------- Temperature extraction ----------
function extractNumericFromString(s) {
  if (s == null) return NaN;
  if (typeof s === 'number') return s;
  const m = String(s).match(/-?\d+(\.\d+)?/);
  if (m) return parseFloat(m[0]);
  const n = parseNumber(s);
  return isNaN(n) ? NaN : n;
}

// ✅ Extract current_temp (zone temperature)
function extractCurrentTempFromRow(row) {
  if (!row) return NaN;
  const tempFields = ['current_temp', 'zone_temp', 'temperature', 'temp', 'zoneTemp', 'currentTemp'];
  for (const f of tempFields) {
    const v = safeGet(row, f);
    if (v != null) {
      const n = extractNumericFromString(v);
      if (!isNaN(n)) return n;
    }
  }
  return NaN;
}

// ✅ Extract preferred_temp (setpoint)
function extractPreferredTempFromRow(row) {
  if (!row) return NaN;
  const setpointFields = ['preferred_temp', 'setpoint', 'set_point', 'target_temp', 'targetTemperature', 'preferredTemp'];
  for (const f of setpointFields) {
    const v = safeGet(row, f);
    if (v != null) {
      const n = extractNumericFromString(v);
      if (!isNaN(n)) return n;
    }
  }
  return NaN;
}

// ---------- Normalization & matching ----------
function normalizeRoomString(s) {
  if (!s && s !== 0) return '';
  const t = String(s).trim().toLowerCase();
  let u = t.replace(/^(zone_|room_)/, '');
  u = u.replace(/[^a-z0-9]/g, '');
  return u;
}
function numericSuffix(s) {
  const m = String(s).match(/(\d{2,})$/);
  return m ? m[1] : null;
}
function roomMatches(roomSelected, roomFromRow) {
  const a = normalizeRoomString(roomSelected);
  const b = normalizeRoomString(roomFromRow);
  if (!a || !b) return false;
  if (a === b) return true;
  if (a.includes(b) || b.includes(a)) return true;
  const an = numericSuffix(roomSelected);
  const bn = numericSuffix(roomFromRow);
  if (an && bn && an === bn) return true;
  return false;
}

// ---------- Discover rooms ----------
async function discoverRooms() {
  const dataType = getSelectedDataType();
  const set = new Set();
  const promises = [];
  for (const scenario of SCENARIOS) {
    for (const policy of POLICIES) {
      promises.push(fetchCsvJson(dataType, scenario, policy).catch(err => ({ rows: [] })));
    }
  }
  const results = await Promise.all(promises);
  for (const res of results) {
    const rows = res.rows ?? res.data ?? res;
    if (!Array.isArray(rows)) continue;
    for (const r of rows) {
      const z = extractZoneFromRow(r);
      if (z && z.trim()) set.add(z.trim());
    }
  }
  const arr = Array.from(set).sort((a,b) => a.localeCompare(b, undefined, {sensitivity: 'base'}));
  const sel = document.getElementById('room-select');
  if (!sel) return;
  sel.innerHTML = '';
  if (arr.length === 0) {
    const opt = document.createElement('option'); opt.value = ''; opt.text = '(no rooms found)'; sel.appendChild(opt); return;
  }
  const placeholder = document.createElement('option'); placeholder.value = ''; placeholder.text = '(select a room)'; sel.appendChild(placeholder);
  for (const z of arr) {
    const o = document.createElement('option'); o.value = z; o.text = z; sel.appendChild(o);
  }
}

// ✅ FIXED: Build comfort matrix using Python formula
async function fetchComfortMatrixForRoom(room) {
  const dataType = getSelectedDataType();
  const hours = []; for (let h = 6; h <= 17; h++) hours.push(h);
  const cols = SCENARIOS.length * POLICIES.length;
  const matrix = Array.from({length: hours.length}, () => Array(cols).fill(NaN));
  const headers = [];

  const fetchList = [];
  for (const scenario of SCENARIOS) {
    for (const policy of POLICIES) {
      headers.push(`${scenario}\n${policy.replace(/_/g, ' ')}`);
      fetchList.push(fetchCsvJson(dataType, scenario, policy).catch(err => ({ rows: [] })));
    }
  }
  const results = await Promise.all(fetchList);

  // ✅ Process each column using FIXED formula
  const debugCounts = [];
  for (let col = 0; col < results.length; col++) {
    const res = results[col];
    const rows = res.rows ?? res.data ?? res;
    let matchedRowCount = 0;
    if (Array.isArray(rows)) {
      const sums = new Array(hours.length).fill(0);
      const counts = new Array(hours.length).fill(0);
      for (const r of rows) {
        const zoneName = extractZoneFromRow(r);
        if (!zoneName || !roomMatches(room, zoneName)) continue;
        matchedRowCount++;
        const hour = extractHourFromRow(r);
        if (isNaN(hour) || hour < 6 || hour > 17) continue;
        const idx = hour - 6;
        
        // ✅ Extract current_temp and preferred_temp
        const currentTemp = extractCurrentTempFromRow(r);
        const preferredTemp = extractPreferredTempFromRow(r);
        if (isNaN(currentTemp) || isNaN(preferredTemp)) continue;
        
        // ✅ Apply FIXED formula matching Python: 10 × (1 - |deviation| / 5)
        const comfortScore = computeComfortFromTemperature(currentTemp, preferredTemp);
        if (isNaN(comfortScore)) continue;
        
        sums[idx] += comfortScore;
        counts[idx] += 1;
      }
      for (let hi = 0; hi < hours.length; hi++) {
        if (counts[hi] > 0) matrix[hi][col] = Number(sums[hi] / counts[hi]);
        else matrix[hi][col] = NaN;
      }
    }
    debugCounts.push(matchedRowCount);
  }

  console.debug('✅ fetchComfortMatrixForRoom (FIXED):', {
    room,
    dataType: getSelectedDataType(),
    headers,
    hours,
    formula: 'comfort = 10 × (1 - |current_temp - preferred_temp| / 5)',
    matchedCountsPerColumn: debugCounts,
    sampleRow0: matrix[0]
  });

  return { headers, hours, matrix };
}

// ---------- Draw heatmap ----------
let selectedRoomForHeatmap = '';

function drawZoneComfortHeatmap(canvasSelector, headers, hours, matrix) {
  const canvas = document.querySelector(canvasSelector);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  const DPR = window.devicePixelRatio || 1;
  const containerW = canvas.parentElement ? canvas.parentElement.clientWidth : 1200;
  const containerH = Math.max(420, hours.length * 40);
  canvas.style.width = containerW + 'px';
  canvas.style.height = containerH + 'px';
  canvas.width = Math.round(containerW * DPR);
  canvas.height = Math.round(containerH * DPR);
  ctx.setTransform(DPR, 0, 0, DPR, 0, 0);

  const paddingLeft = 120;
  const paddingRight = 140;
  const paddingTop = 58;
  const paddingBottom = 80;
  const w = containerW - paddingLeft - paddingRight;
  const h = containerH - paddingTop - paddingBottom;
  const cols = headers.length || 1;
  const rows = hours.length || matrix.length;
  const cellW = w / Math.max(1, cols);
  const cellH = h / Math.max(1, rows);

  // ✅ Colorbar range: 0–10 (full scale)
  let cbMin = 0, cbMax = 10;

  console.debug('drawZoneComfortHeatmap: colorbar range 0–10 (FIXED)', { cbMin, cbMax });

  ctx.clearRect(0, 0, containerW, containerH);

  ctx.fillStyle = '#000';
  ctx.font = '16px Arial';
  ctx.textAlign = 'center';
  ctx.fillText(`Comfort Level Throughout the Day - Room ${selectedRoomForHeatmap || ''} (All Scenarios+Policies)`, paddingLeft + w/2, 26);

  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const x = paddingLeft + c * cellW;
      const y = paddingTop + r * cellH;
      const v = matrix[r]?.[c];
      const fillColor = (v == null || isNaN(v)) ? '#efefef' : comfortToColor(Number(v), cbMin, cbMax);
      ctx.fillStyle = fillColor;
      ctx.fillRect(Math.round(x)+1, Math.round(y)+1, Math.max(0, Math.round(cellW)-2), Math.max(0, Math.round(cellH)-2));
      if (!isNaN(v)) {
        ctx.fillStyle = '#000';
        ctx.font = '11px Arial';
        const txt = Number(v).toFixed(2);
        ctx.fillText(txt, x + cellW/2, y + cellH/2);
      }
    }
  }

  ctx.fillStyle = '#000';
  ctx.font = '12px Arial';
  ctx.textAlign = 'right';
  for (let r = 0; r < rows; r++) {
    const y = paddingTop + r * cellH + cellH/2;
    ctx.fillText(String(hours[r]), paddingLeft - 10, y);
  }

  ctx.save();
  ctx.translate(paddingLeft, paddingTop + h + 12);
  for (let c = 0; c < cols; c++) {
    const x = c * cellW + cellW/2;
    ctx.save();
    ctx.translate(x, 0);
    ctx.rotate(-Math.PI/4);
    ctx.textAlign = 'right';
    ctx.font = '11px Arial';
    const parts = headers[c]?.split('\n') ?? [headers[c]];
    for (let i = 0; i < parts.length; i++) {
      ctx.fillStyle = '#000';
      ctx.fillText(parts[i], 0, i * 12);
    }
    ctx.restore();
  }
  ctx.restore();

  const cbX = paddingLeft + w + 18;
  const cbY = paddingTop;
  const cbW = 26;
  const cbH = h;
  const grad = ctx.createLinearGradient(cbX, cbY + cbH, cbX, cbY);
  const steps = 40;
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    const val = cbMin + t * (cbMax - cbMin);
    grad.addColorStop(t, comfortToColor(val, cbMin, cbMax));
  }
  ctx.fillStyle = grad;
  ctx.fillRect(cbX, cbY, cbW, cbH);
  ctx.strokeStyle = '#000';
  ctx.strokeRect(cbX, cbY, cbW, cbH);

  ctx.textAlign = 'left';
  ctx.font = '12px Arial';
  for (let i = 0; i <= 5; i++) {
    const t = i / 5;
    const y = cbY + (1 - t) * cbH;
    const val = cbMin + t * (cbMax - cbMin);
    const txt = Number(val).toFixed(1);
    ctx.fillStyle = '#000';
    ctx.fillText(txt, cbX + cbW + 8, y);
  }
}

// ---------- Export helper ----------
function exportCanvasAsPNG(selector, filename = 'zone_heatmap.png') {
  const canvas = document.querySelector(selector);
  if (!canvas) return;
  const dataURL = canvas.toDataURL('image/png');
  const link = document.createElement('a');
  link.href = dataURL;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

// ---------- UI wiring ----------
async function generateZoneComfortHeatmapForSelectedRoom() {
  const sel = document.getElementById('room-select');
  const room = (sel && sel.value) ? sel.value : '';
  if (!room) { alert('Please select a room/zone first.'); return; }
  selectedRoomForHeatmap = room;
  const btn = document.getElementById('show-zone-heatmap-btn');
  const prevText = btn ? btn.innerText : null;
  if (btn) btn.innerText = 'Loading...';
  try {
    const { headers, hours, matrix } = await fetchComfortMatrixForRoom(room);
    console.debug('✅ generateZoneComfortHeatmapForSelectedRoom: matrix sample (FIXED)', matrix.slice(0,3));
    drawZoneComfortHeatmap('#heatmap', headers, hours, matrix);
  } catch (err) {
    console.error('Error generating heatmap:', err);
    alert('Failed to generate heatmap - see console.');
  } finally {
    if (btn && prevText) btn.innerText = prevText;
  }
}

document.addEventListener('DOMContentLoaded', async () => {
  try { await discoverRooms(); } catch (err) { console.error('discoverRooms failed', err); const sel = document.getElementById('room-select'); if (sel) sel.innerHTML = '<option value="">(error loading rooms)</option>'; }

  const dataTypeSelect = document.getElementById('data-type');
  if (dataTypeSelect) {
    dataTypeSelect.addEventListener('change', async () => { try { await discoverRooms(); } catch (err) { console.error('rediscover rooms failed', err); } });
  }

  const showBtn = document.getElementById('show-zone-heatmap-btn');
  const exportBtn = document.getElementById('export-heatmap-btn');
  if (showBtn) showBtn.addEventListener('click', generateZoneComfortHeatmapForSelectedRoom);
  if (exportBtn) exportBtn.addEventListener('click', () => { exportCanvasAsPNG('#heatmap', `zone_${(selectedRoomForHeatmap||'room')}_comfort_heatmap.png`); });

  const roomSelect = document.getElementById('room-select');
  if (roomSelect) roomSelect.addEventListener('change', () => { if (roomSelect.value) generateZoneComfortHeatmapForSelectedRoom(); });

  const loadBtn = document.getElementById('load-csv-btn');
  if (loadBtn) {
    loadBtn.addEventListener('click', async () => {
      const dataType = document.getElementById('data-type').value;
      const scenario = document.getElementById('scenario').value;
      const policy = document.getElementById('policy').value;
      try {
        const json = await fetchCsvJson(dataType, scenario, policy);
        displayCsvPreview(json);
      } catch (err) {
        console.error('Load CSV failed', err);
        alert('Failed to load CSV. See console.');
      }
    });
  }
});

// ---------- CSV preview ----------
function displayCsvPreview(json) {
  const container = document.getElementById('csv-preview');
  container.innerHTML = '';
  const rows = json.rows ?? json.data ?? json;
  if (!Array.isArray(rows) || rows.length === 0) {
    container.textContent = '(no rows)';
    return;
  }
  const table = document.createElement('table');
  table.style.borderCollapse = 'collapse';
  table.style.width = '100%';
  table.style.fontSize = '12px';
  const thead = document.createElement('thead');
  const headerRow = document.createElement('tr');
  const keys = Object.keys(rows[0] || {});
  for (const k of keys) {
    const th = document.createElement('th');
    th.textContent = k;
    th.style.border = '1px solid #eee';
    th.style.padding = '4px';
    th.style.background = '#fafafa';
    headerRow.appendChild(th);
  }
  thead.appendChild(headerRow);
  table.appendChild(thead);
  const tbody = document.createElement('tbody');
  for (let i = 0; i < Math.min(200, rows.length); i++) {
    const r = rows[i];
    const tr = document.createElement('tr');
    for (const k of keys) {
      const td = document.createElement('td');
      td.textContent = r?.[k] ?? '';
      td.style.border = '1px solid #eee';
      td.style.padding = '3px';
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  container.appendChild(table);
}
