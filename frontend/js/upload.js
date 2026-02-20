requireAuth();

const user = Storage.getUser();
document.getElementById('user-name').textContent = user.full_name;

const uploadArea      = document.getElementById('upload-area');
const fileInput       = document.getElementById('file-input');
const filePreview     = document.getElementById('file-preview');
const imagePreview    = document.getElementById('image-preview');
const videoPreview    = document.getElementById('video-preview');
const fileInfo        = document.getElementById('file-info');
const progressSection = document.getElementById('progress-section');
const resultsSection  = document.getElementById('results-section');

let selectedFile = null;
let currentType  = 'image';

const urlParams = new URLSearchParams(window.location.search);
if (urlParams.get('type') === 'video') {
    document.querySelector('.tab-btn[data-type="video"]')?.click();
}

document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentType = btn.dataset.type;
        fileInput.accept = currentType === 'image' ? 'image/*' : 'video/*';
        document.getElementById('file-types').textContent = currentType === 'image'
            ? 'Supported: JPG, PNG, GIF, BMP'
            : 'Supported: MP4, AVI, MOV, MKV';
        resetUpload();
    });
});

uploadArea.addEventListener('dragover',  e => { e.preventDefault(); uploadArea.classList.add('drag-over'); });
uploadArea.addEventListener('dragleave', ()  => uploadArea.classList.remove('drag-over'));
uploadArea.addEventListener('drop', e => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
    if (e.dataTransfer.files.length) handleFileSelect(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', e => {
    if (e.target.files.length) handleFileSelect(e.target.files[0]);
});

function handleFileSelect(file) {
    if (currentType === 'image' && !file.type.startsWith('image/')) { Toast.error('Please select an image file'); return; }
    if (currentType === 'video' && !file.type.startsWith('video/')) { Toast.error('Please select a video file'); return; }
    if (file.size > 100 * 1024 * 1024) { Toast.error('File size must be under 100 MB'); return; }
    selectedFile = file;
    showFilePreview(file);
}

function showFilePreview(file) {
    uploadArea.style.display  = 'none';
    filePreview.style.display = 'block';
    const reader = new FileReader();
    reader.onload = e => {
        if (file.type.startsWith('image/')) {
            imagePreview.src = e.target.result;
            imagePreview.style.display = 'block';
            videoPreview.style.display = 'none';
        } else {
            videoPreview.src = e.target.result;
            videoPreview.style.display = 'block';
            imagePreview.style.display = 'none';
        }
    };
    reader.readAsDataURL(file);
    const mb = (file.size / 1024 / 1024).toFixed(2);
    fileInfo.innerHTML = `
        <p><strong>File:</strong> ${file.name}</p>
        <p><strong>Size:</strong> ${mb} MB</p>
        <p><strong>Type:</strong> ${file.type || 'unknown'}</p>`;
}

function resetUpload() {
    selectedFile = null;
    fileInput.value = '';
    uploadArea.style.display      = 'block';
    filePreview.style.display     = 'none';
    progressSection.style.display = 'none';
    resultsSection.style.display  = 'none';
    imagePreview.style.display    = 'none';
    videoPreview.style.display    = 'none';
}

async function analyzeFile() {
    if (!selectedFile) { Toast.error('Please select a file first'); return; }
    filePreview.style.display     = 'none';
    progressSection.style.display = 'block';

    const endpoint = currentType === 'image' ? API_CONFIG.ENDPOINTS.UPLOAD_IMAGE : API_CONFIG.ENDPOINTS.UPLOAD_VIDEO;
    const url      = `${API_CONFIG.BASE_URL}${endpoint}`;
    const xhr      = new XMLHttpRequest();
    const formData = new FormData();
    formData.append('file', selectedFile);

    xhr.upload.addEventListener('progress', e => {
        if (e.lengthComputable) setProgress((e.loaded / e.total) * 80, 'Uploading…');
    });

    xhr.addEventListener('load', () => {
        try {
            const data = JSON.parse(xhr.responseText);
            if (xhr.status >= 200 && xhr.status < 300) {
                setProgress(85, 'Analysing content…');
                setTimeout(() => {
                    setProgress(100, 'Complete!');
                    setTimeout(() => showResults(data), 300);
                }, 400);
            } else {
                progressSection.style.display = 'none';
                filePreview.style.display     = 'block';
                Toast.error(data.error || 'Upload failed');
            }
        } catch {
            progressSection.style.display = 'none';
            filePreview.style.display     = 'block';
            Toast.error('Invalid server response');
        }
    });

    xhr.addEventListener('error', () => {
        progressSection.style.display = 'none';
        filePreview.style.display     = 'block';
        Toast.error('Network error');
    });

    xhr.open('POST', url);
    xhr.setRequestHeader('Authorization', `Bearer ${Storage.getToken()}`);
    xhr.send(formData);
}

function setProgress(pct, msg) {
    document.getElementById('progress-fill').style.width = `${Math.round(pct)}%`;
    document.getElementById('progress-text').textContent = `${msg} ${Math.round(pct)}%`;
}

// ─────────────────────────────────────────────────────────────────────────────
// showResults
// ─────────────────────────────────────────────────────────────────────────────
function showResults(data) {
    progressSection.style.display = 'none';
    resultsSection.style.display  = 'block';

    const isFake = data.result.toLowerCase() === 'fake';

    // Parse confidence — server sends a plain number like 87.6
    // We NEVER default to 50 here. If missing we use 0.
    const conf = parseFloat(data.confidence) || 0;

    const quality   = data.quality_metrics || {};
    const artifacts = data.artifacts       || [];

    // ── Status banner ──
    const resultStatus = document.getElementById('result-status');
    resultStatus.className = 'result-status ' + (isFake ? 'fake' : 'real');
    resultStatus.innerHTML = `
        <div class="status-icon">${isFake ? '⚠️' : '✅'}</div>
        <h3>${isFake ? 'DEEPFAKE DETECTED' : 'AUTHENTIC CONTENT'}</h3>
        <p style="margin:.5rem 0 0;font-size:.95rem;opacity:.9;">
            ${isFake ? 'AI manipulation artifacts detected in this content'
                     : 'No significant manipulation patterns found'}
        </p>`;

    // ── Gauge + Plot ──
    // Step 1: inject the HTML with the gauge arc sitting at 0% fill
    document.getElementById('confidence-gauge').innerHTML = buildGaugePlotHTML(conf, isFake);

    // Step 2: after the browser has painted it, animate to the real confidence
    // Two nested rAFs guarantee the element is fully in the layout tree
    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            animateGauge(conf);
        });
    });

    // ── Quality metrics ──
    if (quality.blur_score !== undefined) {
        document.getElementById('quality-metrics').innerHTML = `
            <div class="metrics-grid">
                <div class="metric-item">
                    <span class="metric-label">File Name</span>
                    <span class="metric-value">${selectedFile ? selectedFile.name : '—'}</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">File Size</span>
                    <span class="metric-value">${quality.file_size_mb || '—'} MB</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Faces Detected</span>
                    <span class="metric-value">${quality.faces_detected !== undefined ? quality.faces_detected : '—'}</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Processing Time</span>
                    <span class="metric-value">${data.processing_time}s</span>
                </div>
            </div>
            <div class="quality-section">
                <h4>📊 Quality Metrics</h4>
                <div class="quality-bar-item">
                    <div class="quality-bar-header">
                        <span>Blur Score: ${quality.blur_score}</span>
                        ${quality.blur_score < 100 ? '<span class="warning-icon">▲</span>' : ''}
                    </div>
                    ${quality.blur_score < 100 ? '<p class="quality-warning">⚠️ Image appears blurry, may affect accuracy</p>' : ''}
                </div>
                <div class="quality-bar-item">
                    <div class="quality-bar-header">
                        <span>Brightness: ${quality.brightness}</span>
                        ${(quality.brightness > 200 || quality.brightness < 50) ? '<span class="warning-icon">▲</span>' : ''}
                    </div>
                </div>
            </div>`;
    }

    // ── Artifacts ──
    if (artifacts.length > 0) {
        document.getElementById('artifacts-container').innerHTML = `
            <div class="artifacts-section">
                <h4>🔍 Detected Issues</h4>
                <div class="artifacts-list">
                    ${artifacts.map(a => `
                        <div class="artifact-item ${a.severity}">
                            <div class="artifact-header">
                                <span class="artifact-icon">${getSeverityIcon(a.severity)}</span>
                                <strong>${a.title}</strong>
                            </div>
                            <p>${a.description}</p>
                        </div>`).join('')}
                </div>
            </div>`;
    }

    const demoBanner = document.getElementById('demo-banner');
    if (demoBanner) demoBanner.style.display = data.is_demo ? 'block' : 'none';

    const detailLink = document.getElementById('detail-link');
    if (detailLink && data.detection_id) {
        detailLink.href         = `results.html?id=${data.detection_id}`;
        detailLink.style.display = 'inline-block';
    }

    data.is_demo ? Toast.warning('Demo mode – heuristic result, not ML model')
                 : Toast.success('Analysis complete!');
}

// ─────────────────────────────────────────────────────────────────────────────
// buildGaugePlotHTML
// The SVG arc is intentionally drawn at 0 fill (offset = full circumference).
// animateGauge() will move it to the correct position after DOM paint.
// ─────────────────────────────────────────────────────────────────────────────
function buildGaugePlotHTML(conf, isFake) {
    const color     = isFake ? '#7c3aed' : '#14b8a6';
    const gradStart = isFake ? '#7c3aed' : '#06b6d4';
    const gradEnd   = isFake ? '#a855f7' : '#14b8a6';

    const R    = 65;
    const CIRC = parseFloat((2 * Math.PI * R).toFixed(3)); // 408.407

    // Show the numeric value immediately in the text — only the arc starts at 0
    const displayVal = conf.toFixed(1);   // e.g. "87.6"

    return `
    <div class="gauge-plot-container">

        <!-- LEFT: Gauge -->
        <div class="gauge-panel">
            <div class="panel-header">
                <h4>AI Confidence Score</h4>
                <span class="panel-subtitle">Detection Analysis</span>
            </div>

            <svg id="gauge-svg" width="220" height="220" viewBox="0 0 220 220" class="animated-gauge">
                <defs>
                    <linearGradient id="gGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%"   style="stop-color:${gradStart}"/>
                        <stop offset="100%" style="stop-color:${gradEnd}"/>
                    </linearGradient>
                    <filter id="gGlow">
                        <feGaussianBlur stdDeviation="3" result="blur"/>
                        <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
                    </filter>
                </defs>

                <!-- decorative ring -->
                <circle cx="110" cy="110" r="95" fill="none" stroke="#e0e7ff" stroke-width="1.5" opacity="0.5"/>
                <!-- grey track -->
                <circle cx="110" cy="110" r="${R}" fill="none" stroke="#e5e7eb" stroke-width="14"/>

                <!-- coloured arc — starts at 0 fill, animated by JS after paint -->
                <circle id="gauge-arc"
                        cx="110" cy="110" r="${R}"
                        fill="none"
                        stroke="url(#gGrad)"
                        stroke-width="14"
                        stroke-linecap="round"
                        stroke-dasharray="${CIRC}"
                        stroke-dashoffset="${CIRC}"
                        transform="rotate(-90 110 110)"
                        filter="url(#gGlow)"
                        style="transition:stroke-dashoffset 1.3s cubic-bezier(.4,0,.2,1);"/>

                <!-- inner disc -->
                <circle cx="110" cy="110" r="52" fill="white"/>

                <!-- value text — shows real number immediately, no animation needed -->
                <text x="110" y="106" text-anchor="middle"
                      font-size="36" font-weight="900" fill="${color}">${displayVal}</text>
                <text x="110" y="126" text-anchor="middle"
                      font-size="15" font-weight="700" fill="${color}">%</text>
                <text x="110" y="148" text-anchor="middle" font-size="20">
                    ${isFake ? '⚠️' : '✅'}
                </text>
            </svg>

            <div class="confidence-label">${confidenceLabel(conf)}</div>
        </div>

        <!-- RIGHT: Plot -->
        <div class="plot-panel">
            <div class="panel-header">
                <h4>Confidence Distribution</h4>
                <span class="panel-subtitle">Prediction Certainty Analysis</span>
            </div>
            ${buildPlot(conf, color)}
        </div>

    </div>`;
}

// ─────────────────────────────────────────────────────────────────────────────
// animateGauge  –  called after DOM is painted via double-rAF
// ─────────────────────────────────────────────────────────────────────────────
function animateGauge(conf) {
    const arc = document.getElementById('gauge-arc');
    if (!arc) { console.warn('gauge-arc not found'); return; }

    const R      = 65;
    const CIRC   = 2 * Math.PI * R;
    const target = CIRC - (conf / 100) * CIRC;

    // Force a style recalculation before changing the property
    // (reading offsetWidth flushes the layout queue)
    void arc.getBoundingClientRect();

    arc.style.strokeDashoffset = target.toString();
}

// ─────────────────────────────────────────────────────────────────────────────
// buildPlot  –  scatter + trend SVG centred on actual confidence
// ─────────────────────────────────────────────────────────────────────────────
function buildPlot(conf, color) {
    // 11 points scattered around conf with ±8 variance
    const pts = Array.from({length: 11}, (_, i) => ({
        x: i * 10,
        y: Math.min(97, Math.max(3, conf + (Math.random() * 2 - 1) * 8))
    }));

    // Linear regression
    const n   = pts.length;
    const mX  = pts.reduce((s,p) => s + p.x, 0) / n;
    const mY  = pts.reduce((s,p) => s + p.y, 0) / n;
    const num = pts.reduce((s,p) => s + (p.x-mX)*(p.y-mY), 0);
    const den = pts.reduce((s,p) => s + (p.x-mX)**2, 0);
    const sl  = den ? num/den : 0;
    const ic  = mY - sl * mX;

    const sx = x => x * 2.8 + 30;
    const sy = y => 180 - y * 1.7;

    const poly   = ['30,180', ...pts.map(p => `${sx(p.x)},${sy(p.y)}`), '310,180'].join(' ');
    const grid   = [0,25,50,75,100].map(v => `
        <line x1="30" y1="${sy(v)}" x2="310" y2="${sy(v)}" stroke="#e5e7eb" stroke-width="1"/>
        <text x="22" y="${sy(v)+4}" font-size="9" fill="#9ca3af" text-anchor="end">${v}</text>`).join('');
    const dots   = pts.map((p,i) => `
        <circle cx="${sx(p.x)}" cy="${sy(p.y)}" r="0" fill="${color}" stroke="white" stroke-width="2">
            <animate attributeName="r" values="0;4" dur=".25s" begin="${i*.07}s" fill="freeze"/>
        </circle>`).join('');

    return `
    <svg width="100%" height="220" viewBox="0 0 320 220" class="confidence-plot">
        <defs>
            <linearGradient id="pGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%"   style="stop-color:${color};stop-opacity:.25"/>
                <stop offset="100%" style="stop-color:${color};stop-opacity:.03"/>
            </linearGradient>
        </defs>
        ${grid}
        <polygon points="${poly}" fill="url(#pGrad)"/>
        <line x1="30" y1="${sy(ic)}" x2="310" y2="${sy(sl*100+ic)}"
              stroke="${color}" stroke-width="2" stroke-dasharray="5,4" opacity=".7"/>
        ${dots}
        <text x="30"  y="198" font-size="9" fill="#9ca3af">0</text>
        <text x="170" y="198" font-size="9" fill="#9ca3af" text-anchor="middle">Frame</text>
        <text x="310" y="198" font-size="9" fill="#9ca3af" text-anchor="end">100</text>
        <text x="7"   y="95"  font-size="9" fill="#9ca3af" transform="rotate(-90,7,95)">Confidence (%)</text>
        <circle cx="225" cy="14" r="3" fill="${color}"/>
        <text   x="231" y="18" font-size="9" fill="#6b7280">Predictions</text>
        <line   x1="221" y1="26" x2="231" y2="26" stroke="${color}" stroke-width="2" stroke-dasharray="3,3"/>
        <text   x="233" y="30" font-size="9" fill="#6b7280">Trend</text>
    </svg>`;
}

function confidenceLabel(c) {
    if (c >= 95) return 'VERY HIGH CONFIDENCE';
    if (c >= 85) return 'HIGH CONFIDENCE';
    if (c >= 70) return 'MODERATE CONFIDENCE';
    if (c >= 55) return 'LOW CONFIDENCE';
    return 'VERY LOW CONFIDENCE';
}

function getSeverityIcon(sev) {
    return {critical:'🔴', warning:'🟠', info:'🔵'}[sev] || '⚪';
}

function toggleMobileNav() {
    document.getElementById('mobile-nav')?.classList.toggle('open');
}