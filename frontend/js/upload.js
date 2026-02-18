requireAuth();

const user = Storage.getUser();
document.getElementById('user-name').textContent = user.full_name;

// Elements
const uploadArea      = document.getElementById('upload-area');
const fileInput       = document.getElementById('file-input');
const filePreview     = document.getElementById('file-preview');
const imagePreview    = document.getElementById('image-preview');
const videoPreview    = document.getElementById('video-preview');
const fileInfo        = document.getElementById('file-info');
const progressSection = document.getElementById('progress-section');
const resultsSection  = document.getElementById('results-section');

let selectedFile   = null;
let currentType    = 'image';

// Check URL for mode parameter
const urlParams = new URLSearchParams(window.location.search);
const typeParam = urlParams.get('type');
const modeParam = urlParams.get('mode');

if (typeParam === 'video') {
    document.querySelector('.tab-btn[data-type="video"]')?.click();
}

// Tab switching
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentType = btn.dataset.type;
        
        if (currentType === 'image') {
            fileInput.accept = 'image/*';
            document.getElementById('file-types').textContent = 'Supported: JPG, PNG, GIF, BMP';
        } else {
            fileInput.accept = 'video/*';
            document.getElementById('file-types').textContent = 'Supported: MP4, AVI, MOV, MKV';
        }
        resetUpload();
    });
});

// Drag and drop
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

// File selection
function handleFileSelect(file) {
    const isImage = file.type.startsWith('image/');
    const isVideo = file.type.startsWith('video/');

    if (currentType === 'image' && !isImage) {
        Toast.error('Please select an image file');
        return;
    }
    if (currentType === 'video' && !isVideo) {
        Toast.error('Please select a video file');
        return;
    }
    if (file.size > 100 * 1024 * 1024) {
        Toast.error('File size must be under 100 MB');
        return;
    }

    selectedFile = file;
    showFilePreview(file);
}

function showFilePreview(file) {
    uploadArea.style.display  = 'none';
    filePreview.style.display = 'block';

    const reader = new FileReader();
    reader.onload = e => {
        if (file.type.startsWith('image/')) {
            imagePreview.src          = e.target.result;
            imagePreview.style.display = 'block';
            videoPreview.style.display = 'none';
        } else {
            videoPreview.src          = e.target.result;
            videoPreview.style.display = 'block';
            imagePreview.style.display = 'none';
        }
    };
    reader.readAsDataURL(file);

    const mb = (file.size / 1024 / 1024).toFixed(2);
    fileInfo.innerHTML = `
        <p><strong>File:</strong> ${file.name}</p>
        <p><strong>Size:</strong> ${mb} MB</p>
        <p><strong>Type:</strong> ${file.type || 'unknown'}</p>
    `;
}

// Reset
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

// Analyse
async function analyzeFile() {
    if (!selectedFile) {
        Toast.error('Please select a file first');
        return;
    }

    filePreview.style.display     = 'none';
    progressSection.style.display = 'block';

    try {
        const endpoint = currentType === 'image'
            ? API_CONFIG.ENDPOINTS.UPLOAD_IMAGE
            : API_CONFIG.ENDPOINTS.UPLOAD_VIDEO;

        const url = `${API_CONFIG.BASE_URL}${endpoint}`;
        
        const xhr = new XMLHttpRequest();
        const formData = new FormData();
        formData.append('file', selectedFile);

        xhr.upload.addEventListener('progress', e => {
            if (e.lengthComputable) {
                const pct = (e.loaded / e.total) * 80;
                setProgress(pct, 'Uploading…');
            }
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
            } catch(err) {
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

    } catch (err) {
        progressSection.style.display = 'none';
        filePreview.style.display     = 'block';
        Toast.error(err.message || 'Analysis failed');
    }
}

function setProgress(pct, msg) {
    document.getElementById('progress-fill').style.width = `${Math.round(pct)}%`;
    document.getElementById('progress-text').textContent = `${msg} ${Math.round(pct)}%`;
}

// Results
function showResults(data) {
    progressSection.style.display = 'none';
    resultsSection.style.display  = 'block';

    const isFake = data.result.toLowerCase() === 'fake';
    const conf   = parseFloat(data.confidence);

    let confLabel = '';
    if (conf >= 90)      confLabel = 'Very High Confidence';
    else if (conf >= 75) confLabel = 'High Confidence';
    else if (conf >= 60) confLabel = 'Moderate Confidence';
    else                 confLabel = 'Low Confidence';

    const explanation = isFake
        ? _fakeExplanation(conf)
        : `This content shows no significant manipulation artefacts. The model is ${confLabel.toLowerCase()} it is authentic.`;

    const resultStatus = document.getElementById('result-status');
    resultStatus.className = 'result-status ' + (isFake ? 'fake' : 'real');
    resultStatus.innerHTML = `
        <div class="status-icon">${isFake ? '⚠️' : '✅'}</div>
        <h3>${isFake ? 'FAKE DETECTED' : 'AUTHENTIC'}</h3>
        <p>${explanation}</p>
    `;

    document.getElementById('confidence-value').textContent  = `${conf}% (${confLabel})`;
    document.getElementById('processing-time').textContent   = `${data.processing_time}s`;

    const bar = document.getElementById('confidence-bar');
    if (bar) {
        bar.style.width      = `${conf}%`;
        bar.style.background = isFake ? '#dc2626' : '#059669';
    }

    const demoBanner = document.getElementById('demo-banner');
    if (demoBanner) {
        demoBanner.style.display = data.is_demo ? 'block' : 'none';
    }

    const detailLink = document.getElementById('detail-link');
    if (detailLink && data.detection_id) {
        detailLink.href = `results.html?id=${data.detection_id}`;
        detailLink.style.display = 'inline-block';
    }

    if (data.is_demo) {
        Toast.warning('Running in demo mode - results are random');
    } else {
        Toast.success('Analysis complete!');
    }
}

function _fakeExplanation(confidence) {
    if (confidence >= 90)
        return 'Strong manipulation artefacts detected. The model is very confident this content has been artificially generated or altered.';
    if (confidence >= 75)
        return 'Probable manipulation detected. The model found significant signs of artificial generation or editing.';
    if (confidence >= 60)
        return 'Possible manipulation detected. Some artefacts were found, but the model has moderate confidence.';
    return 'Weak manipulation signal. The model flagged some irregularities, but confidence is low — treat with caution.';
}

// Mobile nav toggle
function toggleMobileNav() {
    document.getElementById('mobile-nav')?.classList.toggle('open');
}