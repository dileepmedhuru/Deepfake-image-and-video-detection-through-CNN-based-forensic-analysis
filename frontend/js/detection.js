requireAuth();

// Load user info
const user = Storage.getUser();
document.getElementById('user-name').textContent = user.full_name;

// Get detection ID from URL
const urlParams = new URLSearchParams(window.location.search);
const detectionId = urlParams.get('id');

if (!detectionId) {
    showError();
} else {
    loadDetectionDetails(detectionId);
}

async function loadDetectionDetails(id) {
    try {
        const data = await API.request(API_CONFIG.ENDPOINTS.DETECTION_DETAIL + '/' + id);
        const detection = data.detection;
        
        if (!detection) {
            showError();
            return;
        }
        
        displayResults(detection);
        
    } catch (error) {
        console.error('Failed to load detection:', error);
        showError();
    }
}

function displayResults(detection) {
    document.getElementById('loading').style.display = 'none';
    document.getElementById('result-details').style.display = 'block';
    
    // File name
    document.getElementById('file-name-display').textContent = detection.file_name;
    
    // Result status
    const resultStatus = document.getElementById('result-status');
    const isFake = detection.result.toLowerCase() === 'fake';
    
    resultStatus.className = 'result-status ' + (isFake ? 'fake' : 'real');
    resultStatus.innerHTML = `
        <div class="status-icon">${isFake ? '⚠️' : '✅'}</div>
        <h3>${isFake ? 'FAKE DETECTED' : 'AUTHENTIC'}</h3>
        <p>This content appears to be ${isFake ? 'manipulated or artificially generated' : 'genuine and unaltered'}</p>
    `;
    
    // Details
    document.getElementById('result-value').innerHTML = 
        `<span class="result ${detection.result}">${detection.result.toUpperCase()}</span>`;
    document.getElementById('confidence-value').textContent = detection.confidence + '%';
    document.getElementById('processing-time').textContent = detection.processing_time + ' seconds';
    document.getElementById('file-type').textContent = detection.file_type.toUpperCase();
    document.getElementById('analyzed-on').textContent = 
        new Date(detection.created_at).toLocaleString();
    
    // Metadata
    if (detection.metadata) {
        const metadata = typeof detection.metadata === 'string' ? 
            JSON.parse(detection.metadata) : detection.metadata;
        
        let metadataHtml = '';
        
        // File info
        if (metadata.file_info) {
            metadataHtml += '<h4 style="margin-top: 1rem; color: #2d3748;">File Information</h4>';
            for (const [key, value] of Object.entries(metadata.file_info)) {
                const label = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                metadataHtml += `
                    <div class="detail-item">
                        <span class="label">${label}:</span>
                        <span class="value">${formatMetadataValue(key, value)}</span>
                    </div>
                `;
            }
        }
        
        // Detection method
        if (metadata.detection_method) {
            metadataHtml += '<h4 style="margin-top: 1rem; color: #2d3748;">Detection Method</h4>';
            metadataHtml += `
                <div class="detail-item">
                    <span class="label">Method:</span>
                    <span class="value">${metadata.detection_method}</span>
                </div>
            `;
        }
        
        // Additional info
        if (metadata.num_faces_detected !== undefined) {
            metadataHtml += `
                <div class="detail-item">
                    <span class="label">Faces Detected:</span>
                    <span class="value">${metadata.num_faces_detected}</span>
                </div>
            `;
        }
        
        if (metadata.frames_analyzed) {
            metadataHtml += `
                <div class="detail-item">
                    <span class="label">Frames Analyzed:</span>
                    <span class="value">${metadata.frames_analyzed}</span>
                </div>
            `;
        }
        
        if (metadata.model_version) {
            metadataHtml += `
                <div class="detail-item">
                    <span class="label">Model Version:</span>
                    <span class="value">${metadata.model_version}</span>
                </div>
            `;
        }
        
        document.getElementById('metadata-content').innerHTML = metadataHtml;
    }
}

function formatMetadataValue(key, value) {
    if (key.includes('size_bytes')) {
        const mb = (value / (1024 * 1024)).toFixed(2);
        return `${mb} MB`;
    }
    if (key.includes('duration')) {
        return `${value.toFixed(2)} seconds`;
    }
    if (typeof value === 'number') {
        return value.toLocaleString();
    }
    return value;
}

function showError() {
    document.getElementById('loading').style.display = 'none';
    document.getElementById('error-section').style.display = 'block';
}