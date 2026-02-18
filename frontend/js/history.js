requireAuth();
const user = Storage.getUser();
document.getElementById('user-name').textContent = user.full_name;

let currentPage = 1;
const perPage   = 20;
let totalPages  = 1;

function getFilters() {
    return {
        type:   document.getElementById('filter-type').value,
        result: document.getElementById('filter-result').value,
        search: document.getElementById('search-input')?.value.trim() || '',
        sort:   document.getElementById('sort-by')?.value   || 'date',
        order:  document.getElementById('sort-order')?.value || 'desc',
    };
}

async function loadHistory(page = 1) {
    const f = getFilters();
    const params = new URLSearchParams({ page, limit: perPage });
    if (f.type   !== 'all') params.append('type',   f.type);
    if (f.result !== 'all') params.append('result', f.result);
    if (f.search)           params.append('search', f.search);
    params.append('sort',  f.sort);
    params.append('order', f.order);

    try {
        const data = await API.request(`${API_CONFIG.ENDPOINTS.HISTORY}?${params}`);
        totalPages  = data.pages || 1;
        currentPage = data.page  || 1;
        renderHistory(data.history, data.total);
        renderPagination();
    } catch (err) {
        document.getElementById('history-list').innerHTML =
            '<p class="error">Failed to load history.</p>';
    }
}

function renderHistory(items, total) {
    const countEl = document.getElementById('total-count');
    if (countEl) countEl.textContent = `${total} detection${total!==1?'s':''}`;

    const list = document.getElementById('history-list');
    if (!items.length) {
        list.innerHTML = '<p class="empty-state">No detections found.</p>'; return;
    }
    list.innerHTML = items.map(d => {
        const fmt = new Date(d.created_at).toLocaleString();
        return `
        <div class="history-card" id="card-${d.id}">
            <div class="history-icon">${d.file_type==='image'?'🖼️':'🎥'}</div>
            <div class="history-details">
                <h3>${_esc(d.file_name)}</h3>
                <div class="history-info">
                    <span class="result-badge ${d.result}">${d.result.toUpperCase()}</span>
                    <span class="confidence">Confidence: ${d.confidence}%</span>
                    <span class="time">Processing: ${d.processing_time}s</span>
                    ${d.is_demo?'<span class="demo-tag">DEMO</span>':''}
                </div>
                <p class="timestamp">${fmt}</p>
            </div>
            <div class="history-actions">
                <button class="btn btn-sm btn-secondary" onclick="viewDetails(${d.id})">View</button>
                <button class="btn btn-sm btn-danger"    onclick="deleteDetection(${d.id})">Delete</button>
            </div>
        </div>`;
    }).join('');
}

function renderPagination() {
    const c = document.getElementById('pagination');
    if (!c) return;
    if (totalPages <= 1) { c.innerHTML=''; return; }
    let h = '';
    if (currentPage > 1) h += `<button class="btn btn-sm btn-secondary" onclick="goToPage(${currentPage-1})">← Prev</button>`;
    h += `<span class="page-info"> Page ${currentPage} of ${totalPages} </span>`;
    if (currentPage < totalPages) h += `<button class="btn btn-sm btn-secondary" onclick="goToPage(${currentPage+1})">Next →</button>`;
    c.innerHTML = h;
}

function goToPage(p) { currentPage=p; loadHistory(p); window.scrollTo({top:0,behavior:'smooth'}); }
function viewDetails(id) { window.location.href=`results.html?id=${id}`; }

async function deleteDetection(id) {
    if (!confirm('Delete this detection? This cannot be undone.')) return;
    try {
        await API.delete(`${API_CONFIG.ENDPOINTS.DETECTION_DETAIL}/${id}`);
        document.getElementById(`card-${id}`)?.remove();
        Toast.success('Detection deleted.');
        loadHistory(currentPage);
    } catch (err) { Toast.error('Delete failed: '+(err.message||'Unknown error')); }
}

function exportCSV() {
    window.location.href = `${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.EXPORT_CSV}`;
}

// Debounced search
let _searchTimer;
function onSearchInput() {
    clearTimeout(_searchTimer);
    _searchTimer = setTimeout(() => loadHistory(1), 400);
}

// Filter/sort listeners
['filter-type','filter-result','sort-by','sort-order'].forEach(id => {
    document.getElementById(id)?.addEventListener('change', () => loadHistory(1));
});
document.getElementById('search-input')?.addEventListener('input', onSearchInput);

function _esc(s) {
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

loadHistory(1);
