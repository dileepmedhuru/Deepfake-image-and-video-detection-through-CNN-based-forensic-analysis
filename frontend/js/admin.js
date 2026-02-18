// ── Shared utilities ──────────────────────────────────────────────────────
function showError(msg) {
    const el = document.getElementById('error-message');
    if (el) { el.textContent = msg; el.style.display = 'block'; }
}
function showSuccess(msg) {
    const el = document.getElementById('success-message');
    if (el) { el.textContent = msg; el.style.display = 'block'; }
}
function hideMessages() {
    ['error-message','success-message'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.style.display = 'none';
    });
}
function setLoading(on) {
    const btn    = document.querySelector('button[type="submit"]');
    const text   = document.getElementById('btn-text');
    const loader = document.getElementById('btn-loader');
    if (btn)    btn.disabled = on;
    if (text)   text.style.display   = on ? 'none'         : 'inline';
    if (loader) loader.style.display = on ? 'inline-block' : 'none';
}

// ── Admin Login ───────────────────────────────────────────────────────────
const adminLoginForm = document.getElementById('admin-login-form');
if (adminLoginForm) {
    adminLoginForm.addEventListener('submit', async e => {
        e.preventDefault();
        hideMessages();

        const email    = document.getElementById('email').value.trim().toLowerCase();
        const password = document.getElementById('password').value;

        setLoading(true);
        try {
            const res = await API.request(API_CONFIG.ENDPOINTS.LOGIN, {
                method: 'POST',
                body: JSON.stringify({ email, password }),
                skipAuth: true,
            });

            if (!res.user.is_admin) {
                return showError('Access denied. Admin privileges required.');
            }

            Storage.setToken(res.token);
            Storage.setUser(res.user);
            showSuccess('Admin login successful! Redirecting…');

            setTimeout(() => {
                if (res.user.force_password_change) {
                    window.location.href = '/profile.html';
                } else {
                    Redirect.toAdminDashboard();
                }
            }, 1000);

        } catch (err) {
            showError(err.message || 'Login failed.');
        } finally {
            setLoading(false);
        }
    });
}

// ── Admin Dashboard ───────────────────────────────────────────────────────
if (window.location.pathname.includes('admin-dashboard')) {
    requireAdmin();

    const user = Storage.getUser();
    const nameEl = document.getElementById('admin-name');
    if (nameEl) nameEl.textContent = user.full_name;

    // Pagination state
    let usersPage      = 1;
    let detectionsPage = 1;
    const perPage      = 20;

    // ── Stats + Chart ──────────────────────────────────────────────────────
    async function loadStats() {
        try {
            const data = await API.request(API_CONFIG.ENDPOINTS.ADMIN_DASHBOARD);
            document.getElementById('total-users').textContent      = data.total_users      || 0;
            document.getElementById('total-detections').textContent = data.total_detections || 0;
            document.getElementById('fake-detections').textContent  = data.fake_detections  || 0;
            document.getElementById('real-detections').textContent  = data.real_detections  || 0;

            // Draw chart if canvas exists and Chart.js available
            if (data.daily_activity && window.Chart) {
                drawActivityChart(data.daily_activity);
            }
        } catch (err) {
            console.error('Failed to load stats:', err);
        }
    }

    function drawActivityChart(daily) {
        const ctx = document.getElementById('activity-chart');
        if (!ctx) return;

        new Chart(ctx, {
            type: 'bar',
            data: {
                labels:   daily.map(d => d.date),
                datasets: [{
                    label:           'Detections',
                    data:            daily.map(d => d.count),
                    backgroundColor: 'rgba(6, 182, 212, 0.6)',
                    borderColor:     'rgba(6, 182, 212, 1)',
                    borderWidth:     1,
                    borderRadius:    4,
                }],
            },
            options: {
                responsive: true,
                plugins: { legend: { display: false } },
                scales:  { y: { beginAtZero: true, ticks: { stepSize: 1 } } },
            },
        });
    }

    // ── Users table ────────────────────────────────────────────────────────
    async function loadUsers(page = 1) {
        try {
            const data = await API.request(
                `${API_CONFIG.ENDPOINTS.ADMIN_USERS}?page=${page}&limit=${perPage}`
            );
            const container = document.getElementById('users-list');

            if (!data.users.length) {
                container.innerHTML = '<p class="empty-state">No users found.</p>';
                return;
            }

            container.innerHTML = `
                <div class="users-summary">
                    <span>Showing <strong>${data.users.length}</strong> of <strong>${data.total}</strong> registered users</span>
                </div>
                <table class="admin-table">
                    <thead><tr>
                        <th>#</th>
                        <th>Full Name</th>
                        <th>Email Address</th>
                        <th>Role</th>
                        <th>Detections</th>
                        <th>Joined</th>
                        <th>Actions</th>
                    </tr></thead>
                    <tbody>
                    ${data.users.map(u => `
                        <tr class="user-row" style="cursor:pointer;"
                            onclick="viewUserDetail(${u.id})">
                            <td>${u.id}</td>
                            <td>
                                <div class="user-name-cell">
                                    <div class="user-avatar">${_esc(u.full_name).charAt(0).toUpperCase()}</div>
                                    <span>${_esc(u.full_name)}</span>
                                </div>
                            </td>
                            <td><a href="mailto:${_esc(u.email)}" onclick="event.stopPropagation()">${_esc(u.email)}</a></td>
                            <td>
                                <span class="role-badge ${u.is_admin ? 'admin' : 'user'}">
                                    ${u.is_admin ? '👑 Admin' : '👤 User'}
                                </span>
                            </td>
                            <td><span class="detection-count">${u.detection_count || 0}</span></td>
                            <td>${new Date(u.created_at).toLocaleDateString('en-GB', {day:'2-digit',month:'short',year:'numeric'})}</td>
                            <td onclick="event.stopPropagation()">
                                <button class="btn btn-sm btn-secondary"
                                        onclick="viewUserDetail(${u.id})">
                                    View
                                </button>
                                ${!u.is_admin
                                    ? `<button class="btn btn-sm btn-danger"
                                              onclick="deleteUser(${u.id}, '${_esc(u.email)}')">
                                         Delete
                                       </button>`
                                    : ''}
                            </td>
                        </tr>
                    `).join('')}
                    </tbody>
                </table>
                ${_paginationHtml(data.page, data.pages, 'loadUsers')}
            `;
        } catch (err) {
            document.getElementById('users-list').innerHTML =
                '<p class="error">Failed to load users.</p>';
        }
    }

    // ── User Detail Modal ──────────────────────────────────────────────────
    async function viewUserDetail(userId) {
        // Show modal with loading state
        _showModal(`
            <div class="modal-loading">
                <div class="loader" style="width:40px;height:40px;border-width:4px;"></div>
                <p>Loading user details…</p>
            </div>
        `);

        try {
            const data = await API.request(
                `${API_CONFIG.ENDPOINTS.ADMIN_USER_DETAIL}/${userId}`
            );
            const u = data.user;

            const joinDate = new Date(u.created_at).toLocaleDateString('en-GB', {
                day: '2-digit', month: 'long', year: 'numeric'
            });

            // Build detections mini-table
            const detectionsHtml = u.detections && u.detections.length
                ? `<table class="admin-table" style="font-size:.9rem;">
                    <thead><tr>
                        <th>File</th><th>Type</th><th>Result</th><th>Confidence</th><th>Date</th>
                    </tr></thead>
                    <tbody>
                    ${u.detections.map(d => `
                        <tr>
                            <td>${_esc(d.file_name)}</td>
                            <td>${d.file_type}</td>
                            <td><span class="result-badge ${d.result}">${d.result.toUpperCase()}</span></td>
                            <td>${d.confidence}%</td>
                            <td>${new Date(d.created_at).toLocaleDateString()}</td>
                        </tr>
                    `).join('')}
                    </tbody>
                   </table>
                   ${u.detection_count > u.detections.length
                     ? `<p style="color:#64748b;font-size:.85rem;margin-top:.5rem;">
                            Showing latest 10 of ${u.detection_count} total detections.
                        </p>`
                     : ''}`
                : '<p style="color:#64748b;">No detections yet.</p>';

            _showModal(`
                <div class="user-detail-modal">

                    <!-- Header -->
                    <div class="modal-user-header">
                        <div class="modal-avatar">${_esc(u.full_name).charAt(0).toUpperCase()}</div>
                        <div>
                            <h2>${_esc(u.full_name)}</h2>
                            <p class="modal-email">
                                <a href="mailto:${_esc(u.email)}">${_esc(u.email)}</a>
                            </p>
                            <span class="role-badge ${u.is_admin ? 'admin' : 'user'}">
                                ${u.is_admin ? '👑 Admin' : '👤 User'}
                            </span>
                        </div>
                    </div>

                    <!-- Info cards -->
                    <div class="modal-info-grid">
                        <div class="modal-info-card">
                            <p class="info-label">User ID</p>
                            <p class="info-value">#${u.id}</p>
                        </div>
                        <div class="modal-info-card">
                            <p class="info-label">Member Since</p>
                            <p class="info-value">${joinDate}</p>
                        </div>
                        <div class="modal-info-card">
                            <p class="info-label">Total Detections</p>
                            <p class="info-value">${u.detection_count || 0}</p>
                        </div>
                        <div class="modal-info-card">
                            <p class="info-label">Account Status</p>
                            <p class="info-value">${u.is_verified ? '✅ Verified' : '❌ Unverified'}</p>
                        </div>
                    </div>

                    <!-- Detection history -->
                    <div class="modal-section">
                        <h3>Detection History</h3>
                        ${detectionsHtml}
                    </div>

                    <!-- Actions -->
                    <div class="modal-actions">
                        ${!u.is_admin
                            ? `<button class="btn btn-danger"
                                       onclick="deleteUser(${u.id},'${_esc(u.email)}');closeModal();">
                                   Delete User
                               </button>`
                            : ''}
                        <button class="btn btn-secondary" onclick="closeModal()">Close</button>
                    </div>
                </div>
            `);

        } catch (err) {
            _showModal(`
                <div style="text-align:center;padding:2rem;">
                    <p class="error">Failed to load user details.</p>
                    <button class="btn btn-secondary" onclick="closeModal()">Close</button>
                </div>
            `);
        }
    }

    window.viewUserDetail = viewUserDetail;

    // ── Modal helpers ──────────────────────────────────────────────────────
    function _showModal(contentHtml) {
        let overlay = document.getElementById('admin-modal-overlay');
        if (!overlay) {
            overlay = document.createElement('div');
            overlay.id = 'admin-modal-overlay';
            overlay.className = 'modal-overlay';
            overlay.addEventListener('click', e => {
                if (e.target === overlay) closeModal();
            });
            document.body.appendChild(overlay);
        }
        overlay.innerHTML = `
            <div class="modal-box">
                <button class="modal-close" onclick="closeModal()">✕</button>
                ${contentHtml}
            </div>
        `;
        overlay.style.display = 'flex';
        document.body.style.overflow = 'hidden';
    }

    window.closeModal = function() {
        const overlay = document.getElementById('admin-modal-overlay');
        if (overlay) overlay.style.display = 'none';
        document.body.style.overflow = '';
    };

    async function deleteUser(id, email) {
        if (!confirm(`Delete user "${email}" and all their data? This cannot be undone.`)) return;
        try {
            await API.delete(`${API_CONFIG.ENDPOINTS.ADMIN_USER_DETAIL}/${id}`);
            loadUsers(usersPage);
            loadStats();
        } catch (err) {
            alert('Delete failed: ' + (err.message || 'Unknown error'));
        }
    }
    window.deleteUser = deleteUser;   // expose to inline onclick

    // ── Detections table ───────────────────────────────────────────────────
    async function loadDetections(page = 1) {
        try {
            const data = await API.request(
                `${API_CONFIG.ENDPOINTS.ADMIN_DETECTIONS}?page=${page}&limit=${perPage}`
            );
            const container = document.getElementById('detections-list');

            if (!data.detections.length) {
                container.innerHTML = '<p class="empty-state">No detections found.</p>';
                return;
            }

            container.innerHTML = `
                <table class="admin-table">
                    <thead><tr>
                        <th>ID</th><th>User</th><th>File</th><th>Type</th>
                        <th>Result</th><th>Confidence</th><th>Demo</th><th>Date</th>
                    </tr></thead>
                    <tbody>
                    ${data.detections.map(d => `
                        <tr>
                            <td>${d.id}</td>
                            <td>${_esc(d.full_name)}</td>
                            <td>${_esc(d.file_name)}</td>
                            <td>${d.file_type}</td>
                            <td><span class="result-badge ${d.result}">
                                    ${d.result.toUpperCase()}
                                </span></td>
                            <td>${d.confidence}%</td>
                            <td>${d.is_demo ? '⚠️ Demo' : '—'}</td>
                            <td>${new Date(d.created_at).toLocaleDateString()}</td>
                        </tr>
                    `).join('')}
                    </tbody>
                </table>
                ${_paginationHtml(data.page, data.pages, 'loadDetections')}
            `;
        } catch (err) {
            document.getElementById('detections-list').innerHTML =
                '<p class="error">Failed to load detections.</p>';
        }
    }

    // ── Tab switching ──────────────────────────────────────────────────────
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById(btn.dataset.tab + '-tab').classList.add('active');
        });
    });

    // ── Init ───────────────────────────────────────────────────────────────
    loadStats();
    loadUsers(1);
    loadDetections(1);

    // Expose for pagination buttons
    window.loadUsers      = loadUsers;
    window.loadDetections = loadDetections;
}

// ── Helpers ────────────────────────────────────────────────────────────────
function _esc(str) {
    return String(str)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;')
        .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function _paginationHtml(page, pages, fnName) {
    if (pages <= 1) return '';
    let html = '<div class="pagination" style="margin-top:1rem;text-align:center;">';
    if (page > 1)
        html += `<button class="btn btn-sm btn-secondary" onclick="${fnName}(${page - 1})">← Prev</button> `;
    html += `<span class="page-info"> ${page} / ${pages} </span>`;
    if (page < pages)
        html += ` <button class="btn btn-sm btn-secondary" onclick="${fnName}(${page + 1})">Next →</button>`;
    return html + '</div>';
}
