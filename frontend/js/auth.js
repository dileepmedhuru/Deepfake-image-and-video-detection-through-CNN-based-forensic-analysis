// ── Shared utilities ───────────────────────────────────────────────────────
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

// ── Client-side password strength check (mirrors backend rules) ───────────
function validatePassword(password) {
    if (password.length < 8)
        return 'Password must be at least 8 characters.';
    if (!/[A-Z]/.test(password))
        return 'Password must contain at least one uppercase letter.';
    if (!/[a-z]/.test(password))
        return 'Password must contain at least one lowercase letter.';
    if (!/[0-9]/.test(password))
        return 'Password must contain at least one number.';
    if (!/[!@#$%^&*(),.?":{}|<>]/.test(password))
        return 'Password must contain at least one special character.';
    return null;
}

// ── Signup ─────────────────────────────────────────────────────────────────
const signupForm = document.getElementById('signup-form');
if (signupForm) {
    signupForm.addEventListener('submit', async e => {
        e.preventDefault();
        hideMessages();

        const fullName        = document.getElementById('full_name').value.trim();
        const email           = document.getElementById('email').value.trim().toLowerCase();
        const password        = document.getElementById('password').value;
        const confirmPassword = document.getElementById('confirm_password').value;

        if (!fullName || !email || !password) {
            return showError('All fields are required.');
        }
        if (password !== confirmPassword) {
            return showError('Passwords do not match.');
        }
        const pwError = validatePassword(password);
        if (pwError) return showError(pwError);

        setLoading(true);
        try {
            const res = await API.request(API_CONFIG.ENDPOINTS.SIGNUP, {
                method: 'POST',
                body: JSON.stringify({ full_name: fullName, email, password }),
                skipAuth: true,
            });
            Storage.setToken(res.token);
            Storage.setUser(res.user);
            showSuccess('Registration successful! Redirecting…');
            setTimeout(() => {
                res.user.is_admin ? Redirect.toAdminDashboard() : Redirect.toDashboard();
            }, 1000);
        } catch (err) {
            showError(err.message || 'Signup failed. Please try again.');
        } finally {
            setLoading(false);
        }
    });
}

// ── Login ──────────────────────────────────────────────────────────────────
const loginForm = document.getElementById('login-form');
if (loginForm) {
    loginForm.addEventListener('submit', async e => {
        e.preventDefault();
        hideMessages();

        const email    = document.getElementById('email').value.trim().toLowerCase();
        const password = document.getElementById('password').value;

        if (!email || !password) return showError('Email and password are required.');

        setLoading(true);
        try {
            const res = await API.request(API_CONFIG.ENDPOINTS.LOGIN, {
                method: 'POST',
                body: JSON.stringify({ email, password }),
                skipAuth: true,
            });
            Storage.setToken(res.token);
            Storage.setUser(res.user);
            showSuccess('Login successful! Redirecting…');
            setTimeout(() => {
                // If admin needs to change their default password, redirect to profile
                if (res.user.force_password_change) {
                    window.location.href = '/profile.html';
                } else if (res.user.is_admin) {
                    Redirect.toAdminDashboard();
                } else {
                    Redirect.toDashboard();
                }
            }, 1000);
        } catch (err) {
            showError(err.message || 'Login failed. Please try again.');
        } finally {
            setLoading(false);
        }
    });
}
