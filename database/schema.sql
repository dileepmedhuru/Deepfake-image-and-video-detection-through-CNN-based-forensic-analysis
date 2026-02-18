-- ─────────────────────────────────────────────────────────────────
-- Deepfake Detection System – Database Schema
-- ─────────────────────────────────────────────────────────────────

-- Users
CREATE TABLE IF NOT EXISTS users (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name             TEXT    NOT NULL,
    email                 TEXT    UNIQUE NOT NULL,
    password              TEXT    NOT NULL,           -- bcrypt hash (NEVER plain-text)
    is_admin              BOOLEAN DEFAULT 0,
    is_verified           BOOLEAN DEFAULT 1,
    force_password_change BOOLEAN DEFAULT 0,          -- flag admin to change pw on first login
    created_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Detection history
CREATE TABLE IF NOT EXISTS detection_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER NOT NULL,
    file_name       TEXT    NOT NULL,
    file_type       TEXT    NOT NULL,     -- 'image' | 'video'
    file_path       TEXT    NOT NULL,
    result          TEXT    NOT NULL,     -- 'real'  | 'fake'
    confidence      REAL    NOT NULL,
    processing_time REAL    NOT NULL,
    is_demo         BOOLEAN DEFAULT 0,   -- true when no ML model loaded
    metadata        TEXT,                -- JSON string with additional info
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_users_email
    ON users(email);

CREATE INDEX IF NOT EXISTS idx_detection_user_id
    ON detection_history(user_id);

CREATE INDEX IF NOT EXISTS idx_detection_created_at
    ON detection_history(created_at);

CREATE INDEX IF NOT EXISTS idx_detection_result
    ON detection_history(result);

-- NOTE: Default admin is seeded by app.py on first run (not here),
--       so the password is always hashed and printed once to the console.
