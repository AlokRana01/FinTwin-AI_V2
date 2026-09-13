-- Database Schema for AI Financial Health Digital Twin

-- 1. Users Table (Core Demographics & Income)
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    email TEXT UNIQUE,               -- login identifier for real, registered users
    password_hash TEXT,              -- PBKDF2-HMAC hash (see utils/auth.py); NULL for legacy/demo rows
    password_salt TEXT,
    name TEXT NOT NULL,
    age INTEGER NOT NULL CHECK (age >= 18 AND age <= 100),
    city TEXT,
    occupation TEXT,
    monthly_income REAL NOT NULL DEFAULT 0.0,
    bonus REAL DEFAULT 0.0,
    additional_income REAL DEFAULT 0.0,
    email_verified INTEGER DEFAULT 0,
    verification_token_hash TEXT,
    verification_expires_at TIMESTAMP,
    reset_token_hash TEXT,
    reset_expires_at TIMESTAMP,
    avatar_id TEXT DEFAULT 'avatar_01',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Digital Twin Table (Assets, Liabilities, Expenses, Goals)
CREATE TABLE IF NOT EXISTS digital_twins (
    user_id TEXT PRIMARY KEY,
    net_worth REAL DEFAULT 0.0,
    bank_savings REAL DEFAULT 0.0,
    fd_amount REAL DEFAULT 0.0,
    emergency_fund REAL DEFAULT 0.0,
    sip_amount REAL DEFAULT 0.0,
    mutual_funds REAL DEFAULT 0.0,
    stocks REAL DEFAULT 0.0,
    ppf_investment REAL DEFAULT 0.0,
    nps_investment REAL DEFAULT 0.0,
    loan_amount REAL DEFAULT 0.0,
    car_loan REAL DEFAULT 0.0,
    home_loan REAL DEFAULT 0.0,
    credit_card_debt REAL DEFAULT 0.0,
    monthly_emi REAL DEFAULT 0.0,
    health_insurance REAL DEFAULT 0.0,
    life_insurance REAL DEFAULT 0.0,
    
    -- Spending Behavior fields
    rent REAL DEFAULT 0.0,
    groceries REAL DEFAULT 0.0,
    utilities REAL DEFAULT 0.0,
    transport REAL DEFAULT 0.0,
    food_delivery REAL DEFAULT 0.0,
    entertainment REAL DEFAULT 0.0,
    shopping REAL DEFAULT 0.0,
    
    -- Goal Planning fields
    goal_type TEXT,
    goal_amount REAL DEFAULT 0.0,
    
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

-- 3. Transactions Table (Historical Ledger for Behavioral Analysis)
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    date DATE NOT NULL,
    category TEXT NOT NULL,
    amount REAL NOT NULL,
    type TEXT CHECK (type IN ('Income', 'Expense')),
    FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

-- 4. Goals Table (Goal Planning)
CREATE TABLE IF NOT EXISTS goals (
    goal_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    goal_name TEXT NOT NULL,
    target_amount REAL NOT NULL,
    current_amount REAL DEFAULT 0.0,
    target_date DATE NOT NULL,
    goal_type TEXT,
    FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

-- 5. Chat Usage Table (Daily rate-limit tracking for the FinBot widget)
CREATE TABLE IF NOT EXISTS chat_usage (
    user_id TEXT NOT NULL,
    usage_date DATE NOT NULL,
    message_count INTEGER DEFAULT 0,
    PRIMARY KEY (user_id, usage_date)
);
