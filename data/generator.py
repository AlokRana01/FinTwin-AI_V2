"""
Synthetic Indian Financial Data Generator
Generates realistic financial profiles, spending habits, tax configurations,
and transaction logs for 10,000 Indian salaried individuals aged 22-40.
Saves the generated dataset as a CSV file in data/indian_salaried_financial_data.csv.
"""

import os
import random
import logging
import numpy as np
import pandas as pd
from faker import Faker

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants for Generation
TIER_1_CITIES = ["Bengaluru", "Mumbai", "Delhi NCR", "Pune", "Hyderabad", "Chennai"]
TIER_2_CITIES = ["Ahmedabad", "Jaipur", "Kolkata", "Lucknow", "Chandigarh", "Coimbatore"]
TIER_3_CITIES = ["Mysuru", "Trichy", "Indore", "Patna", "Dehradun", "Nashik"]

OCCUPATIONS = [
    "Software Engineer",
    "Data Scientist",
    "Financial Analyst",
    "Marketing Executive",
    "HR Specialist",
    "Bank Manager",
    "Operations Analyst",
    "Consultant",
    "Doctor",
    "Educator"
]

GOAL_TYPES = [
    "Home Purchase",
    "Wedding",
    "Higher Education",
    "Child Education",
    "Retirement",
    "Travel",
    "Emergency Fund",
    "Car Purchase"
]

def generate_synthetic_data(num_records: int = 10000, seed: int = 42) -> pd.DataFrame:
    """
    Generates synthetic Indian financial data for salaried individuals.
    Ensures logical consistency and realistic distributions.
    """
    # Set seeds for reproducibility
    np.random.seed(seed)
    random.seed(seed)
    fake = Faker("en_IN")
    Faker.seed(seed)

    logger.info(f"Starting generation of {num_records} Indian salaried financial records...")

    data = []

    for i in range(num_records):
        # 1. Demographics
        age = int(np.random.randint(22, 41)) # Age 22 to 40
        
        # City selection based on weighted distributions (Tier 1: 65%, Tier 2: 25%, Tier 3: 10%)
        city_rand = random.random()
        if city_rand < 0.65:
            city = random.choice(TIER_1_CITIES)
            city_tier = "Tier 1"
        elif city_rand < 0.90:
            city = random.choice(TIER_2_CITIES)
            city_tier = "Tier 2"
        else:
            city = random.choice(TIER_3_CITIES)
            city_tier = "Tier 3"

        occupation = random.choice(OCCUPATIONS)

        # 2. Monthly Income (correlated with age, city tier, and occupation)
        # Base income log-normal scale
        if age <= 25:
            base_income = np.random.lognormal(mean=10.5, sigma=0.25)  # Median ~₹36,000
        elif age <= 30:
            base_income = np.random.lognormal(mean=11.2, sigma=0.3)   # Median ~₹73,000
        elif age <= 35:
            base_income = np.random.lognormal(mean=11.8, sigma=0.35)  # Median ~₹133,000
        else:
            base_income = np.random.lognormal(mean=12.2, sigma=0.4)   # Median ~₹200,000

        # Adjust for city tier
        tier_multipliers = {"Tier 1": 1.25, "Tier 2": 0.95, "Tier 3": 0.75}
        base_income *= tier_multipliers[city_tier]

        # Adjust for high/low paying roles
        role_multiplier = 1.0
        if occupation in ["Software Engineer", "Data Scientist", "Consultant", "Doctor"]:
            role_multiplier = 1.20
        elif occupation in ["Educator", "Marketing Executive"]:
            role_multiplier = 0.85
        base_income *= role_multiplier

        # Clip minimum and maximum monthly incomes for salaried profiles
        monthly_income = float(np.clip(base_income, 22000, 550000))
        # Round to nearest thousand for realistic representation
        monthly_income = round(monthly_income, -3)

        # Bonus (Annual bonus, scaled with salary, between 8% and 25% of annual income)
        bonus_pct = random.uniform(0.08, 0.25)
        bonus = float(round((monthly_income * 12) * bonus_pct, -3))

        # Additional Income (Dividends, interest, side hustles - non-zero for ~18% of users)
        additional_income = 0.0
        if random.random() < 0.18:
            additional_income = float(round(np.random.uniform(3000, 35000), -3))

        # 3. Liabilities Setup (determined early to compute EMI, which impacts rent/savings)
        home_loan = 0.0
        car_loan = 0.0
        credit_card_debt = 0.0

        # Home loans (primarily for ages 28-40, ~22% ownership with loan)
        if age >= 28 and random.random() < 0.22:
            # Home loan sizes scale with income
            home_loan = float(round(np.random.uniform(1500000, min(monthly_income * 60, 8000000)), -4))

        # Car loans (~28% of users)
        if random.random() < 0.28:
            car_loan = float(round(np.random.uniform(300000, min(monthly_income * 10, 1500000)), -4))

        # Credit card debt (~42% of users)
        if random.random() < 0.42:
            credit_card_debt = float(round(np.random.exponential(scale=25000), -2))
            credit_card_debt = float(np.clip(credit_card_debt, 2000, min(monthly_income * 1.5, 150000)))

        # Compute Monthly EMIs
        # Home Loan EMI: 8.5% interest, 20-year term -> multiplier ~0.008678 per Rupee
        home_loan_emi = home_loan * 0.008678 if home_loan > 0 else 0.0
        # Car Loan EMI: 9.5% interest, 5-year term -> multiplier ~0.021002 per Rupee
        car_loan_emi = car_loan * 0.021002 if car_loan > 0 else 0.0
        
        monthly_emi = float(round(home_loan_emi + car_loan_emi, -2))

        # Ensure EMI is not overly crippling (>60% of income); scale down if so
        if monthly_emi > monthly_income * 0.60:
            scale_factor = (monthly_income * 0.50) / monthly_emi
            home_loan = float(round(home_loan * scale_factor, -4))
            car_loan = float(round(car_loan * scale_factor, -4))
            home_loan_emi = home_loan * 0.008678 if home_loan > 0 else 0.0
            car_loan_emi = car_loan * 0.021002 if car_loan > 0 else 0.0
            monthly_emi = float(round(home_loan_emi + car_loan_emi, -2))

        loan_amount = float(home_loan + car_loan + credit_card_debt)

        # 4. Monthly Expenses
        # Rent (Rent is 0 if they have an active home loan, else scales by city tier and income)
        if home_loan > 0:
            rent = 0.0
        else:
            rent_pct = random.uniform(0.15, 0.32)
            rent_est = monthly_income * rent_pct
            if city_tier == "Tier 1":
                rent = float(np.clip(rent_est, 10000, 45000))
            elif city_tier == "Tier 2":
                rent = float(np.clip(rent_est, 6000, 22000))
            else:
                rent = float(np.clip(rent_est, 3000, 11000))
            rent = float(round(rent, -2))

        # Grocery expenses
        groceries = float(round(np.clip(3500 + monthly_income * 0.04 + np.random.normal(0, 1000), 2000, 15000), -2))
        # Utilities
        utilities = float(round(np.clip(1800 + monthly_income * 0.02 + np.random.normal(0, 500), 1200, 9000), -2))
        # Transport
        transport = float(round(np.clip(1200 + monthly_income * 0.03 + np.random.normal(0, 800), 1000, 12000), -2))
        # Food Delivery
        food_delivery_pct = random.uniform(0.04, 0.12) if age < 30 else random.uniform(0.02, 0.07)
        food_delivery = float(round(np.clip(monthly_income * food_delivery_pct + np.random.normal(0, 800), 1000, 15000), -2))
        # Entertainment
        entertainment = float(round(np.clip(monthly_income * random.uniform(0.03, 0.10) + np.random.normal(0, 1000), 1000, 18000), -2))
        # Shopping
        shopping = float(round(np.clip(monthly_income * random.uniform(0.04, 0.12) + np.random.normal(0, 1500), 1500, 25000), -2))

        # Total monthly basic expenses (excluding investments and EMIs)
        monthly_basic_expenses = rent + groceries + utilities + transport + food_delivery + entertainment + shopping
        total_monthly_outflow = monthly_basic_expenses + monthly_emi

        # 5. Savings & Investments (SIP, Mutual Funds, Stocks, PPF, NPS)
        # Determine available budget for savings/investments
        remaining_budget = (monthly_income + additional_income) - total_monthly_outflow
        
        # Savings profile setting
        if remaining_budget <= 0:
            sip_amount = 0.0
            savings_pct = 0.02
        else:
            # SIP is a fraction of remaining budget
            savings_pct = random.uniform(0.20, 0.60)
            sip_amount = float(round(remaining_budget * savings_pct, -3))
            sip_amount = float(np.clip(sip_amount, 0, min(monthly_income * 0.40, 50000)))

        # Accumulation calculations based on age (active years = age - 21)
        active_years = max(1, age - 21)
        
        # Compound factor assuming average 12% returns with variance
        compound_factor = sum([(1 + random.uniform(0.08, 0.16)) ** yr for yr in range(active_years)])

        mutual_fund_value = float(round(sip_amount * 12 * compound_factor, -3)) if sip_amount > 0 else 0.0

        # Stock market investments (~36% of users)
        stock_value = 0.0
        if random.random() < 0.36 and mutual_fund_value > 0:
            stock_value = float(round(mutual_fund_value * random.uniform(0.15, 1.2), -3))

        # PPF (Public Provident Fund) (~52% participation, tax-free savings)
        ppf_investment = 0.0
        if random.random() < 0.52:
            annual_ppf_est = np.clip(monthly_income * 0.05, 10000, 150000)
            ppf_investment = float(round(annual_ppf_est * active_years * 1.07, -3))

        # NPS (National Pension Scheme) (~26% participation)
        nps_investment = 0.0
        if random.random() < 0.26:
            annual_nps_est = np.clip(monthly_income * 0.03, 10000, 50000)
            nps_investment = float(round(annual_nps_est * active_years * 1.09, -3))

        # 6. Cash Reserves
        # Bank Savings (cash in hand, e.g. 0.5x to 2x monthly income)
        bank_savings = float(round(monthly_income * random.uniform(0.4, 2.2), -3))

        # FD Amount (Fixed Deposits, more popular with older/conservative users)
        fd_amount = 0.0
        if random.random() < 0.45:
            fd_multiplier = random.uniform(0.5, 8.0) if age >= 30 else random.uniform(0.2, 3.0)
            fd_amount = float(round(monthly_income * fd_multiplier, -3))

        # Emergency Fund Value (typically 3 to 6 months of total expenses)
        # Some users might have insufficient emergency funds (random multiplier between 0 and 7)
        emergency_fund_months = random.choice([0, 1, 2, 3, 6, 6, 6, 8])  # weighted towards 6 months
        emergency_fund = float(round(total_monthly_outflow * emergency_fund_months, -3))

        # 7. Insurance Covers (typical corporate/individual values in India)
        # Health Insurance
        health_insurance_choices = [0.0, 300000.0, 500000.0, 700000.0, 1000000.0, 1500000.0]
        health_insurance = random.choices(health_insurance_choices, weights=[10, 25, 35, 15, 10, 5])[0]

        # Life Insurance
        life_insurance_choices = [0.0, 5000000.0, 7500000.0, 10000000.0, 15000000.0, 20000000.0]
        life_insurance = random.choices(life_insurance_choices, weights=[30, 20, 20, 15, 10, 5])[0]

        # 8. Goals
        goal_type = random.choice(GOAL_TYPES)
        # Goal amount ranges between 1.5L to 40L, correlated with income
        goal_amount = float(round(np.random.uniform(150000, min(monthly_income * 24, 4000000)), -3))

        # Store record
        data.append({
            "User_ID": f"IND_USR_{i+1:05d}",
            "Name": fake.name(),
            "Age": age,
            "City": city,
            "Occupation": occupation,
            "Monthly_Income": monthly_income,
            "Bonus": bonus,
            "Additional_Income": additional_income,
            "Rent": rent,
            "Groceries": groceries,
            "Utilities": utilities,
            "Transport": transport,
            "Food_Delivery": food_delivery,
            "Entertainment": entertainment,
            "Shopping": shopping,
            "Bank_Savings": bank_savings,
            "FD_Amount": fd_amount,
            "Emergency_Fund": emergency_fund,
            "SIP_Amount": sip_amount,
            "Mutual_Fund_Value": mutual_fund_value,
            "Stock_Value": stock_value,
            "PPF_Investment": ppf_investment,
            "NPS_Investment": nps_investment,
            "Loan_Amount": loan_amount,
            "Car_Loan": car_loan,
            "Home_Loan": home_loan,
            "Credit_Card_Debt": credit_card_debt,
            "Monthly_EMI": monthly_emi,
            "Health_Insurance": health_insurance,
            "Life_Insurance": life_insurance,
            "Goal_Type": goal_type,
            "Goal_Amount": goal_amount
        })

    logger.info("Synthetic data generation finished.")
    return pd.DataFrame(data)

def validate_financial_data(df: pd.DataFrame) -> bool:
    """
    Performs data validation on the generated DataFrame.
    Returns True if validation checks pass, raises ValueError otherwise.
    """
    logger.info("Initializing financial data validation rules...")
    
    # 1. Row & column counts check
    expected_cols = [
        "User_ID", "Name", "Age", "City", "Occupation", "Monthly_Income", "Bonus", "Additional_Income",
        "Rent", "Groceries", "Utilities", "Transport", "Food_Delivery", "Entertainment", "Shopping",
        "Bank_Savings", "FD_Amount", "Emergency_Fund", "SIP_Amount", "Mutual_Fund_Value", "Stock_Value",
        "PPF_Investment", "NPS_Investment", "Loan_Amount", "Car_Loan", "Home_Loan", "Credit_Card_Debt",
        "Monthly_EMI", "Health_Insurance", "Life_Insurance", "Goal_Type", "Goal_Amount"
    ]
    
    if len(df) != 10000:
        raise ValueError(f"Validation Error: Expected exactly 10,000 records, got {len(df)}")
        
    for col in expected_cols:
        if col not in df.columns:
            raise ValueError(f"Validation Error: Expected column '{col}' is missing in the dataset.")

    # 2. Null values check
    null_counts = df.isnull().sum().sum()
    if null_counts > 0:
        raise ValueError(f"Validation Error: Found {null_counts} null values in the dataset.")

    # 3. Demographics boundary checks
    if not df["Age"].between(22, 40).all():
        raise ValueError("Validation Error: Age values are out of the 22-40 age limits.")
        
    if (df["Monthly_Income"] < 0).any():
        raise ValueError("Validation Error: Monthly_Income contains negative values.")

    # 4. Logical constraint checks
    # Rent + EMI vs Income Check
    housing_ratio = (df["Rent"] + df["Monthly_EMI"]) / (df["Monthly_Income"] + df["Additional_Income"])
    if (housing_ratio > 0.90).any():
        logger.warning(f"Validation Warning: {(housing_ratio > 0.90).sum()} users have Rent + EMI exceeding 90% of monthly income.")

    # Home Loan vs Rent check
    homeowners_paying_rent = df[(df["Home_Loan"] > 0) & (df["Rent"] > 0)]
    if len(homeowners_paying_rent) > 0:
        raise ValueError(f"Validation Error: Found {len(homeowners_paying_rent)} records where a user has both a Home Loan and Rent > 0.")

    # Total Outstanding Loans calculation check
    calculated_loan = df["Home_Loan"] + df["Car_Loan"] + df["Credit_Card_Debt"]
    if not np.isclose(df["Loan_Amount"], calculated_loan).all():
        raise ValueError("Validation Error: 'Loan_Amount' is not equal to Home_Loan + Car_Loan + Credit_Card_Debt.")

    # Non-negative numerical check (except User_ID, Name, City, Occupation, Goal_Type)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if (df[col] < 0).any():
            raise ValueError(f"Validation Error: Column '{col}' contains negative values.")

    logger.info("All data validation rules passed successfully!")
    return True

def run_pipeline():
    """
    Runs the data generation, validation, and saving pipelines.
    """
    df = generate_synthetic_data(num_records=10000)
    
    # Run validation
    try:
        validate_financial_data(df)
    except ValueError as e:
        logger.error(f"Data validation failed! Details: {e}")
        return

    # Create destination directories if they don't exist
    output_dir = os.path.join(os.path.dirname(__file__), "raw")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory at {output_dir}")

    output_path = os.path.join(output_dir, "indian_salaried_financial_data.csv")
    df.to_csv(output_path, index=False)
    logger.info(f"Dataset successfully saved to {output_path}")

    # Log summary statistics for reassurance
    logger.info("--- Data Summary Statistics ---")
    logger.info(f"Mean Income: ₹{df['Monthly_Income'].mean():,.2f}")
    logger.info(f"Max Income: ₹{df['Monthly_Income'].max():,.2f}")
    logger.info(f"Mean Net Worth Proxy (Savings + MF + Stock + FD): ₹{(df['Bank_Savings'] + df['Mutual_Fund_Value'] + df['Stock_Value'] + df['FD_Amount']).mean():,.2f}")
    logger.info(f"Total Outstanding Home Loan: ₹{df['Home_Loan'].sum():,.2f} across {df['Home_Loan'].gt(0).sum()} users")

if __name__ == "__main__":
    run_pipeline()
