import numpy as np
import pandas as pd

# Set seed for reproducible synthetic generation
np.random.seed(42)
NUM_ROWS = 100_000

print("Generating 100,000 synthetic passenger records...")

# ==========================================
# 1. PASSENGER INFORMATION
# ==========================================
passenger_id = [f"T2-{i:06d}" for i in range(1, NUM_ROWS + 1)]

first_names = ["James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda", 
               "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
               "Wei", "Aarav", "Sofia", "Mateo", "Yuki", "Fatima", "Lars", "Elena"]
last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", 
              "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
              "Chen", "Patel", "Kim", "Müller", "Silva", "Sato", "Ivanov", "Ali", "O'Connor"]

full_names = np.random.choice(first_names, NUM_ROWS) + " " + np.random.choice(last_names, NUM_ROWS)

# Age distribution
ages = np.random.gamma(shape=3.5, scale=10.0, size=NUM_ROWS)
ages = np.clip(ages, 1, 85).round(1)

gender = np.random.choice(["Male", "Female", "Non-Binary"], NUM_ROWS, p=[0.50, 0.48, 0.02])

nationalities = ["USA", "UK", "France", "Germany", "Canada", "China", "India", "Japan", "Brazil", "Australia"]
nationality = np.random.choice(nationalities, NUM_ROWS, p=[0.25, 0.20, 0.10, 0.10, 0.08, 0.08, 0.07, 0.05, 0.04, 0.03])

marital_status = np.where(ages < 18, "Single", 
                 np.random.choice(["Single", "Married", "Divorced", "Widowed"], NUM_ROWS, p=[0.35, 0.50, 0.10, 0.05]))

occupations = ["Business", "Engineering", "Healthcare", "Tourism", "Education", "Laborer", "Tech", "Military", "Student", "None"]
occupation = np.random.choice(occupations, NUM_ROWS)
occupation = np.where(ages < 18, "Student", occupation)

# ==========================================
# 2. TRAVEL INFORMATION
# ==========================================
pclass = np.random.choice([1, 2, 3], NUM_ROWS, p=[0.15, 0.25, 0.60])
ticket_number = np.random.randint(100000, 999999, NUM_ROWS).astype(str)

fare_base = {1: 1200, 2: 450, 3: 150}
fare_std = {1: 400, 2: 120, 3: 50}
ticket_price = np.array([
    np.random.normal(fare_base[c], fare_std[c]) for c in pclass
]).round(2)
ticket_price = np.clip(ticket_price, 25.0, 5000.0)

deck_map = {
    1: ["A", "B", "C", "D"],
    2: ["C", "D", "E"],
    3: ["E", "F", "G"]
}
cabin_deck = [np.random.choice(deck_map[c]) for c in pclass]

embarkation_ports = ["Southampton", "Cherbourg", "Queenstown", "New York"]
embarkation_port = np.random.choice(embarkation_ports, NUM_ROWS, p=[0.55, 0.25, 0.12, 0.08])

travel_purposes = ["Leisure", "Business", "Relocation", "Family Visit"]
travel_purpose = np.random.choice(travel_purposes, NUM_ROWS, p=[0.45, 0.25, 0.20, 0.10])

journey_duration_days = np.random.randint(low=3, high=14, size=NUM_ROWS)

# ==========================================
# 3. FAMILY & SOCIAL
# ==========================================
siblings_spouses = np.random.poisson(0.6, NUM_ROWS)
parents_children = np.random.poisson(0.5, NUM_ROWS)
family_size = siblings_spouses + parents_children + 1
is_alone = (family_size == 1).astype(int)

family_survival_rate = np.random.beta(a=0.8, b=1.5, size=NUM_ROWS).round(2)

# ==========================================
# 4. EMERGENCY & BEHAVIORAL
# ==========================================
boarding_position = np.random.randint(1, 1001, NUM_ROWS)

deck_score = {"A": 0.9, "B": 0.85, "C": 0.75, "D": 0.6, "E": 0.45, "F": 0.3, "G": 0.15}
deck_access = np.array([deck_score[d] for d in cabin_deck])
lifeboat_access = np.clip(deck_access + np.random.normal(0, 0.1, NUM_ROWS), 0.0, 1.0).round(2)

medical_conditions = ["None", "Mild Chronic", "Asthma", "Cardiovascular", "Mobility Impaired"]
medical_condition = np.random.choice(medical_conditions, NUM_ROWS, p=[0.75, 0.12, 0.06, 0.04, 0.03])

mobility_base = 100 - (ages * 0.5)
med_penalty = {"None": 0, "Mild Chronic": 10, "Asthma": 15, "Cardiovascular": 30, "Mobility Impaired": 50}
med_losses = np.array([med_penalty[m] for m in medical_condition])
mobility_score = np.clip(mobility_base - med_losses + np.random.normal(0, 5, NUM_ROWS), 5, 100).round(1)

emergency_response_time = np.random.gamma(shape=2.0, scale=12.0, size=NUM_ROWS).round(1)

evacuation_priority = np.where((ages < 12) | (gender == "Female"), "High",
                      np.where(pclass == 1, "Medium", "Low"))

# ==========================================
# 5. DERIVED FEATURES
# ==========================================
fare_per_person = (ticket_price / family_size).round(2)
wealth_index = ((ticket_price / 1000) * (4 - pclass) + np.random.normal(0, 0.2, NUM_ROWS)).round(2)
age_group = pd.cut(ages, bins=[0, 12, 18, 35, 60, 100], labels=["Child", "Teen", "Adult", "Middle_Aged", "Senior"])
family_risk_score = (family_size * 0.15 + (1 - lifeboat_access) * 0.8).round(2)

# ==========================================
# 6. RE-CALIBRATED TARGET GENERATION
# ==========================================
# Intercept shifted to -3.20 to target ~29-30% survival rate
logit = (
    -3.20
    + 1.10 * (gender == "Female").astype(int)
    + 0.80 * (ages < 12).astype(int)
    - 0.015 * np.maximum(0, ages - 50)
    + 0.85 * (pclass == 1).astype(int)
    + 0.35 * (pclass == 2).astype(int)
    + 1.55 * lifeboat_access
    + 0.012 * mobility_score
    - 0.02 * emergency_response_time
    + 0.60 * family_survival_rate
    - 0.30 * is_alone
    + np.random.normal(0, 0.80, NUM_ROWS)
)

survival_probability = (1 / (1 + np.exp(-logit))).round(4)
survived = (np.random.uniform(0, 1, NUM_ROWS) < survival_probability).astype(int)

# ==========================================
# 7. CONSTRUCT DATAFRAME & MISSING VALUES
# ==========================================
df = pd.DataFrame({
    'passenger_id': passenger_id,
    'full_name': full_names,
    'age': ages,
    'gender': gender,
    'nationality': nationality,
    'marital_status': marital_status,
    'occupation': occupation,
    'class': pclass,
    'ticket_number': ticket_number,
    'ticket_price': ticket_price,
    'cabin_deck': cabin_deck,
    'embarkation_port': embarkation_port,
    'travel_purpose': travel_purpose,
    'journey_duration_days': journey_duration_days,
    'siblings_spouses': siblings_spouses,
    'parents_children': parents_children,
    'family_size': family_size,
    'is_alone': is_alone,
    'family_survival_rate': family_survival_rate,
    'boarding_position': boarding_position,
    'lifeboat_access': lifeboat_access,
    'medical_condition': medical_condition,
    'mobility_score': mobility_score,
    'emergency_response_time': emergency_response_time,
    'evacuation_priority': evacuation_priority,
    'wealth_index': wealth_index,
    'age_group': age_group,
    'fare_per_person': fare_per_person,
    'family_risk_score': family_risk_score,
    'survival_probability': survival_probability,
    'survived': survived
})

# Missing value injection
missing_deck_idx = df.sample(frac=0.18, random_state=42).index
df.loc[missing_deck_idx, 'cabin_deck'] = np.nan

missing_resp_idx = df.sample(frac=0.05, random_state=42).index
df.loc[missing_resp_idx, 'emergency_response_time'] = np.nan

# Save to CSV
file_name = "titanic_2_0_passenger_survival_dataset.csv"
df.to_csv(file_name, index=False)

print(f"Dataset regenerated and saved to '{file_name}'.")
print(f"Overall Survival Rate: {df['survived'].mean():.2%}")
