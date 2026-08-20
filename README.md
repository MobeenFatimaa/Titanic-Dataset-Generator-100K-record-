#  Titanic : 100K Passenger Dataset Generator

A high-performance Python generation framework designed to synthesize **100,000 passenger records across 31 features** for large-scale tabular machine learning.

Inspired by the classic historical Titanic problem, **Titanic** provides a modern, scale-ready benchmark dataset for binary classification, missing-value imputation, non-linear risk modeling, and explainable AI (XAI).

---

##  Repository Structure

```text
.
├── generate.py          # Core script generating 100k synthetic passenger records
├── validate.py          # Data validation suite verifying schema & logic constraints
├── requirements.txt     # Minimal dependencies (numpy, pandas)
├── LICENSE              # Apache-2.0 License
└── README.md            # Repository documentation
```

##  Dataset Features & Specifications

* **Total Records:** 100,000 passenger rows
* **Feature Count:** 31 total (30 predictor variables + 1 target)
* **Target Variable:** `survived` (`1` = Survived, `0` = Deceased)
* **Target Balance:** Calibrated to **~28.5% survival rate**
* **Realism & Noise:** Non-linear logistic risk modeling combined with Gaussian noise ($\sigma = 0.80$) to prevent trivial 100% classification accuracy.

### Feature Summary

* **Demographics:** `passenger_id`, `full_name`, `age`, `gender`, `nationality`, `marital_status`, `occupation`
* **Travel Details:** `class`, `ticket_number`, `ticket_price`, `cabin_deck`, `embarkation_port`, `travel_purpose`, `journey_duration_days`
* **Family Structure:** `siblings_spouses`, `parents_children`, `family_size`, `is_alone`, `family_survival_rate`
* **Emergency & Health:** `boarding_position`, `lifeboat_access`, `medical_condition`, `mobility_score`, `emergency_response_time`, `evacuation_priority`
* **Derived Metrics:** `wealth_index`, `age_group`, `fare_per_person`, `family_risk_score`, `survival_probability`

##  Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/MobeenFatimaa/Titanic-Dataset-Generator-100K-record-.git
cd Titanic-Dataset-Generator-100K-record-
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate the Dataset

Run the generator script to construct `titanic_2_0_passenger_survival_dataset.csv` in seconds:

```bash
python generate.py
```

### 4. Validate Dataset Integrity

Run the validation script to verify schema consistency, missingness, and target balance:

```bash
python validate.py
```

##  Ideal Use Cases

1. **Binary Classification Benchmark:** Train models like XGBoost, LightGBM, CatBoost, and Tabular Neural Networks at scale.
2. **Missing Value Imputation:** Practice handling realistic missing data in `cabin_deck` (~18%) and `emergency_response_time` (~5%).
3. **Feature Engineering:** Extract complex interactions between age, deck access, family size, and economic status.
4. **Explainable AI (SHAP / LIME):** Interpret non-linear feature interactions driving survival probabilities.

##  License

This project is licensed under the **Apache-2.0 License** — see the [LICENSE](LICENSE) file for details.
