import pandas as pd
import numpy as np
from taipy.gui import Gui, notify, State, Markdown

# --- 1. Data Preparation ---
def generate_churn_data():
    """Generates a synthetic DataFrame simulating Telco Customer Churn."""
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        'CustomerID': [f'C{i:04d}' for i in range(n)],
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n, p=[0.55, 0.25, 0.20]),
        'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n),
        'Region': np.random.choice(['North', 'South', 'East', 'West'], n),
        'MonthlyCharges': np.random.uniform(20, 120, n).round(2),
        'Tenure': np.random.randint(1, 72, n),
    })
    # Logic to create realistic churn probability
    df['Churn_Prob'] = 0.1
    df.loc[df['Contract'] == 'Month-to-month', 'Churn_Prob'] += 0.3
    df.loc[df['PaymentMethod'] == 'Electronic check', 'Churn_Prob'] += 0.1
    df.loc[df['MonthlyCharges'] > 90, 'Churn_Prob'] += 0.15
    df.loc[df['Tenure'] < 12, 'Churn_Prob'] += 0.15
    df['Churn_Prob'] = np.clip(df['Churn_Prob'], 0.05, 0.85)
    df['Churn'] = (np.random.rand(n) < df['Churn_Prob']).astype(int)
    df['ChurnLabel'] = df['Churn'].map({1: 'Yes', 0: 'No'})
    return df

full_data = generate_churn_data()
all_contracts = full_data['Contract'].unique().tolist()

# --- 2. State Initialization ---
selected_contract = all_contracts[0]
filtered_data = full_data[full_data['Contract'] == selected_contract]

# KPI Calculations
def calculate_kpis(data):
    avg_tenure = data['Tenure'].mean()
    total_revenue = (data['MonthlyCharges'] * data['Tenure']).sum()
    churn_rate = data['Churn'].mean() * 100
    return avg_tenure, total_revenue, churn_rate

avg_tenure, total_revenue, churn_rate = calculate_kpis(filtered_data)

# Initial Color Logic
def get_color(rate):
    if rate > 50: return "red"
    if rate > 30: return "orange"
    return "green"

churn_color = get_color(churn_rate)

# ML Inputs
ml_monthly_charges = 75.0
ml_tenure = 24
ml_contract = 'One year'
ml_churn_prediction = "Ready to predict"
llm_recommendation = "No prediction generated yet."

# --- 3. Core Logic Functions ---

def filter_data_fixed(state: State):
    """Updates EDA charts when contract filter changes."""
    state.filtered_data = full_data[full_data['Contract'] == state.selected_contract]
    
    # Recalculate KPIs
    at, tr, cr = calculate_kpis(state.filtered_data)
    state.avg_tenure = at
    state.total_revenue = tr
    state.churn_rate = cr
    
    state.churn_color = get_color(cr)
    notify(state, 'info', f"Updated view for {state.selected_contract}")

def predict_churn(state: State):
    """Simulates ML prediction logic."""
    risk_score = 0
    if state.ml_contract == 'Month-to-month': risk_score += 0.3
    elif state.ml_contract == 'One year': risk_score += 0.15
    
    if state.ml_monthly_charges > 100: risk_score += 0.2
    elif state.ml_monthly_charges > 70: risk_score += 0.1
        
    if state.ml_tenure < 18: risk_score += 0.2

    if risk_score > 0.6:
        prediction = "HIGH RISK"
        color = "error" 
    elif risk_score > 0.3:
        prediction = "MEDIUM RISK"
        color = "warning"
    else:
        prediction = "LOW RISK"
        color = "success"
    
    state.ml_churn_prediction = prediction
    notify(state, color, f"Result: {prediction}")
    generate_recommendation(state) 

def generate_recommendation(state: State):
    """Simulates LLM generation based on risk level."""
    risk_level = state.ml_churn_prediction.split(' ')[0]
    
    if risk_level == "HIGH":
        rec = "🔴 **Urgent:** Offer 20% discount for 6mo & assign agent."
    elif risk_level == "MEDIUM":
        rec = "🟠 **Watch:** Send loyalty rewards email. Monitor usage."
    else:
        rec = "🟢 **Grow:** Upsell premium features. Customer is stable."
    
    state.llm_recommendation = rec

# --- 4. Enhanced GUI Definition ---

page = Markdown("""
<|toggle|theme|>

<|part|class_name=header|
# 🔮 **Customer Churn** Prediction & Analytics

This interactive dashboard combines **Historical Data Analysis** with **Machine Learning** to identify risk.
* **👈 Left Sidebar:** Use the controls to simulate a customer profile and predict their churn risk using our ML model.
* **📊 Main Dashboard:** Analyze historical trends. Use the dropdown below to filter data by Contract Type.
|>
<br/>

<|layout|columns=320px 1fr|gap=30px|

<|part|class_name=sidebar card|
### 🧠 Model **Inputs**
<br/>

**Tenure (Months)**
<|{ml_tenure}|slider|min=1|max=72|>

**Monthly Charges ($)**
<|{ml_monthly_charges}|slider|min=20|max=120|>

**Contract Type**
<|{ml_contract}|selector|lov={all_contracts}|dropdown=True|width=100%|>
<br/>
<br/>

<|Predict Risk|button|on_action=predict_churn|class_name=full-width primary|>

<br/>
<br/>
### **AI Analysis Results**
<br/>

**Prediction:**
<|{ml_churn_prediction}|text|class_name=h4 text-center|>

**Strategy:**
<|{llm_recommendation}|text|class_name=text-small|>
|>

<|part|
<|part|class_name=card p-3|
### 📊 **Executive Dashboard**
<|{selected_contract}|selector|lov={all_contracts}|dropdown=True|on_change=filter_data_fixed|label=Filter by Contract|>
|>

<|layout|columns=1 1 1|gap=20px|
<|part|class_name=card|
#### Churn Rate
<|{churn_rate}|indicator|value={churn_rate}|min=0|max=100|format=%.1f%%|color={churn_color}|width=100%|>
|>
<|part|class_name=card|
#### Avg Tenure
<|{avg_tenure}|text|format=%.1f Months|class_name=h2 text-center|>
|>
<|part|class_name=card|
#### Total Revenue
<|{total_revenue}|text|format=$%.2f|class_name=h2 text-center|>
|>
|>

<|part|class_name=card p-4|
### 📉 Monthly Charges Distribution
<|{filtered_data}|chart|type=bar|x=MonthlyCharges|y=Churn|color=ChurnLabel|title=Does higher cost lead to churn?|height=400px|>
|>

<|part|class_name=card p-4|
### 📍 Churn by Region
<|{filtered_data.groupby('Region')['Churn'].sum().reset_index()}|chart|type=bar|x=Region|y=Churn|title=High Risk Geographic Areas|height=400px|>
|>

<|layout|columns=1 1|gap=20px|
<|part|class_name=card p-4|
### 💳 Payment Methods
<|{filtered_data}|chart|type=pie|values=Churn|labels=PaymentMethod|title=Churn by Payment Method|height=400px|>
|>
<|part|class_name=card p-4|
### ⏱️ Tenure vs Churn
<|{full_data.groupby(['Tenure', 'Contract'])['Churn'].sum().reset_index()}|chart|type=scatter|x=Tenure|y=Churn|color=Contract|height=400px|>
|>
|>

|>
|>
""")

# --- 5. Run the Application ---
if __name__ == "__main__":
    style = {
        ".card": {
            "background-color": "white",
            "border-radius": "12px",
            "box-shadow": "0 4px 6px rgba(0,0,0,0.05)",
            "padding": "20px",
            "margin-bottom": "15px"
        },
        ".sidebar": {
             "background-color": "#1e293b", # Slate 800
             "color": "white",
        },
        ".header": {
            "margin-bottom": "20px",
        },
        ".primary": {
            "width": "100%", 
            "font-weight": "bold", 
            "background-color": "#ff5f5f", # Distinct accent color
            "color": "white"
        },
        "body": {
            "background-color": "#f4f6f8" # Light gray background for separation
        }
    }
    
    Gui(page).run(title="Churn Analytics Pro", dark_mode=False, host="0.0.0.0", port=5000, use_reloader=True, style=style)