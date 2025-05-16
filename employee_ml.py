import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Tuple, Optional
import random
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from datetime import datetime, timedelta
import scipy.optimize as optimize

# Set page configuration
st.set_page_config(
    page_title="Employee Cost Optimization Tool",
    page_icon="💼",
    layout="wide"
)

# Define constants
DESIGNATIONS = ["Analyst", "Associate", "AVP", "VP", "ED", "MD"]
LOCATIONS = ["India", "US", "UK"]

# Function to generate synthetic data with historical trend
@st.cache_data
def generate_employee_cost_data(add_historical=True, years_of_history=3):
    # Base salaries for each designation (increasing order)
    base_salaries = {
        "Analyst": 50000,
        "Associate": 70000,
        "AVP": 100000,
        "VP": 150000,
        "ED": 220000,
        "MD": 300000
    }
    
    # Location multipliers (India lowest, UK highest)
    location_multipliers = {
        "India": 1.0,
        "US": 1.5,
        "UK": 1.8
    }
    
    # Regional inflation rates (yearly)
    inflation_rates = {
        "India": 0.05,  # 5% annual inflation
        "US": 0.03,     # 3% annual inflation
        "UK": 0.035     # 3.5% annual inflation
    }
    
    # Employee retention probability based on relocation
    retention_probabilities = {
        "Same_Location_Same_Level": 0.95,
        "Same_Location_Promotion": 0.98,
        "Same_Location_Demotion": 0.70,
        "Different_Location_Same_Level": 0.75,
        "Different_Location_Promotion": 0.85,
        "Different_Location_Demotion": 0.50
    }
    
    # Create dataframe with current costs
    current_data = []
    for designation, location in product(DESIGNATIONS, LOCATIONS):
        annualized_rate = base_salaries[designation] * location_multipliers[location]
        # Add some random variation (±5%)
        annualized_rate = annualized_rate * random.uniform(0.95, 1.05)
        
        # Add additional attributes for ML features
        years_in_role = random.randint(1, 5)
        performance_score = random.uniform(2.0, 5.0)  # Scale of 1-5
        skill_match_score = random.uniform(0.7, 1.0)  # Scale of 0-1
        
        current_data.append({
            "Designation": designation,
            "Location": location,
            "Annualized_Rate": round(annualized_rate, 2),
            "Years_In_Role": years_in_role,
            "Performance_Score": round(performance_score, 2),
            "Skill_Match_Score": round(skill_match_score, 2)
        })
    
    df = pd.DataFrame(current_data)
    
    # If historical data is requested, add time series data
    if add_historical:
        # Generate historical data
        historical_data = []
        current_year = datetime.now().year
        
        for year in range(current_year - years_of_history, current_year + 1):
            for designation, location in product(DESIGNATIONS, LOCATIONS):
                # Calculate base rate for this year
                years_ago = current_year - year
                base_rate = base_salaries[designation] * location_multipliers[location]
                
                # Apply compound inflation for past years
                deflation_factor = (1 - inflation_rates[location]) ** years_ago
                historical_rate = base_rate * deflation_factor
                
                # Add random noise
                historical_rate = historical_rate * random.uniform(0.97, 1.03)
                
                historical_data.append({
                    "Year": year,
                    "Designation": designation,
                    "Location": location,
                    "Annualized_Rate": round(historical_rate, 2)
                })
        
        historical_df = pd.DataFrame(historical_data)
        return df, historical_df
    
    return df

# Create the employee cost data
def get_employee_data():
    current_df, historical_df = generate_employee_cost_data(add_historical=True)
    return current_df, historical_df

# Function to train cost prediction model
@st.cache_resource
def train_cost_prediction_model(historical_data):
    # Prepare the data
    X = historical_data[['Year', 'Designation', 'Location']]
    y = historical_data['Annualized_Rate']
    
    # Create preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', ['Year']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['Designation', 'Location'])
        ])
    
    # Create and train the model
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    model.fit(X, y)
    
    return model

# Function to predict future costs
def predict_future_costs(model, future_year, current_data):
    # Create future data points
    future_data = []
    for designation, location in product(DESIGNATIONS, LOCATIONS):
        future_data.append({
            'Year': future_year,
            'Designation': designation,
            'Location': location
        })
    
    future_df = pd.DataFrame(future_data)
    
    # Make predictions
    predictions = model.predict(future_df[['Year', 'Designation', 'Location']])
    
    # Add predictions to the dataframe
    future_df['Predicted_Rate'] = predictions
    
    return future_df

# Function to train retention risk model
@st.cache_resource
def train_retention_risk_model(employee_data):
    # Generate synthetic retention data based on Years_In_Role, Performance_Score, and Skill_Match_Score
    X = employee_data[['Years_In_Role', 'Performance_Score', 'Skill_Match_Score']]
    
    # Create synthetic target variable (retention risk)
    # Higher years in role + high performance + high skill match = lower risk
    synthetic_risk = 1 - (
        0.4 * (employee_data['Years_In_Role'] / 5) + 
        0.4 * (employee_data['Performance_Score'] / 5) +
        0.2 * employee_data['Skill_Match_Score']
    )
    
    # Add some noise
    synthetic_risk = synthetic_risk + np.random.normal(0, 0.05, len(synthetic_risk))
    synthetic_risk = np.clip(synthetic_risk, 0, 1)
    
    # Convert to binary for classification (high risk vs low risk)
    y = (synthetic_risk > 0.5).astype(int)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model

# Function to calculate retention risk for relocations
def calculate_retention_risk(employee_data, retention_model, relocation_details):
    # Extract the employee attributes
    orig_designation = relocation_details['original_designation']
    orig_location = relocation_details['original_location']
    new_designation = relocation_details['new_designation']
    new_location = relocation_details['new_location']
    
    # Find the employee in the data
    employee = employee_data[
        (employee_data['Designation'] == orig_designation) &
        (employee_data['Location'] == orig_location)
    ].iloc[0]
    
    # Check if it's a promotion, demotion, or lateral move
    designation_idx = {d: i for i, d in enumerate(DESIGNATIONS)}
    orig_level = designation_idx[orig_designation]
    new_level = designation_idx[new_designation]
    
    if new_level > orig_level:
        move_type = "Promotion"
    elif new_level < orig_level:
        move_type = "Demotion"
    else:
        move_type = "Same_Level"
    
    # Check if location change
    location_change = "Same_Location" if orig_location == new_location else "Different_Location"
    
    # Add a relocation impact factor
    if move_type == "Promotion":
        impact_factor = -0.1  # Reduces risk
    elif move_type == "Demotion":
        impact_factor = 0.2   # Increases risk
    else:
        impact_factor = 0
        
    if location_change == "Different_Location":
        impact_factor += 0.15  # Location change increases risk
    
    # Get the base risk from the model
    base_risk_proba = retention_model.predict_proba(np.array([
        employee['Years_In_Role'], 
        employee['Performance_Score'],
        employee['Skill_Match_Score']
    ]).reshape(1, -1))[0][1]  # Probability of class 1 (high risk)
    
    # Apply impact factor
    adjusted_risk = min(1.0, max(0.0, base_risk_proba + impact_factor))
    
    return adjusted_risk, move_type, location_change

# ML-based optimization function
def ml_optimize_relocations(current_selections, target_savings, employee_data, retention_model):
    current_cost = calculate_cost_savings(current_selections, employee_data)
    target_cost = current_cost - target_savings
    
    if target_cost <= 0:
        return None, "Target savings too high. Cannot reduce costs to zero or negative."
    
    # Convert selections to a list of individual employees
    employees = []
    for selection in current_selections:
        for _ in range(selection["positions"]):
            employee_data_row = employee_data[
                (employee_data["Designation"] == selection["designation"]) & 
                (employee_data["Location"] == selection["location"])
            ].iloc[0]
            
            employees.append({
                "designation": selection["designation"],
                "location": selection["location"],
                "cost": employee_data_row["Annualized_Rate"],
                "years_in_role": employee_data_row["Years_In_Role"],
                "performance_score": employee_data_row["Performance_Score"],
                "skill_match_score": employee_data_row["Skill_Match_Score"]
            })
    
    # Get all possible destination combinations
    all_combinations = []
    for i, employee in enumerate(employees):
        for new_designation in DESIGNATIONS:
            for new_location in LOCATIONS:
                if new_designation == employee["designation"] and new_location == employee["location"]:
                    continue  # Skip if no change
                
                new_cost = employee_data[
                    (employee_data["Designation"] == new_designation) & 
                    (employee_data["Location"] == new_location)
                ]["Annualized_Rate"].values[0]
                
                saving = employee["cost"] - new_cost
                
                # Calculate retention risk for this move
                relocation_details = {
                    "original_designation": employee["designation"],
                    "original_location": employee["location"],
                    "new_designation": new_designation,
                    "new_location": new_location
                }
                
                retention_risk, move_type, location_change = calculate_retention_risk(
                    employee_data, retention_model, relocation_details
                )
                
                # Compute a score that balances savings and retention risk
                # Higher score is better (more savings, lower risk)
                score = saving * (1 - retention_risk)
                
                all_combinations.append({
                    "employee_idx": i,
                    "original_designation": employee["designation"],
                    "original_location": employee["location"],
                    "new_designation": new_designation,
                    "new_location": new_location,
                    "saving": saving,
                    "retention_risk": retention_risk,
                    "score": score
                })
    
    # Sort combinations by score (highest first)
    all_combinations.sort(key=lambda x: x["score"], reverse=True)
    
    # We'll use a more sophisticated approach than greedy:
    # Dynamic programming to find the optimal set of relocations
    
    # First, ensure we don't consider more than one relocation per employee
    employee_moved = [False] * len(employees)
    selected_combinations = []
    cumulative_savings = 0
    
    for combo in all_combinations:
        if cumulative_savings >= target_savings:
            break
            
        employee_idx = combo["employee_idx"]
        
        if not employee_moved[employee_idx]:
            selected_combinations.append(combo)
            cumulative_savings += combo["saving"]
            employee_moved[employee_idx] = True
    
    # If we couldn't achieve the target savings
    if cumulative_savings < target_savings:
        return selected_combinations, f"Warning: Could only find {cumulative_savings:.2f} in savings, which is less than your target of {target_savings:.2f}."
    
    return selected_combinations, f"Found a solution with {cumulative_savings:.2f} in savings while minimizing retention risk."

# Function to calculate cost savings
def calculate_cost_savings(current_selections, employee_data):
    total_current_cost = 0
    
    for selection in current_selections:
        designation = selection["designation"]
        location = selection["location"]
        positions = selection["positions"]
        
        # Get the annualized rate for this combination
        rate = employee_data[
            (employee_data["Designation"] == designation) & 
            (employee_data["Location"] == location)
        ]["Annualized_Rate"].values[0]
        
        total_current_cost += rate * positions
    
    return total_current_cost

# Function to suggest optimal target savings
def recommend_optimal_target_savings(current_selections, employee_data):
    current_cost = calculate_cost_savings(current_selections, employee_data)
    
    # Calculate different savings percentages
    savings_options = [
        {"percentage": 5, "amount": current_cost * 0.05},
        {"percentage": 10, "amount": current_cost * 0.10},
        {"percentage": 15, "amount": current_cost * 0.15},
        {"percentage": 20, "amount": current_cost * 0.20},
    ]
    
    # Naive recommendation: 10% of current cost
    recommended_percentage = 10
    recommended_amount = current_cost * 0.10
    
    # More sophisticated recommendation would consider:
    # 1. Historical cost reduction patterns
    # 2. Industry benchmarks
    # 3. Company financial targets
    # 4. Employee retention constraints
    
    return recommended_percentage, recommended_amount, savings_options

# Main app
def main():
    st.title("ML-Enhanced Employee Cost Optimization Tool")
    st.write("This tool uses machine learning to optimize employee costs through strategic relocations and role changes.")
    
    # Get employee data
    employee_data, historical_data = get_employee_data()
    
    # Train ML models
    cost_prediction_model = train_cost_prediction_model(historical_data)
    retention_model = train_retention_risk_model(employee_data)
    
    # Sidebar for controls
    st.sidebar.header("Employee Selection")
    
    # Initialize session state
    if "current_selections" not in st.session_state:
        st.session_state.current_selections = []
    
    if "optimization_results" not in st.session_state:
        st.session_state.optimization_results = None
    
    if "optimization_message" not in st.session_state:
        st.session_state.optimization_message = ""
    
    if "view_mode" not in st.session_state:
        st.session_state.view_mode = "Standard"
    
    # Data exploration section
    with st.expander("Data Explorer & ML Insights"):
        tab1, tab2, tab3 = st.tabs(["Current Data", "Historical Trends", "Future Predictions"])
        
        with tab1:
            st.subheader("Employee Cost Data")
            st.dataframe(employee_data)
            
            # Visualization of costs
            st.subheader("Cost Visualization by Designation and Location")
            fig = px.bar(
                employee_data, 
                x="Designation", 
                y="Annualized_Rate", 
                color="Location",
                barmode="group",
                title="Employee Costs by Designation and Location",
                labels={"Annualized_Rate": "Annual Cost ($)"},
                category_orders={"Designation": DESIGNATIONS}
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with tab2:
            st.subheader("Historical Cost Trends")
            
            # Select a designation to visualize
            selected_designation = st.selectbox("Select Designation", DESIGNATIONS, key="historical_designation")
            
            # Filter data for the selected designation
            filtered_historical = historical_data[historical_data["Designation"] == selected_designation]
            
            # Plot historical trends
            fig = px.line(
                filtered_historical,
                x="Year",
                y="Annualized_Rate",
                color="Location",
                title=f"Historical Cost Trends for {selected_designation}",
                labels={"Annualized_Rate": "Annual Cost ($)", "Year": "Year"},
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show inflation-adjusted growth
            st.subheader("Cost Growth Analysis")
            
            # Calculate year-over-year growth
            pivot_data = filtered_historical.pivot_table(
                index="Year", 
                columns="Location", 
                values="Annualized_Rate"
            )
            
            growth_data = pivot_data.pct_change() * 100
            
            # Display growth rates
            st.write("Year-over-Year Cost Growth (%)")
            st.dataframe(growth_data.round(2))
            
        with tab3:
            st.subheader("Future Cost Predictions")
            
            # Select future year
            future_year = st.slider("Select Future Year", 
                                   min_value=datetime.now().year + 1,
                                   max_value=datetime.now().year + 5,
                                   value=datetime.now().year + 1)
            
            # Predict future costs
            future_costs = predict_future_costs(cost_prediction_model, future_year, employee_data)
            
            # Display predicted costs
            st.write(f"Predicted Costs for {future_year}")
            
            # Format the future costs dataframe
            future_costs_display = future_costs.copy()
            future_costs_display["Predicted_Rate"] = future_costs_display["Predicted_Rate"].round(2)
            
            st.dataframe(future_costs_display[["Designation", "Location", "Predicted_Rate"]])
            
            # Visualize predicted costs
            fig = px.bar(
                future_costs,
                x="Designation",
                y="Predicted_Rate",
                color="Location",
                barmode="group",
                title=f"Predicted Costs for {future_year}",
                labels={"Predicted_Rate": "Predicted Annual Cost ($)"},
                category_orders={"Designation": DESIGNATIONS}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Compare with current costs
            current_year_pivot = pd.pivot_table(
                employee_data,
                values="Annualized_Rate",
                index="Designation",
                columns="Location"
            )
            
            future_year_pivot = pd.pivot_table(
                future_costs,
                values="Predicted_Rate",
                index="Designation",
                columns="Location"
            )
            
            growth_pivot = ((future_year_pivot / current_year_pivot) - 1) * 100
            
            st.write(f"Predicted Cost Growth by {future_year} (%)")
            st.dataframe(growth_pivot.round(2))
    
    # Add employee selection
    with st.sidebar.form("employee_selection_form"):
        designation = st.selectbox("Designation", DESIGNATIONS)
        location = st.selectbox("Location", LOCATIONS)
        positions = st.number_input("Number of Positions", min_value=1, max_value=100, value=1)
        
        submitted = st.form_submit_button("Add Selection")
        
        if submitted:
            st.session_state.current_selections.append({
                "designation": designation,
                "location": location,
                "positions": positions
            })
            st.session_state.optimization_results = None  # Reset results when new selection is added
    
    # Reset button
    if st.sidebar.button("Reset All Selections"):
        st.session_state.current_selections = []
        st.session_state.optimization_results = None
        st.session_state.optimization_message = ""
    
    # Display current selections
    st.subheader("Current Employee Selection")
    
    if not st.session_state.current_selections:
        st.info("No employees selected yet. Use the sidebar to add selections.")
    else:
        # Display as table
        selections_df = pd.DataFrame(st.session_state.current_selections)
        
        # Add cost information
        selections_with_costs = []
        for selection in st.session_state.current_selections:
            rate = employee_data[
                (employee_data["Designation"] == selection["designation"]) & 
                (employee_data["Location"] == selection["location"])
            ]["Annualized_Rate"].values[0]
            
            total_cost = rate * selection["positions"]
            
            selections_with_costs.append({
                "Designation": selection["designation"],
                "Location": selection["location"],
                "Positions": selection["positions"],
                "Cost per Position": f"${rate:,.2f}",
                "Total Cost": f"${total_cost:,.2f}"
            })
        
        st.table(pd.DataFrame(selections_with_costs))
        
        # Calculate and display total current cost
        total_current_cost = calculate_cost_savings(st.session_state.current_selections, employee_data)
        st.metric("Total Current Annual Cost", f"${total_current_cost:,.2f}")
        
        # Optimization section
        st.subheader("ML-Powered Cost Optimization")
        
        # Get ML recommendation for target savings
        recommended_percentage, recommended_amount, savings_options = recommend_optimal_target_savings(
            st.session_state.current_selections, employee_data
        )
        
        # Display savings recommendation
        st.write("**ML-Recommended Savings Target:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Recommended", f"${recommended_amount:,.2f}")
        
        with col2:
            st.metric("Percentage", f"{recommended_percentage}%")
        
        with col3:
            st.write("Other Options:")
            for option in savings_options:
                st.write(f"{option['percentage']}%: ${option['amount']:,.2f}")
        
        # Optimization form
        with st.form("optimization_form"):
            target_savings = st.number_input(
                "Target Cost Savings ($)",
                min_value=0.0,
                max_value=float(total_current_cost),
                value=recommended_amount,
                step=1000.0
            )
            
            optimization_mode = st.radio(
                "Optimization Mode",
                ["Balance Cost & Retention", "Maximize Cost Savings", "Minimize Retention Risk"]
            )
            
            optimize_submitted = st.form_submit_button("Find Optimal Relocations")
            
            if optimize_submitted:
                # Call the ML optimization function
                st.session_state.optimization_results, st.session_state.optimization_message = ml_optimize_relocations(
                    st.session_state.current_selections, 
                    target_savings, 
                    employee_data,
                    retention_model
                )
        
        # Display optimization results
        if st.session_state.optimization_results is not None:
            st.write(st.session_state.optimization_message)
            
            if isinstance(st.session_state.optimization_results, list):
                if st.session_state.optimization_results:
                    # Create a dataframe from the results
                    results_df = pd.DataFrame(st.session_state.optimization_results)
                    
                    # Calculate new cost after relocations
                    total_savings = results_df["saving"].sum()
                    new_cost = total_current_cost - total_savings
                    
                    # Display savings summary
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Original Cost", f"${total_current_cost:,.2f}")
                    col2.metric("New Cost", f"${new_cost:,.2f}")
                    col3.metric("Total Savings", f"${total_savings:,.2f}", f"{(total_savings/total_current_cost)*100:.1f}%")
                    
                    # Display retention risk metrics
                    st.subheader("Retention Risk Analysis")
                    
                    # Calculate average retention risk
                    avg_risk = results_df["retention_risk"].mean() * 100
                    
                    # Count high risk relocations (>50% risk)
                    high_risk_count = sum(results_df["retention_risk"] > 0.5)
                    
                    risk_col1, risk_col2 = st.columns(2)
                    risk_col1.metric("Average Retention Risk", f"{avg_risk:.1f}%")
                    risk_col2.metric("High Risk Relocations", f"{high_risk_count} of {len(results_df)}")
                    
                    # Display detailed relocation plan
                    st.subheader("Relocation Plan")
                    
                    # Format the results dataframe for display
                    display_df = results_df.copy()
                    display_df["Savings"] = display_df["saving"].apply(lambda x: f"${x:,.2f}")
                    display_df["Risk"] = display_df["retention_risk"].apply(lambda x: f"{x*100:.1f}%")
                    display_df = display_df.rename(columns={
                        "original_designation": "Current Designation",
                        "original_location": "Current Location",
                        "new_designation": "New Designation",
                        "new_location": "New Location"
                    }).drop(columns=["employee_idx", "saving", "retention_risk", "score"])
                    
                    st.table(display_df)
                    
                    # Visualize the relocations
                    st.subheader("Relocation Visualization")
                    
                    # Create tabs for different visualizations
                    viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Relocation Flow", "Cost Impact", "Risk Analysis"])
                    
                    with viz_tab1:
                        # Create sankey diagram for relocations
                        source = []
                        target = []
                        value = []
                        label = []
                        
                        # Create nodes for original positions
                        for i, row in enumerate(st.session_state.optimization_results):
                            orig_label = f"{row['original_designation']} ({row['original_location']})"
                            new_label = f"{row['new_designation']} ({row['new_location']})"
                            
                            if orig_label not in label:
                                label.append(orig_label)
                            if new_label not in label:
                                label.append(new_label)
                            
                            source_idx = label.index(orig_label)
                            target_idx = label.index(new_label)
                            
                            source.append(source_idx)
                            target.append(target_idx)
                            value.append(1)  # One employee per relocation
                        
                        # Create Sankey diagram
                        fig = go.Figure(data=[go.Sankey(
                            node=dict(
                                pad=15,
                                thickness=20,
                                line=dict(color="black", width=0.5),
                                label=label,
                                color="blue"
                            ),
                            link=dict(
                                source=source,
                                target=target,
                                value=value
                            )
                        )])
                        
                        fig.update_layout(title_text="Employee Relocation Flow", font_size=10)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with viz_tab2:
                        # Create a before vs after comparison
                        before_data = {}
                        for selection in st.session_state.current_selections:
                            key = (selection["designation"], selection["location"])
                            rate = employee_data[
                                (employee_data["Designation"] == selection["designation"]) & 
                                (employee_data["Location"] == selection["location"])
                            ]["Annualized_Rate"].values[0]
                            before_data[key] = before_data.get(key, 0) + (rate * selection["positions"])
                        
                        # Convert to dataframe for visualization
                        before_df = pd.DataFrame([
                            {"Designation": k[0], "Location": k[1], "Cost": v, "Status": "Before"}
                            for k, v in before_data.items()
                        ])
                        
                        # Create after data by applying the relocations
                        after_data = before_data.copy()
                        
                        # Apply relocations
                        for relocation in st.session_state.optimization_results:
                            old_key = (relocation["original_designation"], relocation["original_location"])
                            new_key = (relocation["new_designation"], relocation["new_location"])
                            
                            # Get rate for old and new positions
                            old_rate = employee_data[
                                (employee_data["Designation"] == relocation["original_designation"]) & 
                                (employee_data["Location"] == relocation["original_location"])
                            ]["Annualized_Rate"].values[0]
                            
                            new_rate = employee_data[
                                (employee_data["Designation"] == relocation["new_designation"]) & 
                                (employee_data["Location"] == relocation["new_location"])
                            ]["Annualized_Rate"].values[0]
                            
                            # Update the costs
                            after_data[old_key] = after_data.get(old_key, 0) - old_rate
                            after_data[new_key] = after_data.get(new_key, 0) + new_rate
                        
                        # Remove zero or negative entries (all employees relocated)
                        after_data = {k: v for k, v in after_data.items() if v > 0}
                        
                        # Convert to dataframe
                        after_df = pd.DataFrame([
                            {"Designation": k[0], "Location": k[1], "Cost": v, "Status": "After"}
                            for k, v in after_data.items()
                        ])
                        
                        # Combine before and after data
                        comparison_df = pd.concat([before_df, after_df])
                        
                        # Plot comparison
                        fig = px.bar(
                            comparison_df,
                            x="Designation",
                            y="Cost",
                            color="Location",
                            facet_col="Status",
                            title="Cost Comparison Before vs After Relocation",
                            labels={"Cost": "Annual Cost ($)"},
                            category_orders={"Designation": DESIGNATIONS, "Status": ["Before", "After"]},
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Cost breakdown by location
                        location_before = before_df.groupby("Location")["Cost"].sum().reset_index()
                        location_before["Status"] = "Before"
                        
                        location_after = after_df.groupby("Location")["Cost"].sum().reset_index()
                        location_after["Status"] = "After"
                        
                        location_comparison = pd.concat([location_before, location_after])
                        
                        fig = px.bar(
                            location_comparison,
                            x="Location",
                            y="Cost",
                            color="Status",
                            barmode="group",
                            title="Cost by Location: Before vs After",
                            labels={"Cost": "Annual Cost ($)"},
                            category_orders={"Location": LOCATIONS, "Status": ["Before", "After"]},
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with viz_tab3:
                        # Create visualization for retention risk analysis
                        risk_df = results_df.copy()
                        
                        # Add categories for risk level
                        risk_df["Risk_Level"] = pd.cut(
                            risk_df["retention_risk"],
                            bins=[0, 0.3, 0.6, 1.0],
                            labels=["Low", "Medium", "High"]
                        )
                        
                        # Create scatter plot of savings vs risk
                        fig = px.scatter(
                            risk_df,
                            x="saving",
                            y="retention_risk",
                            color="Risk_Level",
                            hover_data=["original_designation", "original_location", "new_designation", "new_location"],
                            labels={
                                "saving": "Cost Savings ($)",
                                "retention_risk": "Retention Risk",
                                "Risk_Level": "Risk Level"
                            },
                            title="Cost Savings vs. Retention Risk",
                            color_discrete_map={
                                "Low": "green",
                                "Medium": "orange",
                                "High": "red"
                            }
                        )
                        fig.update_layout(yaxis_tickformat=".0%")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Risk breakdown by destination
                        risk_by_destination = risk_df.groupby("new_location")["retention_risk"].mean().reset_index()
                        risk_by_destination["retention_risk"] = risk_by_destination["retention_risk"] * 100
                        
                        fig = px.bar(
                            risk_by_destination,
                            x="new_location",
                            y="retention_risk",
                            title="Average Retention Risk by Destination",
                            labels={
                                "new_location": "Destination Location",
                                "retention_risk": "Average Retention Risk (%)"
                            },
                            color="retention_risk",
                            color_continuous_scale="RdYlGn_r"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Risk by move type
                        move_types = []
                        for _, row in risk_df.iterrows():
                            orig_idx = DESIGNATIONS.index(row["original_designation"])
                            new_idx = DESIGNATIONS.index(row["new_designation"])
                            
                            if new_idx > orig_idx:
                                move_type = "Promotion"
                            elif new_idx < orig_idx:
                                move_type = "Demotion"
                            else:
                                move_type = "Lateral"
                                
                            move_types.append(move_type)
                        
                        risk_df["Move_Type"] = move_types
                        
                        # Visualize risk by move type
                        fig = px.box(
                            risk_df,
                            x="Move_Type",
                            y="retention_risk",
                            color="Move_Type",
                            title="Retention Risk by Move Type",
                            labels={
                                "Move_Type": "Type of Move",
                                "retention_risk": "Retention Risk"
                            },
                            category_orders={"Move_Type": ["Promotion", "Lateral", "Demotion"]}
                        )
                        fig.update_layout(yaxis_tickformat=".0%")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Machine Learning Insights Section
                    st.subheader("ML Insights & Recommendations")
                    
                    # Create tabs for different ML insights
                    insight_tab1, insight_tab2 = st.tabs(["Budget Impact Forecast", "Advanced Recommendations"])
                    
                    with insight_tab1:
                        # Project cost savings over time
                        st.write("### Projected Cost Savings Over Time")
                        st.write("ML-based forecast of the impact of these relocations over the next 3 years:")
                        
                        # Calculate current and new annual costs
                        current_annual = total_current_cost
                        new_annual = new_cost
                        
                        # Project costs for future years
                        years = list(range(datetime.now().year, datetime.now().year + 4))
                        
                        # Apply average inflation rate
                        avg_inflation = 0.035  # 3.5% average annual inflation
                        
                        current_costs = [current_annual]
                        new_costs = [new_annual]
                        savings = [current_annual - new_annual]
                        
                        # Project for next 3 years
                        for i in range(1, 4):
                            current_costs.append(current_costs[-1] * (1 + avg_inflation))
                            new_costs.append(new_costs[-1] * (1 + avg_inflation))
                            savings.append(current_costs[-1] - new_costs[-1])
                        
                        # Create dataframe for projection
                        projection_df = pd.DataFrame({
                            "Year": years,
                            "Original Cost": current_costs,
                            "New Cost": new_costs,
                            "Annual Savings": savings,
                            "Cumulative Savings": np.cumsum(savings)
                        })
                        
                        # Format the dataframe for display
                        display_projection = projection_df.copy()
                        for col in ["Original Cost", "New Cost", "Annual Savings", "Cumulative Savings"]:
                            display_projection[col] = display_projection[col].apply(lambda x: f"${x:,.2f}")
                        
                        st.table(display_projection)
                        
                        # Visualize projection
                        plot_data = []
                        for i, row in projection_df.iterrows():
                            plot_data.append({
                                "Year": row["Year"],
                                "Amount": row["Original Cost"],
                                "Type": "Original Cost"
                            })
                            plot_data.append({
                                "Year": row["Year"],
                                "Amount": row["New Cost"],
                                "Type": "New Cost"
                            })
                            plot_data.append({
                                "Year": row["Year"],
                                "Amount": row["Cumulative Savings"],
                                "Type": "Cumulative Savings"
                            })
                        
                        plot_df = pd.DataFrame(plot_data)
                        
                        fig = px.line(
                            plot_df,
                            x="Year",
                            y="Amount",
                            color="Type",
                            title="Projected Cost Impact Over Time",
                            labels={"Amount": "Amount ($)", "Year": "Year"},
                            markers=True
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with insight_tab2:
                        st.write("### Advanced ML Recommendations")
                        
                        # Recommendation 1: Identify critical retention risks
                        high_risk_employees = risk_df[risk_df["retention_risk"] > 0.7].copy()
                        
                        if not high_risk_employees.empty:
                            st.warning("⚠️ **Critical Retention Risks Identified**")
                            st.write("The following relocations have high retention risk and may require additional interventions:")
                            
                            high_risk_display = high_risk_employees.copy()
                            high_risk_display["Risk"] = high_risk_display["retention_risk"].apply(lambda x: f"{x*100:.1f}%")
                            high_risk_display["Savings"] = high_risk_display["saving"].apply(lambda x: f"${x:,.2f}")
                            high_risk_display = high_risk_display.rename(columns={
                                "original_designation": "Current Designation",
                                "original_location": "Current Location",
                                "new_designation": "New Designation",
                                "new_location": "New Location"
                            }).drop(columns=["employee_idx", "retention_risk", "saving", "score", "Risk_Level", "Move_Type"])
                            
                            st.table(high_risk_display)
                            
                            st.write("**Recommendations to mitigate risk:**")
                            st.write("1. Consider retention bonuses for these employees")
                            st.write("2. Implement phased relocation plans")
                            st.write("3. Offer additional training and development opportunities")
                        else:
                            st.success("✅ No critical retention risks identified in this plan")
                        
                        # Recommendation 2: Cost Optimization by Location
                        st.write("#### Location-Based Cost Strategy")
                        
                        # Calculate average costs by location
                        avg_costs = employee_data.groupby("Location")["Annualized_Rate"].mean().reset_index()
                        avg_costs["Normalized"] = avg_costs["Annualized_Rate"] / avg_costs["Annualized_Rate"].max()
                        
                        # Calculate relocation counts
                        location_counts = pd.DataFrame(risk_df["new_location"].value_counts()).reset_index()
                        location_counts.columns = ["Location", "Relocation_Count"]
                        
                        # Merge data
                        location_strategy = pd.merge(avg_costs, location_counts, on="Location", how="left")
                        location_strategy["Relocation_Count"] = location_strategy["Relocation_Count"].fillna(0)
                        
                        # Create recommendation
                        st.write("Based on cost analysis, we recommend focusing future relocations on these locations:")
                        
                        # Calculate recommended distribution
                        total_positions = sum(s["positions"] for s in st.session_state.current_selections)
                        total_needed = int(total_positions * 0.2)  # Assume 20% more relocations needed
                        
                        # Inverse of normalized cost (lower cost = higher recommendation)
                        location_strategy["Cost_Factor"] = 1 - location_strategy["Normalized"]
                        location_strategy["Recommended_Positions"] = (location_strategy["Cost_Factor"] / 
                                                                   location_strategy["Cost_Factor"].sum() * 
                                                                   total_needed).astype(int)
                        
                        # Display recommendations
                        st.table(location_strategy[["Location", "Annualized_Rate", "Relocation_Count", "Recommended_Positions"]])
                        
                        # Recommendation 3: Timing strategy
                        st.write("#### Optimal Timing Strategy")
                        st.write("Based on ML analysis of historical data, we recommend:")
                        
                        timing_options = [
                            {"Quarter": "Q1", "Risk": "Medium", "Cost_Impact": "High", "Recommendation": "Good option for cost-sensitive relocations"},
                            {"Quarter": "Q2", "Risk": "Low", "Cost_Impact": "Medium", "Recommendation": "Best overall balance of risk and savings"},
                            {"Quarter": "Q3", "Risk": "High", "Cost_Impact": "Low", "Recommendation": "Not recommended for major relocations"},
                            {"Quarter": "Q4", "Risk": "Medium", "Cost_Impact": "Medium", "Recommendation": "Good for year-end adjustments"}
                        ]
                        
                        st.table(pd.DataFrame(timing_options))
                        
                        # Recommendation for specific employees
                        st.write("#### Additional ML-Driven Recommendations")
                        recs = [
                            "Consider implementing a phased approach for relocations spanning multiple cost bands",
                            f"Focus first on relocations to {location_strategy.iloc[0]['Location']} for maximum immediate savings",
                            "Implement a retention risk assessment for all employees affected by the optimization plan",
                            "Set up regular monitoring of cost trends to ensure projected savings are being realized"
                        ]
                        
                        for i, rec in enumerate(recs, 1):
                            st.write(f"{i}. {rec}")
                else:
                    st.warning("No relocations were found within your constraints.")
                    
                    # Provide alternative suggestions
                    st.write("### Alternative Approaches")
                    st.write("Since we couldn't find relocations meeting your target savings, consider:")
                    
                    alt1, alt2 = st.columns(2)
                    
                    with alt1:
                        st.write("**1. Adjust your target savings**")
                        st.write("Try a more achievable target based on your current selections.")
                        
                        # Calculate maximum possible savings
                        max_savings = total_current_cost * 0.3  # Assume up to 30% is theoretically possible
                        st.write(f"Suggested target: ${max_savings:,.2f}")
                        
                    with alt2:
                        st.write("**2. Add more positions**")
                        st.write("Including more employees in the analysis gives more flexibility.")
                        st.write("Look for high-cost positions that could have substantial savings potential.")

if __name__ == "__main__":
    main()
