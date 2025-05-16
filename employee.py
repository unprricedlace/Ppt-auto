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

# Set page configuration
st.set_page_config(
    page_title="Employee Cost Optimization Tool",
    page_icon="💼",
    layout="wide"
)

# Define constants
DESIGNATIONS = ["Analyst", "Associate", "AVP", "VP", "ED", "MD"]
LOCATIONS = ["India", "US", "UK"]

# Function to generate synthetic data
def generate_employee_cost_data():
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
    
    # Create dataframe with all combinations
    data = []
    for designation, location in product(DESIGNATIONS, LOCATIONS):
        annualized_rate = base_salaries[designation] * location_multipliers[location]
        # Add some random variation (±5%)
        annualized_rate = annualized_rate * random.uniform(0.95, 1.05)
        data.append({
            "Designation": designation,
            "Location": location,
            "Annualized_Rate": round(annualized_rate, 2)
        })
    
    return pd.DataFrame(data)

# Create the employee cost data
@st.cache_data
def get_employee_data():
    return generate_employee_cost_data()

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

# Function to suggest optimal relocations
def suggest_optimal_relocations(current_selections, target_savings, employee_data):
    current_cost = calculate_cost_savings(current_selections, employee_data)
    target_cost = current_cost - target_savings
    
    if target_cost <= 0:
        return None, "Target savings too high. Cannot reduce costs to zero or negative."
    
    # Convert selections to a list of individual employees
    employees = []
    for selection in current_selections:
        for _ in range(selection["positions"]):
            employees.append({
                "designation": selection["designation"],
                "location": selection["location"],
                "cost": employee_data[
                    (employee_data["Designation"] == selection["designation"]) & 
                    (employee_data["Location"] == selection["location"])
                ]["Annualized_Rate"].values[0]
            })
    
    # Get all possible destination combinations
    all_combinations = []
    for employee in employees:
        for new_designation in DESIGNATIONS:
            for new_location in LOCATIONS:
                if new_designation == employee["designation"] and new_location == employee["location"]:
                    continue  # Skip if no change
                
                new_cost = employee_data[
                    (employee_data["Designation"] == new_designation) & 
                    (employee_data["Location"] == new_location)
                ]["Annualized_Rate"].values[0]
                
                saving = employee["cost"] - new_cost
                
                all_combinations.append({
                    "original_designation": employee["designation"],
                    "original_location": employee["location"],
                    "new_designation": new_designation,
                    "new_location": new_location,
                    "saving": saving
                })
    
    # Sort combinations by savings (highest first)
    all_combinations.sort(key=lambda x: x["saving"], reverse=True)
    
    # Greedy approach to select the best combinations
    selected_combinations = []
    cumulative_savings = 0
    total_employees = len(employees)
    
    # Try to find a solution moving the minimum number of employees
    for combo in all_combinations:
        if cumulative_savings >= target_savings:
            break
            
        selected_combinations.append(combo)
        cumulative_savings += combo["saving"]
        
        # Ensure we don't select more moves than we have employees
        if len(selected_combinations) >= total_employees:
            break
    
    # If we couldn't achieve the target savings
    if cumulative_savings < target_savings:
        return selected_combinations, f"Warning: Could only find {cumulative_savings:.2f} in savings, which is less than your target of {target_savings:.2f}."
    
    return selected_combinations, f"Found a solution with {cumulative_savings:.2f} in savings by relocating {len(selected_combinations)} employees."

# Main app
def main():
    st.title("Employee Cost Optimization Tool")
    st.write("This tool helps optimize employee costs through strategic relocations and role changes.")
    
    # Get employee data
    employee_data = get_employee_data()
    
    # Sidebar for controls
    st.sidebar.header("Employee Selection")
    
    # Initialize session state
    if "current_selections" not in st.session_state:
        st.session_state.current_selections = []
    
    if "optimization_results" not in st.session_state:
        st.session_state.optimization_results = None
    
    if "optimization_message" not in st.session_state:
        st.session_state.optimization_message = ""
    
    # Data exploration section
    with st.expander("Data Explorer"):
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
        st.subheader("Cost Optimization")
        
        with st.form("optimization_form"):
            target_savings = st.number_input(
                "Target Cost Savings ($)",
                min_value=0.0,
                max_value=float(total_current_cost),
                value=total_current_cost * 0.1,  # Default to 10% savings
                step=1000.0
            )
            
            optimize_submitted = st.form_submit_button("Find Optimal Relocations")
            
            if optimize_submitted:
                st.session_state.optimization_results, st.session_state.optimization_message = suggest_optimal_relocations(
                    st.session_state.current_selections, 
                    target_savings, 
                    employee_data
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
                    
                    # Display detailed relocation plan
                    st.subheader("Relocation Plan")
                    
                    # Format the results dataframe for display
                    display_df = results_df.copy()
                    display_df["Savings"] = display_df["saving"].apply(lambda x: f"${x:,.2f}")
                    display_df = display_df.rename(columns={
                        "original_designation": "Current Designation",
                        "original_location": "Current Location",
                        "new_designation": "New Designation",
                        "new_location": "New Location"
                    }).drop(columns=["saving"])
                    
                    st.table(display_df)
                    
                    # Visualize the relocations
                    st.subheader("Relocation Visualization")
                    
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
                    
                    # Show cost comparison by location and designation
                    st.subheader("Cost Comparison")
                    
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
                    
                    # Cost breakdown by designation
                    designation_before = before_df.groupby("Designation")["Cost"].sum().reset_index()
                    designation_before["Status"] = "Before"
                    
                    designation_after = after_df.groupby("Designation")["Cost"].sum().reset_index()
                    designation_after["Status"] = "After"
                    
                    designation_comparison = pd.concat([designation_before, designation_after])
                    
                    fig = px.bar(
                        designation_comparison,
                        x="Designation",
                        y="Cost",
                        color="Status",
                        barmode="group",
                        title="Cost by Designation: Before vs After",
                        labels={"Cost": "Annual Cost ($)"},
                        category_orders={"Designation": DESIGNATIONS, "Status": ["Before", "After"]},
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No relocations were found within your constraints.")

if __name__ == "__main__":
    main()
