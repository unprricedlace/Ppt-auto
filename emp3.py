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
from collections import defaultdict
import math

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

# Enhanced optimization function with preference constraints
def suggest_optimal_relocations_enhanced(current_selections, target_savings, employee_data, constraints, scenario_type="max_savings"):
    current_cost = calculate_cost_savings(current_selections, employee_data)
    target_cost = current_cost - target_savings
    
    if target_cost <= 0:
        return None, "Target savings too high. Cannot reduce costs to zero or negative."
    
    # Convert selections to a list of individual positions with their details
    positions = []
    position_id = 0
    for selection in current_selections:
        for i in range(selection["positions"]):
            cost = employee_data[
                (employee_data["Designation"] == selection["designation"]) & 
                (employee_data["Location"] == selection["location"])
            ]["Annualized_Rate"].values[0]
            
            positions.append({
                "id": position_id,
                "designation": selection["designation"],
                "location": selection["location"],
                "cost": cost,
                "preferred_locations": selection.get("preferred_locations", []),
                "preferred_designations": selection.get("preferred_designations", []),
                "original_selection_idx": position_id
            })
            position_id += 1
    
    # Get all possible moves for each position
    all_moves = []
    for pos in positions:
        for new_designation in DESIGNATIONS:
            for new_location in LOCATIONS:
                # Skip if no change
                if new_designation == pos["designation"] and new_location == pos["location"]:
                    continue
                
                # Check if move matches preferences (only allow moves that match preferences)
                if not is_move_preferred(pos, new_designation, new_location):
                    continue
                
                # Check other constraints
                if not is_move_allowed(pos, new_designation, new_location, constraints):
                    continue
                
                new_cost = employee_data[
                    (employee_data["Designation"] == new_designation) & 
                    (employee_data["Location"] == new_location)
                ]["Annualized_Rate"].values[0]
                
                saving = pos["cost"] - new_cost
                efficiency = saving / pos["cost"] if pos["cost"] > 0 else 0  # Savings as percentage
                
                all_moves.append({
                    "position_id": pos["id"],
                    "original_designation": pos["designation"],
                    "original_location": pos["location"],
                    "new_designation": new_designation,
                    "new_location": new_location,
                    "saving": saving,
                    "efficiency": efficiency,
                    "original_cost": pos["cost"],
                    "new_cost": new_cost
                })
    
    # Enhanced selection algorithm based on scenario
    selected_moves = smart_move_selection(all_moves, target_savings, constraints, scenario_type)
    
    # Calculate actual savings
    total_savings = sum(move["saving"] for move in selected_moves)
    
    # Group moves by destination for summary
    moves_summary = defaultdict(list)
    for move in selected_moves:
        key = (move["new_designation"], move["new_location"])
        moves_summary[key].append(move)
    
    # Create summary message
    if total_savings >= target_savings * 0.95:  # Within 5% of target
        message = f"✅ Successfully found a solution with ${total_savings:,.2f} in savings (Target: ${target_savings:,.2f})"
    else:
        message = f"⚠️ Partial solution found: ${total_savings:,.2f} in savings (Target: ${target_savings:,.2f})"
    
    message += f"\n📊 Total positions to be relocated: {len(selected_moves)}"
    
    # Add summary by destination
    if moves_summary:
        message += "\n\n📍 Relocation Summary:"
        for (designation, location), moves in moves_summary.items():
            message += f"\n• {len(moves)} position(s) to {designation} in {location}"
    
    return selected_moves, message

def is_move_preferred(position, new_designation, new_location):
    """Check if a move matches user preferences"""
    preferred_locations = position.get("preferred_locations", [])
    preferred_designations = position.get("preferred_designations", [])
    
    # If no preferences specified, allow all moves
    location_match = len(preferred_locations) == 0 or new_location in preferred_locations
    designation_match = len(preferred_designations) == 0 or new_designation in preferred_designations
    
    return location_match and designation_match

def is_move_allowed(position, new_designation, new_location, constraints):
    """Check if a move is allowed based on constraints"""
    
    # Location capacity constraints
    if constraints.get("location_limits", {}).get(new_location, float('inf')) <= 0:
        return False
    
    # Role change constraints
    if constraints.get("allow_role_changes", True) == False:
        if position["designation"] != new_designation:
            return False
    
    # Senior role protection (prevent downgrading senior roles)
    if constraints.get("protect_senior_roles", False):
        designation_hierarchy = {des: i for i, des in enumerate(DESIGNATIONS)}
        if (position["designation"] in ["VP", "ED", "MD"] and 
            designation_hierarchy[new_designation] < designation_hierarchy[position["designation"]]):
            return False
    
    return True

def smart_move_selection(all_moves, target_savings, constraints, scenario_type):
    """Enhanced move selection algorithm"""
    
    if scenario_type == "max_savings":
        # Sort by absolute savings (highest first)
        all_moves.sort(key=lambda x: x["saving"], reverse=True)
    else:  # preference_optimized
        # Sort by efficiency (savings percentage)
        all_moves.sort(key=lambda x: x["efficiency"], reverse=True)
    
    # Track location capacities
    location_counts = defaultdict(int)
    selected_moves = []
    cumulative_savings = 0
    used_positions = set()
    
    # Apply location limits from constraints
    location_limits = constraints.get("location_limits", {})
    
    for move in all_moves:
        # Skip if position already used
        if move["position_id"] in used_positions:
            continue
        
        # Check if we've reached the target
        if cumulative_savings >= target_savings:
            break
        
        # Check location capacity
        new_location = move["new_location"]
        if new_location in location_limits:
            if location_counts[new_location] >= location_limits[new_location]:
                continue
        
        # Select this move
        selected_moves.append(move)
        cumulative_savings += move["saving"]
        used_positions.add(move["position_id"])
        location_counts[new_location] += 1
    
    return selected_moves

# Main app
def main():
    st.title("🚀 Enhanced Employee Cost Optimization Tool")
    st.write("Advanced tool for optimizing employee costs with preference-based constraints and multiple optimization scenarios.")
    
    # Get employee data
    employee_data = get_employee_data()
    st.session_state["employee_data"] = employee_data
    
    # Sidebar for controls
    st.sidebar.header("🔧 Configuration")
    
    # Initialize session state
    if "current_selections" not in st.session_state:
        st.session_state.current_selections = []
    
    if "optimization_results" not in st.session_state:
        st.session_state.optimization_results = {}
    
    if "optimization_messages" not in st.session_state:
        st.session_state.optimization_messages = {}
    
    # Constraints section in sidebar
    st.sidebar.subheader("🎯 Optimization Constraints")
    
    # Location limits
    st.sidebar.write("**Location Capacity Limits:**")
    location_limits = {}
    for location in LOCATIONS:
        limit = st.sidebar.number_input(
            f"Max positions in {location}",
            min_value=0,
            max_value=1000,
            value=100,
            key=f"limit_{location}"
        )
        if limit > 0:
            location_limits[location] = limit
    
    # Other constraints
    allow_role_changes = st.sidebar.checkbox("Allow role/designation changes", value=True)
    protect_senior_roles = st.sidebar.checkbox("Protect senior roles from demotion", value=True)
    
    constraints = {
        "location_limits": location_limits,
        "allow_role_changes": allow_role_changes,
        "protect_senior_roles": protect_senior_roles
    }
    
    # Data exploration section
    with st.expander("📊 Data Explorer"):
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
    
    # Employee selection section
    st.sidebar.subheader("👥 Employee Selection")
    
    # Add employee selection with preferences
    with st.sidebar.form("employee_selection_form"):
        designation = st.selectbox("Current Designation", DESIGNATIONS)
        location = st.selectbox("Current Location", LOCATIONS)
        positions = st.number_input("Number of Positions", min_value=1, max_value=100, value=1)
        
        st.write("**Preferred Destinations:**")
        preferred_locations = st.multiselect(
            "Preferred Locations (leave empty for any)",
            LOCATIONS,
            default=[]
        )
        
        preferred_designations = st.multiselect(
            "Preferred Designations (leave empty for any)",
            DESIGNATIONS,
            default=[]
        )
        
        submitted = st.form_submit_button("Add Selection")
        
        if submitted:
            st.session_state.current_selections.append({
                "designation": designation,
                "location": location,
                "positions": positions,
                "preferred_locations": preferred_locations,
                "preferred_designations": preferred_designations
            })
            st.session_state.optimization_results = {}
            st.session_state.optimization_messages = {}
    
    # Reset button
    if st.sidebar.button("🔄 Reset All Selections"):
        st.session_state.current_selections = []
        st.session_state.optimization_results = {}
        st.session_state.optimization_messages = {}
    
    # Display current selections
    st.subheader("📋 Current Employee Selection")
    
    if not st.session_state.current_selections:
        st.info("No employees selected yet. Use the sidebar to add selections.")
    else:
        # Display as table with enhanced formatting including preferences
        selections_with_costs = []
        for i, selection in enumerate(st.session_state.current_selections):
            rate = employee_data[
                (employee_data["Designation"] == selection["designation"]) & 
                (employee_data["Location"] == selection["location"])
            ]["Annualized_Rate"].values[0]
            
            total_cost = rate * selection["positions"]
            
            pref_locations = ", ".join(selection.get("preferred_locations", [])) or "Any"
            pref_designations = ", ".join(selection.get("preferred_designations", [])) or "Any"
            
            selections_with_costs.append({
                "ID": i + 1,
                "Current Designation": selection["designation"],
                "Current Location": selection["location"],
                "Positions": selection["positions"],
                "Preferred Locations": pref_locations,
                "Preferred Designations": pref_designations,
                "Cost per Position": f"${rate:,.2f}",
                "Total Cost": f"${total_cost:,.2f}"
            })
        
        st.dataframe(pd.DataFrame(selections_with_costs), use_container_width=True)
        
        # Calculate and display total current cost
        total_current_cost = calculate_cost_savings(st.session_state.current_selections, employee_data)
        
        col1, col2 = st.columns(2)
        col1.metric("💰 Total Current Annual Cost", f"${total_current_cost:,.2f}")
        col2.metric("👥 Total Positions", sum(s["positions"] for s in st.session_state.current_selections))
        
        # Optimization section
        st.subheader("🎯 Cost Optimization")
        
        with st.form("optimization_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                target_savings = st.number_input(
                    "Target Cost Savings ($)",
                    min_value=0.0,
                    max_value=float(total_current_cost),
                    value=total_current_cost * 0.15,  # Default to 15% savings
                    step=1000.0
                )
            
            with col2:
                target_percentage = (target_savings / total_current_cost * 100) if total_current_cost > 0 else 0
                st.metric("Target Savings %", f"{target_percentage:.1f}%")
            
            scenario_choice = st.radio(
                "Optimization Scenario:",
                ["Both Scenarios", "Maximum Savings Only", "Preference Optimized Only"],
                index=0
            )
            
            optimize_submitted = st.form_submit_button("🚀 Find Optimal Relocations")
            
            if optimize_submitted:
                with st.spinner("Analyzing optimal relocations..."):
                    scenarios_to_run = []
                    
                    if scenario_choice == "Both Scenarios":
                        scenarios_to_run = [("max_savings", "Maximum Savings"), ("preference_optimized", "Preference Optimized")]
                    elif scenario_choice == "Maximum Savings Only":
                        scenarios_to_run = [("max_savings", "Maximum Savings")]
                    else:
                        scenarios_to_run = [("preference_optimized", "Preference Optimized")]
                    
                    for scenario_type, scenario_name in scenarios_to_run:
                        result, message = suggest_optimal_relocations_enhanced(
                            st.session_state.current_selections, 
                            target_savings, 
                            employee_data,
                            constraints,
                            scenario_type
                        )
                        st.session_state.optimization_results[scenario_name] = result
                        st.session_state.optimization_messages[scenario_name] = message
        
        # Display optimization results
        if st.session_state.optimization_results:
            st.markdown("---")
            st.subheader("📈 Optimization Results")
            
            # Create tabs for different scenarios
            scenario_names = list(st.session_state.optimization_results.keys())
            if len(scenario_names) > 1:
                tabs = st.tabs(scenario_names)
                tab_containers = tabs
            else:
                tab_containers = [st.container()]
            
            for i, scenario_name in enumerate(scenario_names):
                with tab_containers[i]:
                    st.subheader(f"📊 {scenario_name} Results")
                    
                    # Display message
                    st.markdown(st.session_state.optimization_messages[scenario_name])
                    
                    results = st.session_state.optimization_results[scenario_name]
                    
                    if isinstance(results, list) and results:
                        results_df = pd.DataFrame(results)
                        
                        # Calculate metrics
                        total_savings = results_df["saving"].sum()
                        new_cost = total_current_cost - total_savings
                        positions_moved = len(results_df)
                        total_positions = sum(s["positions"] for s in st.session_state.current_selections)
                        
                        # Display key metrics
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("💵 Original Cost", f"${total_current_cost:,.2f}")
                        col2.metric("💰 New Cost", f"${new_cost:,.2f}")
                        col3.metric("📉 Total Savings", f"${total_savings:,.2f}", f"{(total_savings/total_current_cost)*100:.1f}%")
                        col4.metric("🔄 Positions Moved", f"{positions_moved}/{total_positions}", f"{(positions_moved/total_positions)*100:.1f}%")
                        
                        # Detailed relocation table
                        st.subheader("📋 Detailed Relocation Plan")
                        
                        # Format results for display
                        display_df = results_df.copy()
                        display_df["Savings"] = display_df["saving"].apply(lambda x: f"${x:,.2f}")
                        display_df["Efficiency"] = display_df["efficiency"].apply(lambda x: f"{x*100:.1f}%")
                        display_df["Original Cost"] = display_df["original_cost"].apply(lambda x: f"${x:,.2f}")
                        display_df["New Cost"] = display_df["new_cost"].apply(lambda x: f"${x:,.2f}")
                        
                        display_columns = {
                            "position_id": "Position ID",
                            "original_designation": "From Designation",
                            "original_location": "From Location", 
                            "new_designation": "To Designation",
                            "new_location": "To Location",
                            "Original Cost": "Original Cost",
                            "New Cost": "New Cost",
                            "Savings": "Savings",
                            "Efficiency": "Savings %"
                        }
                        
                        final_display_df = display_df.rename(columns=display_columns)[list(display_columns.values())]
                        st.dataframe(final_display_df, use_container_width=True)
                        
                        # Summary by destination
                        st.subheader("📍 Relocation Summary by Destination")
                        
                        summary_data = []
                        for (designation, location), group in results_df.groupby(["new_designation", "new_location"]):
                            summary_data.append({
                                "Destination": f"{designation} - {location}",
                                "Positions": len(group),
                                "Total Savings": f"${group['saving'].sum():,.2f}",
                                "Avg Savings per Position": f"${group['saving'].mean():,.2f}"
                            })
                        
                        st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
                        
                        # Visualizations
                        st.subheader("📊 Relocation Analytics")
                        
                        # Savings by move
                        fig = px.bar(
                            results_df,
                            x=range(len(results_df)),
                            y="saving",
                            title=f"Savings by Individual Move - {scenario_name}",
                            labels={"x": "Move Number", "saving": "Savings ($)"},
                            color="saving",
                            color_continuous_scale="Viridis"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                    else:
                        st.warning("⚠️ No optimal relocations found within the specified constraints. Consider:")
                        st.write("• Expanding preferred locations/designations")
                        st.write("• Increasing location capacity limits")
                        st.write("• Allowing role changes")
                        st.write("• Adjusting target savings amount")

if __name__ == "__main__":
    main()
