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

# Enhanced optimization function with user-specified target locations and positions per employee group
def suggest_optimal_relocations_enhanced(current_selections, target_savings, employee_data, constraints):
    current_cost = calculate_cost_savings(current_selections, employee_data)
    target_cost = current_cost - target_savings
    
    if target_cost <= 0:
        return None, "Target savings too high. Cannot reduce costs to zero or negative."
    
    # Convert selections to a list of individual positions with their details including targets
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
                "target_designation": selection["target_designation"],
                "target_location": selection["target_location"]
            })
            position_id += 1
    
    # Get all possible moves for each position based on their specific targets
    all_moves = []
    for pos in positions:
        new_designation = pos["target_designation"]
        new_location = pos["target_location"]
        
        # Skip if no change
        if new_designation == pos["designation"] and new_location == pos["location"]:
            continue
        
        # Check constraints
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
    
    # Enhanced selection algorithm
    selected_moves = smart_move_selection(all_moves, target_savings, constraints)
    
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

def smart_move_selection(all_moves, target_savings, constraints):
    """Enhanced move selection algorithm"""
    
    # Sort moves by a composite score considering efficiency and absolute savings
    def calculate_score(move):
        efficiency_weight = constraints.get("efficiency_weight", 0.6)
        savings_weight = constraints.get("savings_weight", 0.4)
        
        # Normalize efficiency and absolute savings
        max_efficiency = max(m["efficiency"] for m in all_moves) if all_moves else 1
        max_saving = max(m["saving"] for m in all_moves) if all_moves else 1
        
        normalized_efficiency = move["efficiency"] / max_efficiency if max_efficiency > 0 else 0
        normalized_saving = move["saving"] / max_saving if max_saving > 0 else 0
        
        return efficiency_weight * normalized_efficiency + savings_weight * normalized_saving
    
    # Sort by composite score
    all_moves.sort(key=calculate_score, reverse=True)
    
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
    st.write("Advanced tool for optimizing employee costs with user-specified target locations and positions.")
    
    # Get employee data
    employee_data = get_employee_data()
    st.session_state["employee_data"] = employee_data
    
    # Sidebar for controls
    st.sidebar.header("🔧 Configuration")
    
    # Initialize session state
    if "current_selections" not in st.session_state:
        st.session_state.current_selections = []
    
    if "optimization_results" not in st.session_state:
        st.session_state.optimization_results = None
    
    if "optimization_message" not in st.session_state:
        st.session_state.optimization_message = ""
    
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
    
    # Algorithm tuning
    st.sidebar.subheader("⚙️ Algorithm Parameters")
    efficiency_weight = st.sidebar.slider("Efficiency Weight", 0.0, 1.0, 0.6, 0.1)
    savings_weight = 1.0 - efficiency_weight
    st.sidebar.write(f"Savings Weight: {savings_weight:.1f}")
    
    constraints = {
        "location_limits": location_limits,
        "allow_role_changes": allow_role_changes,
        "protect_senior_roles": protect_senior_roles,
        "efficiency_weight": efficiency_weight,
        "savings_weight": savings_weight
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
    
    # Main content area
    st.subheader("👥 Employee Selection & Target Destinations")
    
    # Add employee selection with target destinations
    with st.form("employee_selection_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Current Employee Details:**")
            designation = st.selectbox("Current Designation", DESIGNATIONS)
            location = st.selectbox("Current Location", LOCATIONS)
            positions = st.number_input("Number of Positions", min_value=1, max_value=100, value=1)
        
        with col2:
            st.write("**Target Destination:**")
            target_designation = st.selectbox("Target Designation", DESIGNATIONS, key="target_des")
            target_location = st.selectbox("Target Location", LOCATIONS, key="target_loc")
            
            # Show potential savings preview
            if designation and location and target_designation and target_location:
                current_cost = employee_data[
                    (employee_data["Designation"] == designation) & 
                    (employee_data["Location"] == location)
                ]["Annualized_Rate"].values[0]
                
                target_cost = employee_data[
                    (employee_data["Designation"] == target_designation) & 
                    (employee_data["Location"] == target_location)
                ]["Annualized_Rate"].values[0]
                
                savings_per_position = current_cost - target_cost
                st.metric("Savings per Position", f"${savings_per_position:,.2f}", 
                         f"{(savings_per_position/current_cost)*100:.1f}%" if current_cost > 0 else "0%")
        
        submitted = st.form_submit_button("Add Employee Group")
        
        if submitted:
            st.session_state.current_selections.append({
                "designation": designation,
                "location": location,
                "positions": positions,
                "target_designation": target_designation,
                "target_location": target_location
            })
            st.session_state.optimization_results = None
        
    # Display current selections
    if not st.session_state.current_selections:
        st.info("No employee groups selected yet.")
    else:
        # Display as table with enhanced formatting including target destinations
        selections_with_costs = []
        for i, selection in enumerate(st.session_state.current_selections):
            current_rate = employee_data[
                (employee_data["Designation"] == selection["designation"]) & 
                (employee_data["Location"] == selection["location"])
            ]["Annualized_Rate"].values[0]
            
            target_rate = employee_data[
                (employee_data["Designation"] == selection["target_designation"]) & 
                (employee_data["Location"] == selection["target_location"])
            ]["Annualized_Rate"].values[0]
            
            current_total_cost = current_rate * selection["positions"]
            target_total_cost = target_rate * selection["positions"]
            savings = current_total_cost - target_total_cost
            
            selections_with_costs.append({
                "ID": i + 1,
                "Current": f"{selection['designation']} - {selection['location']}",
                "Target": f"{selection['target_designation']} - {selection['target_location']}",
                "Positions": selection["positions"],
                "Current Cost": f"${current_total_cost:,.2f}",
                "Target Cost": f"${target_total_cost:,.2f}",
                "Potential Savings": f"${savings:,.2f}"
            })
        
        st.dataframe(pd.DataFrame(selections_with_costs), use_container_width=True)
        
        # Calculate and display totals
        total_current_cost = sum([
            employee_data[
                (employee_data["Designation"] == s["designation"]) & 
                (employee_data["Location"] == s["location"])
            ]["Annualized_Rate"].values[0] * s["positions"]
            for s in st.session_state.current_selections
        ])
        
        total_target_cost = sum([
            employee_data[
                (employee_data["Designation"] == s["target_designation"]) & 
                (employee_data["Location"] == s["target_location"])
            ]["Annualized_Rate"].values[0] * s["positions"]
            for s in st.session_state.current_selections
        ])
        
        total_potential_savings = total_current_cost - total_target_cost
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("💰 Current Total Cost", f"${total_current_cost:,.2f}")
        col2.metric("🎯 Target Total Cost", f"${total_target_cost:,.2f}")
        col3.metric("💵 Max Potential Savings", f"${total_potential_savings:,.2f}")
        col4.metric("👥 Total Positions", sum(s["positions"] for s in st.session_state.current_selections))
    
    with col2:
        st.subheader("🎯 Target Locations & Positions")
        
        # Add target location and position selection
        with st.form("target_selection_form"):
            st.write("**Add Target Destinations:**")
            target_designation = st.selectbox("Target Designation", DESIGNATIONS, key="target_des")
            target_location = st.selectbox("Target Location", LOCATIONS, key="target_loc")
            
            target_submitted = st.form_submit_button("Add Target Option")
            
            if target_submitted:
                # Check if this combination already exists
                exists = any(
                    t["designation"] == target_designation and t["location"] == target_location 
                    for t in st.session_state.target_locations_positions
                )
                
                if not exists:
                    st.session_state.target_locations_positions.append({
                        "designation": target_designation,
                        "location": target_location
                    })
                    st.session_state.optimization_results = None
                else:
                    st.warning("This target combination already exists!")
        
        # Display target selections
        if not st.session_state.target_locations_positions:
            st.info("No target destinations selected yet.")
        else:
            target_display = []
            for i, target in enumerate(st.session_state.target_locations_positions):
                # Get cost for this target combination
                rate = employee_data[
                    (employee_data["Designation"] == target["designation"]) & 
                    (employee_data["Location"] == target["location"])
                ]["Annualized_Rate"].values[0]
                
                target_display.append({
                    "ID": i + 1,
                    "Target Designation": target["designation"],
                    "Target Location": target["location"],
                    "Cost per Position": f"${rate:,.2f}"
                })
            
            st.dataframe(pd.DataFrame(target_display), use_container_width=True)
            
            st.metric("🎯 Total Target Options", len(st.session_state.target_locations_positions))
    
    with col2:
        st.subheader("🎯 Target Locations & Positions")
        
        # Add target location and position selection
        with st.form("target_selection_form"):
            st.write("**Add Target Destinations:**")
            target_designation = st.selectbox("Target Designation", DESIGNATIONS, key="target_des")
            target_location = st.selectbox("Target Location", LOCATIONS, key="target_loc")
            
            target_submitted = st.form_submit_button("Add Target Option")
            
            if target_submitted:
                # Check if this combination already exists
                exists = any(
                    t["designation"] == target_designation and t["location"] == target_location 
                    for t in st.session_state.target_locations_positions
                )
                
                if not exists:
                    st.session_state.target_locations_positions.append({
                        "designation": target_designation,
                        "location": target_location
                    })
                    st.session_state.optimization_results = None
                else:
                    st.warning("This target combination already exists!")
        
        # Display target selections
        if not st.session_state.target_locations_positions:
            st.info("No target destinations selected yet.")
        else:
            target_display = []
            for i, target in enumerate(st.session_state.target_locations_positions):
                # Get cost for this target combination
                rate = employee_data[
                    (employee_data["Designation"] == target["designation"]) & 
                    (employee_data["Location"] == target["location"])
                ]["Annualized_Rate"].values[0]
                
                target_display.append({
                    "ID": i + 1,
                    "Target Designation": target["designation"],
                    "Target Location": target["location"],
                    "Cost per Position": f"${rate:,.2f}"
                })
            
            st.dataframe(pd.DataFrame(target_display), use_container_width=True)
            
            st.metric("🎯 Total Target Options", len(st.session_state.target_locations_positions))
    
    # Reset button
    if st.button("🔄 Reset All Selections"):
        st.session_state.current_selections = []
        st.session_state.optimization_results = None
        st.session_state.optimization_message = ""
    
    # Optimization section
    if st.session_state.current_selections:
        st.markdown("---")
        st.subheader("🚀 Cost Optimization")
        
        total_current_cost = calculate_cost_savings(st.session_state.current_selections, employee_data)
        
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
            
            optimize_submitted = st.form_submit_button("🚀 Find Optimal Relocations")
            
            if optimize_submitted:
                with st.spinner("Analyzing optimal relocations..."):
                    st.session_state.optimization_results, st.session_state.optimization_message = suggest_optimal_relocations_enhanced(
                        st.session_state.current_selections, 
                        target_savings, 
                        employee_data,
                        constraints
                    )
        
        # Display optimization results
        if st.session_state.optimization_results is not None:
            st.markdown("---")
            st.subheader("📈 Optimization Results")
            
            # Display message
            st.markdown(st.session_state.optimization_message)
            
            if isinstance(st.session_state.optimization_results, list) and st.session_state.optimization_results:
                results_df = pd.DataFrame(st.session_state.optimization_results)
                
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
                    title="Savings by Individual Move",
                    labels={"x": "Move Number", "saving": "Savings ($)"},
                    color="saving",
                    color_continuous_scale="Viridis"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Location flow diagram
                st.subheader("🌍 Location Flow Analysis")
                
                # Before vs After by location
                location_before = defaultdict(int)
                location_after = defaultdict(int)
                
                # Count original positions
                for selection in st.session_state.current_selections:
                    location_before[selection["location"]] += selection["positions"]
                    location_after[selection["location"]] += selection["positions"]
                
                # Apply moves
                for move in st.session_state.optimization_results:
                    location_after[move["original_location"]] -= 1
                    location_after[move["new_location"]] += 1
                
                location_comparison = []
                for location in LOCATIONS:
                    location_comparison.extend([
                        {"Location": location, "Count": location_before[location], "Status": "Before"},
                        {"Location": location, "Count": location_after[location], "Status": "After"}
                    ])
                
                fig = px.bar(
                    pd.DataFrame(location_comparison),
                    x="Location",
                    y="Count",
                    color="Status",
                    barmode="group",
                    title="Position Distribution: Before vs After Relocation"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Cost efficiency analysis
                fig = px.scatter(
                    results_df,
                    x="original_cost",
                    y="saving",
                    size="efficiency",
                    color="new_location",
                    title="Cost Efficiency Analysis",
                    labels={"original_cost": "Original Cost ($)", "saving": "Savings ($)"},
                    hover_data=["original_designation", "new_designation"]
                )
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.warning("⚠️ No optimal relocations found within the specified constraints. Consider:")
                st.write("• Adjusting target savings amount")
                st.write("• Increasing location capacity limits")
                st.write("• Allowing role changes")
                st.write("• Reviewing your target destinations for better savings potential")
    
    else:
        st.info("👆 Please add employee groups with their target destinations to enable optimization.")

if __name__ == "__main__":
    main()
