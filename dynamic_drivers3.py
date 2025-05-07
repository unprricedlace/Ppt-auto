import streamlit as st
import pandas as pd
import numpy as np
import io
import base64

st.set_page_config(page_title="Financial Commentary Generator", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 28px;
    font-weight: bold;
    margin-bottom: 20px;
}
.section-header {
    font-size: 20px;
    font-weight: bold;
    margin-top: 20px;
    margin-bottom: 10px;
}
.commentary-box {
    background-color: #f0f2f6;
    border-radius: 5px;
    padding: 15px;
    margin-bottom: 20px;
}
.bullet-point {
    margin-left: 20px;
}
</style>
""", unsafe_allow_html=True)

# App header
st.markdown('<div class="main-header">Financial Commentary Generator</div>', unsafe_allow_html=True)

# Helper functions for commentary generation
def format_currency(value, precision=1):
    """Format numbers in millions with specified precision."""
    if pd.isna(value) or value == 0:
        return "$0"
    
    abs_value = abs(value)
    if abs_value >= 1000:
        return f"${abs_value/1000:.{precision}f}bn"
    else:
        return f"${abs_value:.{precision}f}mm"

def format_percentage(value, precision=0):
    """Format percentage with specified precision."""
    if pd.isna(value) or value == 0:
        return "0%"
    return f"{value:.{precision}f}%"

def get_variance_description(variance_pct):
    """Return descriptive term based on variance percentage."""
    abs_var = abs(variance_pct)
    if abs_var < 1:
        return "in line with"
    elif abs_var < 3:
        return "slightly" + (" favorable to" if variance_pct > 0 else " unfavorable to")
    elif abs_var < 5:
        return "" + ("favorable to" if variance_pct > 0 else "unfavorable to")
    else:
        return "significantly " + ("favorable to" if variance_pct > 0 else "unfavorable to")

def generate_commentary(data, period, at_par_threshold):
    """Generate commentary for a specific period (Current Month, YTD, Full Year)."""
    
    # Extract relevant columns based on period
    if period == "Current Month":
        actuals_col = "Actuals"
        budget_col = "Budget"
        variance_col = "O/U"
        variance_pct_col = "O/U(%)"
    elif period == "YTD":
        actuals_col = "Actuals.1"
        budget_col = "Budget.1"
        variance_col = "O/U.1"
        variance_pct_col = "O/U(%).1"
    else:  # Full Year
        actuals_col = "Outlook"
        budget_col = "Budget.2"
        variance_col = "O/U.2"
        variance_pct_col = "O/U(%).2"
    
    # Extract key metrics
    direct_expense = data.loc[data['Expense Details'] == 'Direct Expense', actuals_col].values[0]
    direct_expense_budget = data.loc[data['Expense Details'] == 'Direct Expense', budget_col].values[0]
    direct_expense_variance = data.loc[data['Expense Details'] == 'Direct Expense', variance_col].values[0]
    direct_expense_variance_pct = abs(data.loc[data['Expense Details'] == 'Direct Expense', variance_pct_col].values[0])
    
    compensation = data.loc[data['Expense Details'] == 'Compensation', actuals_col].values[0]
    compensation_budget = data.loc[data['Expense Details'] == 'Compensation', budget_col].values[0]
    compensation_variance = data.loc[data['Expense Details'] == 'Compensation', variance_col].values[0]
    compensation_variance_pct = abs(data.loc[data['Expense Details'] == 'Compensation', variance_pct_col].values[0])
    
    non_compensation = data.loc[data['Expense Details'] == 'Non-Compensation', actuals_col].values[0]
    non_compensation_budget = data.loc[data['Expense Details'] == 'Non-Compensation', budget_col].values[0]
    non_compensation_variance = data.loc[data['Expense Details'] == 'Non-Compensation', variance_col].values[0]
    non_compensation_variance_pct = abs(data.loc[data['Expense Details'] == 'Non-Compensation', variance_pct_col].values[0])
    
    # Get headcount information
    employees = data.loc[data['Expense Details'] == 'Employees', actuals_col].values[0]
    employees_budget = data.loc[data['Expense Details'] == 'Employees', budget_col].values[0]
    employees_variance = data.loc[data['Expense Details'] == 'Employees', variance_col].values[0]
    employees_variance_pct = data.loc[data['Expense Details'] == 'Employees', variance_pct_col].values[0] if not pd.isna(data.loc[data['Expense Details'] == 'Employees', variance_pct_col].values[0]) else 0
    
    contractors = data.loc[data['Expense Details'] == 'Contractors', actuals_col].values[0]
    contractors_budget = data.loc[data['Expense Details'] == 'Contractors', budget_col].values[0]
    contractors_variance = data.loc[data['Expense Details'] == 'Contractors', variance_col].values[0]
    contractors_variance_pct = data.loc[data['Expense Details'] == 'Contractors', variance_pct_col].values[0] if not pd.isna(data.loc[data['Expense Details'] == 'Contractors', variance_pct_col].values[0]) else 0
    
    # Check if direct expense is favorable or unfavorable to budget
    is_favorable = direct_expense_variance < 0
    variance_term = "favorable" if is_favorable else "unfavorable"
    
    # Main commentary
    commentary = f"{period}:\n"
    commentary += f"Direct expense of {format_currency(direct_expense)} is "
    
    # Format variance amount
    variance_amount = format_currency(abs(direct_expense_variance))
    variance_pct = format_percentage(direct_expense_variance_pct)
    
    if abs(direct_expense_variance_pct) <= at_par_threshold:
        commentary += f"at par with budget.\n"
    else:
        commentary += f"{variance_amount}/({variance_pct}) {variance_term} to budget:\n"
    
    # Analyze drivers for compensation
    comp_favorable_drivers = []
    comp_unfavorable_drivers = []
    
    # Sum salaries and benefits variances
    salaries_variance = data.loc[data['Expense Details'] == 'Salaries', variance_col].values[0] if not pd.isna(data.loc[data['Expense Details'] == 'Salaries', variance_col].values[0]) else 0
    benefits_variance = data.loc[data['Expense Details'] == 'Employee Benefits', variance_col].values[0] if not pd.isna(data.loc[data['Expense Details'] == 'Employee Benefits', variance_col].values[0]) else 0
    
    sal_ben_sum = salaries_variance + benefits_variance
    
    # Compare direction with employees row
    employees_variance = data.loc[data['Expense Details'] == 'Employees', variance_col].values[0] if not pd.isna(data.loc[data['Expense Details'] == 'Employees', variance_col].values[0]) else 0
    
    # Determine if headcount is driving compensation
    if abs(sal_ben_sum) > 0:
        sal_ben_amount = format_currency(abs(sal_ben_sum))
        
        # Checking if salaries+benefits and headcount are moving in same direction
        same_direction = (sal_ben_sum < 0 and employees_variance < 0) or (sal_ben_sum > 0 and employees_variance > 0)
        
        if sal_ben_sum < 0:  # Favorable
            if same_direction:
                comp_favorable_drivers.append(f"lower compensation ({sal_ben_amount}) due to lower headcount")
            else:
                comp_favorable_drivers.append(f"lower compensation ({sal_ben_amount}) due to decreased rates")
        else:  # Unfavorable
            if same_direction:
                comp_unfavorable_drivers.append(f"higher compensation ({sal_ben_amount}) due to higher headcount")
            else:
                comp_unfavorable_drivers.append(f"higher compensation ({sal_ben_amount}) due to increased rates")
    
    # Check severance impact
    severance_variance = data.loc[data['Expense Details'] == 'Severance', variance_col].values[0]
    if not pd.isna(severance_variance) and abs(severance_variance) > 0:
        severance_amount = format_currency(abs(severance_variance))
        if severance_variance < 0:
            comp_favorable_drivers.append(f"lower severance ({severance_amount})")
        else:
            comp_unfavorable_drivers.append(f"higher severance ({severance_amount})")
    
    # Check for IC payout
    incentives_variance = data.loc[data['Expense Details'] == 'Incentives Compensation', variance_col].values[0]
    if not pd.isna(incentives_variance) and abs(incentives_variance) > 0:
        ic_amount = format_currency(abs(incentives_variance))
        if incentives_variance < 0:
            comp_favorable_drivers.append(f"lower IC payout ({ic_amount})")
        else:
            comp_unfavorable_drivers.append(f"higher IC payout ({ic_amount})")
    
    # Analyze drivers for non-compensation
    noncomp_favorable_drivers = []
    noncomp_unfavorable_drivers = []
    noncomp_drivers_direction = []  # Will store 1 for favorable, -1 for unfavorable
    
    # Check Professional & Outside Services
    pos_variance = data.loc[data['Expense Details'] == 'Professional & Outside Services', variance_col].values[0]
    if not pd.isna(pos_variance) and abs(pos_variance) > 0:
        pos_amount = format_currency(abs(pos_variance))
        if pos_variance > 0:
            noncomp_unfavorable_drivers.append(f"higher Professional and Outside Services ({pos_amount})")
            noncomp_drivers_direction.append(-1)
        else:
            noncomp_favorable_drivers.append(f"lower Professional and Outside Services ({pos_amount})")
            noncomp_drivers_direction.append(1)
        
        # Check if contractors variance is significant (more than 40%)
        if abs(contractors_variance_pct) > 40:
            # Add the contractors comment right after Professional & Outside Services
            if (contractors_variance > 0 and pos_variance > 0) or (contractors_variance < 0 and pos_variance < 0):
                # If they're in the same direction, append to the same statement
                if pos_variance > 0:
                    noncomp_unfavorable_drivers[-1] += f" due to increase in contractors"
                else:
                    noncomp_favorable_drivers[-1] += f" due to decrease in contractors"
            else:
                # If they're in opposite directions, note as an offset
                if pos_variance > 0:
                    noncomp_unfavorable_drivers[-1] += f", partially offset by decrease in contractors"
                else:
                    noncomp_favorable_drivers[-1] += f", partially offset by increase in contractors"
    
    # Check Technology & Communications
    tech_variance = data.loc[data['Expense Details'] == 'Technology & Communications', variance_col].values[0]
    if not pd.isna(tech_variance) and abs(tech_variance) > 0:
        tech_amount = format_currency(abs(tech_variance))
        if tech_variance > 0:
            noncomp_unfavorable_drivers.append(f"higher Tech&Comms spend ({tech_amount})")
            noncomp_drivers_direction.append(-1)
        else:
            noncomp_favorable_drivers.append(f"lower Tech&Comms spend ({tech_amount})")
            noncomp_drivers_direction.append(1)
    
    # Check Travel & Entertainment
    te_variance = data.loc[data['Expense Details'] == 'Travel & Entertainment', variance_col].values[0]
    if not pd.isna(te_variance) and abs(te_variance) > 0:
        te_amount = format_currency(abs(te_variance))
        if te_variance > 0:
            noncomp_unfavorable_drivers.append(f"higher T&E ({te_amount})")
            noncomp_drivers_direction.append(-1)
        else:
            noncomp_favorable_drivers.append(f"lower T&E ({te_amount})")
            noncomp_drivers_direction.append(1)
    
    # Check All Other Expense
    other_variance = data.loc[data['Expense Details'] == 'All Other Expense', variance_col].values[0]
    if not pd.isna(other_variance) and abs(other_variance) > 0:
        other_amount = format_currency(abs(other_variance))
        if other_variance > 0:
            noncomp_unfavorable_drivers.append(f"higher Other Expenses ({other_amount})")
            noncomp_drivers_direction.append(-1)
        else:
            noncomp_favorable_drivers.append(f"lower Other Expenses ({other_amount})")
            noncomp_drivers_direction.append(1)
    
    # Generate bullet point for compensation
    if abs(compensation_variance_pct) <= at_par_threshold:
        comp_bullet = f"• Compensation: at par with budget"
    else:
        comp_variance_term = "favorable" if compensation_variance < 0 else "unfavorable"
        comp_amount = format_currency(abs(compensation_variance))
        comp_pct = format_percentage(compensation_variance_pct)
        
        comp_bullet = f"• Compensation: ({comp_amount})/({comp_pct}) {comp_variance_term} to budget"
        
        # MODIFIED: First list drivers aligned with overall comp direction, then list offsetting factors
        aligned_drivers = comp_favorable_drivers if compensation_variance < 0 else comp_unfavorable_drivers
        offsetting_drivers = comp_unfavorable_drivers if compensation_variance < 0 else comp_favorable_drivers
        
        if len(aligned_drivers) > 0:
            comp_bullet += f" driven by {', '.join(aligned_drivers)}"
            
            if len(offsetting_drivers) > 0:
                comp_bullet += f", partly offset by {', '.join(offsetting_drivers)}"
        elif len(offsetting_drivers) > 0:
            # If no aligned drivers but there are offsetting drivers
            comp_bullet += f" with {', '.join(offsetting_drivers)}"
        
        comp_bullet += "."
    
    # Generate bullet point for non-compensation
    if abs(non_compensation_variance_pct) <= at_par_threshold:
        noncomp_bullet = f"• Non-comp: at par with budget"
    else:
        noncomp_variance_term = "favorable" if non_compensation_variance < 0 else "unfavorable"
        noncomp_amount = format_currency(abs(non_compensation_variance))
        noncomp_pct = format_percentage(non_compensation_variance_pct)
        
        noncomp_bullet = f"• Non-comp: {noncomp_amount}/({noncomp_pct}) {noncomp_variance_term} to budget"
        
        # Check if all drivers are in the same direction
        all_same_direction = (len(noncomp_drivers_direction) > 0 and 
                             all(direction == noncomp_drivers_direction[0] for direction in noncomp_drivers_direction))
        
        if len(noncomp_favorable_drivers) > 0 or len(noncomp_unfavorable_drivers) > 0:
            # If all entities are in the same direction, use "vendor" terminology if more than 2
            if all_same_direction and (len(noncomp_favorable_drivers) + len(noncomp_unfavorable_drivers)) > 2:
                noncomp_bullet += " driven mostly by vendor spend"
            else:
                # MODIFIED: First list drivers aligned with overall non-comp direction, then list offsetting factors
                aligned_drivers = noncomp_favorable_drivers if non_compensation_variance < 0 else noncomp_unfavorable_drivers
                offsetting_drivers = noncomp_unfavorable_drivers if non_compensation_variance < 0 else noncomp_favorable_drivers
                
                if len(aligned_drivers) > 0:
                    noncomp_bullet += f" driven mostly by {', '.join(aligned_drivers)}"
                    
                if len(offsetting_drivers) > 0:
                    noncomp_bullet += f", partly offset by {', '.join(offsetting_drivers)}"
        
        noncomp_bullet += "."
    
    # Add bullets to commentary if not at par
    if abs(direct_expense_variance_pct) > at_par_threshold:
        commentary += comp_bullet + "\n" + noncomp_bullet
    
    return commentary

def main():
    # Sidebar configuration
    st.sidebar.header("Configuration")
    at_par_threshold = st.sidebar.slider("At Par Threshold (%)", 0.0, 5.0, 1.0, 0.1,
                                        help="Variance percentage below which values are considered 'at par' with budget")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload Financial Data (Excel)", type=["xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            # Read Excel file
            df = pd.read_excel(uploaded_file)
            
            # Display the uploaded data
            st.markdown('<div class="section-header">Uploaded Financial Data</div>', unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True)
            
            # Generate commentary for all periods
            st.markdown('<div class="section-header">Generated Commentary</div>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Current Month**")
                current_month_commentary = generate_commentary(df, "Current Month", at_par_threshold)
                st.markdown(f'<div class="commentary-box">{current_month_commentary.replace("•", "<div class="bullet-point">•").replace("\n", "</div>")}</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown("**YTD**")
                ytd_commentary = generate_commentary(df, "YTD", at_par_threshold)
                st.markdown(f'<div class="commentary-box">{ytd_commentary.replace("•", "<div class="bullet-point">•").replace("\n", "</div>")}</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown("**Full Year**")
                full_year_commentary = generate_commentary(df, "Full Year", at_par_threshold)
                st.markdown(f'<div class="commentary-box">{full_year_commentary.replace("•", "<div class="bullet-point">•").replace("\n", "</div>")}</div>', unsafe_allow_html=True)
            
            # Export functionality
            st.markdown('<div class="section-header">Export Commentary</div>', unsafe_allow_html=True)
            
            export_text = f"""
            # Financial Commentary Report
            
            ## Current Month
            {current_month_commentary}
            
            ## YTD
            {ytd_commentary}
            
            ## Full Year
            {full_year_commentary}
            """
            
            # Create a download button for the commentary
            st.download_button(
                label="Download Commentary as Text",
                data=export_text,
                file_name="financial_commentary.txt",
                mime="text/plain"
            )
            
            # Generate a sample Excel template
            if st.button("Generate Sample Excel Template"):
                # Create sample dataframe that matches expected format
                sample_data = {
                    "Expense Details": ["Salaries", "Severance", "Employee Benefits", "Incentives Compensation", 
                                        "Compensation", "Occupancy", "Technology & Communications", 
                                        "Professional & Outside Services", "Travel & Entertainment", 
                                        "All Other Expense", "Non-Compensation", "Direct Expense",
                                        "Direct Exp w/o Severance", "Enterprise Technology", 
                                        "Global Real Estate", "Other Allocations", "OFAC Inter/Intra LOB",
                                        "Allocations", "Total Expense", "Employees", "Contractors", "Total Headcount"],
                    # Current Month
                    "Actuals": [9565, 0, 2340, 46, 11951, 3, 23, 531, 27, 33, 617, 12568, 12568, 736, 778, 0, 8, 1523, 14091, 1379, 58, 1437],
                    "Budget": [9491, 0, 2673, 0, 12164, 3, 32, 349, 31, 44, 459, 12624, 12624, 706, 780, 17, 6, 1509, 14132, 1402, 35, 1437],
                    "O/U": [74, 0, -334, 46, -213, 0, -9, 182, -4, -11, 158, -56, -56, 31, -2, -17, 2, 14, -41, -23, 23, 0],
                    "O/U(%)": [1, 0, -12, 0, -2, -2, -28, 52, -13, -26, 34, 0, 0, 4, 0, -97, 40, 1, 0, -2, 66, 0],
                    
                    # YTD
                    "Actuals.1": [18468, 0, 5122, 45, 23635, 4, 70, 985, 32, 30, 1122, 24757, 24757, 1431, 1574, 18, 17, 3040, 27797, 1379, 58, 1437],
                    "Budget.1": [18670, 0, 5843, 0, 24513, 6, 64, 698, 63, 88, 919, 25432, 25432, 1412, 1561, 34, 13, 3019, 28451, 1402, 35, 1437],
                    "O/U.1": [-202, 0, -722, 45, -878, -2, 6, 287, -31, -58, 203, -675, -675, 19, 14, -16, 4, 21, -654, -23, 23, 0],
                    "O/U(%).1": [-1, 0, -12, 0, -4, -28, 10, 41, -49, -66, 22, -3, -3, 1, 1, -47, 31, 1, -2, -2, 66, 0],
                    
                    # Full Year / Outlook
                    "Outlook": [115918, 0, 32797, 45, 148760, 36, 385, 6847, 378, 546, 7931, 156691, 156691, 8446, 8325, 192, 78, 17041, 173732, 1441, 55, 1496],
                    "Budget.2": [115910, 0, 32796, 0, 148706, 36, 385, 4189, 377, 532, 5519, 154225, 154225, 8426, 8484, 208, 75, 17193, 171418, 1441, 35, 1476],
                    "O/U.2": [8, 0, 1, 45, 54, 0, 0, 2398, 0, 14, 2412, 2466, 2466, 19, -159, -16, 4, -152, 2314, 0, 20, 20],
                    "O/U(%).2": [0, 0, 0, 0, 0, 0, 0, 57, 0, 3, 44, 2, 2, 0, -2, -8, 5, -1, 1, 0, 57, 1]
                }
                
                sample_df = pd.DataFrame(sample_data)
                
                # Convert dataframe to Excel
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    sample_df.to_excel(writer, index=False, sheet_name='Financial Data')
                
                # Create download button
                st.download_button(
                    label="Download Sample Excel Template",
                    data=output.getvalue(),
                    file_name="financial_data_template.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
        except Exception as e:
            st.error(f"Error processing the file: {str(e)}")
            st.info("Please make sure your Excel file follows the expected format. You can download a sample template for reference.")

if __name__ == "__main__":
    main()
