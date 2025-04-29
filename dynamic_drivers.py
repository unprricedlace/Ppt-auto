import pandas as pd
import streamlit as st
from string import Template
import numpy as np
import re

def load_and_process_data(file):
    """Load and preprocess the Excel data"""
    try:
        df = pd.read_excel(file, index_col=0)
        # Find current month from column names
        month_pattern = r"Current Month \((.*?)\)"
        current_month = None
        for col in df.columns:
            match = re.search(month_pattern, col)
            if match:
                current_month = match.group(1)
                break
        
        return df, current_month
    except Exception as e:
        st.error(f"Error processing the Excel file: {str(e)}")
        return None, None

def extract_metrics(df, current_month):
    """Extract key metrics from the dataframe"""
    metrics = {}
    
    # Define time periods and their column prefixes
    time_periods = {
        'month': f"Current Month ({current_month})",
        'ytd': "YTD '25",
        'full_year': "Full Year '25"
    }
    
    for period, prefix in time_periods.items():
        try:
            # Extract core metrics
            direct_expense_actual = df.loc['Direct Expense', f"{prefix} Actuals"]
            direct_expense_budget = df.loc['Direct Expense', f"{prefix} Budget"]
            variance = df.loc['Direct Expense', f"{prefix} O/U"]
            variance_pct = df.loc['Direct Expense', f"{prefix} O/U(%)"] if f"{prefix} O/U(%)" in df.columns else None
            
            # Format for display
            if period == 'full_year':
                # Full year is typically in millions
                direct_expense_formatted = f"{direct_expense_actual/1000:.1f}mm"
                variance_formatted = f"{abs(variance)/1000:.1f}"  # Converting to millions
            else:
                # Month and YTD are typically in thousands
                direct_expense_formatted = f"{direct_expense_actual/1000:.1f}mm"
                variance_formatted = f"{abs(variance):.0f}k"
            
            # Determine if favorable or unfavorable
            if period == 'month':
                direction = "under" if variance < 0 else "over"
            else:
                direction = "favorable" if variance < 0 else "unfavorable"
            
            # Store metrics
            metrics[period] = {
                'direct_expense': direct_expense_formatted,
                'direct_expense_raw': direct_expense_actual,
                'budget_variance': variance_formatted,
                'budget_variance_raw': variance,
                'variance_direction': direction
            }
            
            # Add percentage for YTD
            if period == 'ytd' and variance_pct is not None:
                metrics[period]['budget_variance_pct'] = f"{abs(variance_pct):.0f}%"
            
        except Exception as e:
            st.warning(f"Could not extract all metrics for {period}: {str(e)}")
    
    return metrics

def calculate_drivers(df, current_month, significance_threshold=0.2):
    """
    Algorithmically determine drivers by analyzing the contribution 
    of each category to the overall variance
    """
    drivers = {}
    
    # Define time periods and their column prefixes
    time_periods = {
        'month': f"Current Month ({current_month})",
        'ytd': "YTD '25",
        'full_year': "Full Year '25"
    }
    
    # Categories to analyze for drivers
    categories = ['Compensation', 'Non-Compensation', 'Allocations', 'Headcount', 'Employees', 'Contractors']
    
    for period, prefix in time_periods.items():
        try:
            # Get total variance for Direct Expense
            total_variance = df.loc['Direct Expense', f"{prefix} O/U"]
            if total_variance == 0:
                drivers[period] = ["no significant variances"]
                continue
            
            # Analyze each category's contribution
            category_impacts = []
            
            for category in categories:
                if category in df.index:
                    try:
                        variance = df.loc[category, f"{prefix} O/U"]
                        # Skip if variance is zero or null
                        if pd.isna(variance) or variance == 0:
                            continue
                        
                        # Calculate contribution percentage to total variance
                        contribution = (variance / total_variance) * 100
                        
                        # Only consider significant contributors
                        if abs(contribution) >= significance_threshold * 100:
                            direction = "lower" if variance < 0 else "higher"
                            
                            # Special case for headcount
                            if category in ['Headcount', 'Employees', 'Contractors']:
                                impact_text = f"{direction} {category.lower()} ({abs(int(variance))})"
                            else:
                                impact_text = f"{direction} {category}"
                            
                            category_impacts.append({
                                'category': category,
                                'text': impact_text,
                                'contribution': abs(contribution),
                                'variance': variance
                            })
                    except Exception as e:
                        # Skip categories with issues
                        continue
            
            # Sort by contribution (highest impact first)
            category_impacts.sort(key=lambda x: x['contribution'], reverse=True)
            
            # Select top drivers (limited to top 3)
            top_drivers = [impact['text'] for impact in category_impacts[:3]]
            
            # Group related drivers for more coherent commentary
            final_drivers = combine_related_drivers(top_drivers)
            
            # Handle special case: no significant drivers found
            if not final_drivers:
                final_drivers = ["various small variances across categories"]
            
            drivers[period] = final_drivers
            
        except Exception as e:
            st.warning(f"Could not calculate drivers for {period}: {str(e)}")
            drivers[period] = ["data analysis issue"]
    
    return drivers

def combine_related_drivers(drivers):
    """Combine related drivers for more coherent commentary"""
    combined = []
    
    # Example of combining related drivers
    comp_related = [d for d in drivers if 'Compensation' in d or 'headcount' in d or 'employees' in d]
    if len(comp_related) > 1:
        combined.append("compensation-related factors")
        drivers = [d for d in drivers if d not in comp_related]
    
    # Add remaining drivers
    combined.extend(drivers)
    
    return combined[:3]  # Limit to top 3 drivers

def generate_commentary(metrics, drivers):
    """Generate commentary using templates"""
    
    # Define templates for each time period
    templates = {
        'month': Template("Month:\nDirect Expense of $${direct_expense} is $$(${budget_variance}) ${variance_direction} budget mostly driven by ${drivers}."),
        
        'ytd': Template("YTD:\nDirect Expense of $${direct_expense} is $$(${budget_variance})/(${budget_variance_pct}) ${variance_direction} to budget driven mostly by ${drivers}."),
        
        'full_year': Template("Full Year:\nDirect Expense of $${direct_expense} is $$${budget_variance}mm ${variance_direction} driven by ${drivers}.")
    }
    
    commentary = {}
    
    # Generate commentary for each time period
    for period in ['month', 'ytd', 'full_year']:
        if period not in metrics or period not in drivers:
            commentary[period] = f"{period.upper()}: Data not available for commentary generation."
            continue
            
        # Format drivers as text
        if len(drivers[period]) > 1:
            # Use Oxford comma for clarity with multiple drivers
            if len(drivers[period]) > 2:
                drivers_text = ", ".join(drivers[period][:-1]) + ", and " + drivers[period][-1]
            else:
                drivers_text = " and ".join(drivers[period])
        else:
            drivers_text = drivers[period][0]
        
        # Generate commentary using template
        try:
            template_data = {
                'direct_expense': metrics[period]['direct_expense'],
                'budget_variance': metrics[period]['budget_variance'],
                'variance_direction': metrics[period]['variance_direction'],
                'drivers': drivers_text
            }
            
            # Add percentage for YTD if available
            if period == 'ytd' and 'budget_variance_pct' in metrics[period]:
                template_data['budget_variance_pct'] = metrics[period]['budget_variance_pct']
            
            commentary[period] = templates[period].substitute(template_data)
        except KeyError as e:
            st.warning(f"Missing data for {period} commentary: {str(e)}")
            commentary[period] = f"{period.upper()}: Insufficient data for commentary generation."
    
    # Combine all sections
    full_commentary = f"{commentary.get('month', '')}\n\n{commentary.get('ytd', '')}\n\n{commentary.get('full_year', '')}"
    
    return full_commentary

def main():
    st.title("Executive Management Report Commentary Generator")
    st.subheader("Algorithmic Approach")
    
    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])
    
    if uploaded_file is not None:
        # Load and process the data
        df, current_month = load_and_process_data(uploaded_file)
        
        if df is not None and current_month is not None:
            # Extract metrics
            metrics = extract_metrics(df, current_month)
            
            # Calculate drivers
            significance = st.slider("Driver Significance Threshold (%)", 
                                     min_value=5, max_value=50, value=20, step=5) / 100
            drivers = calculate_drivers(df, current_month, significance_threshold=significance)
            
            # Display raw data for verification
            if st.checkbox("Show extracted data"):
                st.subheader("Extracted Metrics")
                st.json(metrics)
                
                st.subheader("Calculated Drivers")
                st.json(drivers)
            
            # Generate commentary
            commentary = generate_commentary(metrics, drivers)
            
            st.subheader("Generated Commentary")
            st.text_area("Commentary", commentary, height=300)
            
            # Add download button
            st.download_button(
                label="Download Commentary",
                data=commentary,
                file_name="commentary.txt",
                mime="text/plain"
            )

if __name__ == "__main__":
    main()
