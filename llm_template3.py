import pandas as pd
import streamlit as st
from string import Template
import re
from io import BytesIO
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

def load_and_process_summary_data(file):
    """Load and preprocess the summary Excel data"""
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
        st.error(f"Error processing the summary Excel file: {str(e)}")
        return None, None

def load_and_process_detailed_data(file):
    """Load and preprocess the detailed Excel data focusing on key columns"""
    try:
        df = pd.read_excel(file)
        
        # Check for required columns based on the raw data format
        required_columns = ['Expense Details', 'CY CM Actuals', 'CY CM Budget']
        
        # Try to find columns that may have similar names or purposes
        column_mapping = {}
        for req_col in required_columns:
            found = False
            # Exact match
            if req_col in df.columns:
                column_mapping[req_col] = req_col
                found = True
            # Check for partial matches for flexibility
            if not found:
                for col in df.columns:
                    if req_col.lower() in col.lower():
                        column_mapping[req_col] = col
                        found = True
                        break
            
            if not found and req_col == 'Expense Details':
                # Try alternative column names for expense details
                for col in df.columns:
                    if any(term in col.lower() for term in ["expense", "category", "detail"]):
                        column_mapping[req_col] = col
                        found = True
                        break
        
        # Verify we have all required mappings
        missing_columns = [col for col in required_columns if col not in column_mapping]
        if missing_columns:
            st.error(f"The detailed Excel file is missing required columns: {', '.join(missing_columns)}")
            return None
        
        # Create a new dataframe with consistent column names
        result_df = pd.DataFrame()
        for req_col, actual_col in column_mapping.items():
            result_df[req_col] = df[actual_col]
        
        return result_df
    except Exception as e:
        st.error(f"Error processing the detailed Excel file: {str(e)}")
        return None

def infer_category_from_subcategory(subcategory):
    """Infer high-level category from subcategory text"""
    subcategory_lower = subcategory.lower()
    
    # Define common terms for each high-level category
    category_keywords = {
        'Compensation': ['salary', 'wage', 'bonus', 'benefit', 'pension', 'healthcare', 'payroll', 'fte', 'headcount', 
                         'employee', 'hr', 'compensation', 'severance', 'ic payout', 'incentive'],
        'Non-Compensation': ['travel', 'entertainment', 'vendor', 'professional', 'service', 'software', 'hardware', 
                             'technology', 'supplies', 'equipment', 'marketing', 'advertising', 'rent', 'lease', 
                             'utility', 'maintenance', 'repair', 'legal', 'training', 'conference', 'subscription'],
        'Allocations': ['allocation', 'charge', 'transfer', 'overhead', 'cross-charge', 'share', 'distributed'],
        'Headcount': ['headcount', 'fte', 'full time', 'part time', 'staff', 'personnel'],
        'Employees': ['employee', 'permanent', 'staff', 'full-time', 'part-time'],
        'Contractors': ['contractor', 'consultant', 'temporary', 'contingent', 'freelance', 'outsource']
    }
    
    for category, keywords in category_keywords.items():
        if any(keyword in subcategory_lower for keyword in keywords):
            return category
            
    # Default category inference
    if "service" in subcategory_lower or "vendor" in subcategory_lower:
        return "Non-Compensation"
    elif "headcount" in subcategory_lower or "personnel" in subcategory_lower:
        return "Compensation"
        
    # Default to Non-Compensation if we can't determine
    return "Non-Compensation"

def create_expense_pivot(detailed_df):
    """Create a pivot table from the detailed data"""
    try:
        # Create a pivot table with Expense Details as rows
        pivot = pd.pivot_table(
            detailed_df,
            values=['CY CM Actuals', 'CY CM Budget'],
            index=['Expense Details'],
            aggfunc=np.sum
        )
        
        # Calculate the variance and percentage
        pivot['Variance'] = pivot['CY CM Actuals'] - pivot['CY CM Budget']
        pivot['Variance_Pct'] = (pivot['Variance'] / pivot['CY CM Budget']) * 100
        
        # Reset index to make Expense Details a column
        pivot = pivot.reset_index()
        
        # Infer category for each subcategory
        pivot['Category'] = pivot['Expense Details'].apply(infer_category_from_subcategory)
        
        # Set multi-level index for better grouping
        pivot = pivot.set_index(['Category', 'Expense Details'])
        
        # Sort by Category and then by absolute variance within each category
        pivot = pivot.groupby(level=0, group_keys=False).apply(
            lambda x: x.sort_values(by='Variance', key=abs, ascending=False)
        )
        
        return pivot
    except Exception as e:
        st.error(f"Error creating pivot table: {str(e)}")
        return None

def get_top_drivers_by_category(pivot, limit_per_category=3):
    """Extract top drivers from each category in the pivot table"""
    if 'Category' not in pivot.index.names:
        # If no category in index, return top overall drivers
        return pivot.head(5), {}
    
    # Get categories from the first level of the multi-index
    categories = pivot.index.get_level_values(0).unique()
    
    # Prepare containers for results
    top_drivers_overall = pd.DataFrame()
    top_drivers_by_category = {}
    
    for category in categories:
        # Filter pivot for this category
        category_pivot = pivot.loc[category]
        
        # Sort by absolute variance
        category_pivot = category_pivot.sort_values(by='Variance', key=abs, ascending=False)
        
        # Get top drivers for this category
        top_category_drivers = category_pivot.head(limit_per_category)
        
        # Add to overall top drivers
        top_drivers_overall = pd.concat([top_drivers_overall, top_category_drivers.head(1)])
        
        # Store in category dictionary
        top_drivers_by_category[category] = top_category_drivers
    
    # Sort the overall top drivers by absolute variance
    top_drivers_overall = top_drivers_overall.sort_values(by='Variance', key=abs, ascending=False)
    
    return top_drivers_overall, top_drivers_by_category

def extract_metrics(df, current_month):
    """Extract key metrics from the summary dataframe"""
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

def extract_category_metrics(df, current_month):
    """Extract metrics for each high-level category from the summary dataframe"""
    category_metrics = {}
    
    # Define time periods and their column prefixes
    time_periods = {
        'month': f"Current Month ({current_month})",
        'ytd': "YTD '25",
        'full_year': "Full Year '25"
    }
    
    # Major categories to track
    categories = ['Compensation', 'Non-Compensation', 'Allocations', 'Headcount', 'Employees', 'Contractors']
    
    for category in categories:
        if category in df.index:
            category_metrics[category] = {}
            
            for period, prefix in time_periods.items():
                try:
                    # Extract metrics for this category and period
                    actual = df.loc[category, f"{prefix} Actuals"]
                    budget = df.loc[category, f"{prefix} Budget"]
                    variance = df.loc[category, f"{prefix} O/U"]
                    variance_pct = df.loc[category, f"{prefix} O/U(%)"] if f"{prefix} O/U(%)" in df.columns else None
                    
                    # Determine if favorable or unfavorable
                    if period == 'month':
                        direction = "under" if variance < 0 else "over"
                    else:
                        direction = "favorable" if variance < 0 else "unfavorable"
                    
                    # Format variance for display
                    if abs(variance) >= 1000 and period == 'full_year':
                        variance_formatted = f"{variance/1000:.1f}mm"
                    else:
                        variance_formatted = f"{variance:.0f}k"
                    
                    # Store the metrics
                    category_metrics[category][period] = {
                        'actual': actual,
                        'budget': budget,
                        'variance': variance,
                        'variance_formatted': variance_formatted,
                        'direction': direction
                    }
                    
                    # Add percentage for YTD if available
                    if variance_pct is not None:
                        category_metrics[category][period]['variance_pct'] = variance_pct
                        
                except Exception:
                    # Skip if data is missing
                    pass
    
    return category_metrics

def prepare_drivers_for_llm(top_drivers_overall, top_drivers_by_category, category_metrics, period):
    """Prepare detailed drivers for LLM processing, combining pivot data with category metrics"""
    drivers_data = []
    
    # Process overall top drivers
    for idx, row in top_drivers_overall.iterrows():
        if isinstance(idx, tuple):
            category, subcategory = idx
        else:
            subcategory = idx
            category = "Unknown"
        
        variance = row['Variance']
        
        # Format for display
        if abs(variance) >= 1000 and period == 'full_year':
            variance_formatted = f"{variance/1000:.1f}mm"
        else:
            variance_formatted = f"{variance:.0f}k"
            
        direction = "under budget" if variance < 0 else "over budget"
        
        # Add to drivers data
        drivers_data.append({
            'category': category,
            'subcategory': subcategory,
            'variance': variance,
            'variance_formatted': variance_formatted,
            'direction': direction
        })
    
    # Complement with category metrics where missing
    for category, metrics in category_metrics.items():
        if period in metrics and abs(metrics[period]['variance']) > 0:
            # Check if this category is already represented in drivers
            if not any(d['category'] == category for d in drivers_data):
                drivers_data.append({
                    'category': category,
                    'subcategory': f"{category} overall",
                    'variance': metrics[period]['variance'],
                    'variance_formatted': metrics[period]['variance_formatted'],
                    'direction': metrics[period]['direction']
                })
    
    return drivers_data

@st.cache_resource
def load_phi3_model():
    """Load the Phi-3-Mini model using HuggingFace Transformers."""
    try:
        with st.spinner("Loading Phi-3-Mini model... This may take a moment."):
            model_name = "microsoft/phi-3-mini-128k-instruct"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading Phi-3-Mini model: {str(e)}")
        return None, None

def get_specific_drivers_from_phi3(metrics, drivers_data, period, model, tokenizer):
    """Generate specific drivers from Phi-3-Mini using detailed subcategory data"""
    # Format period names for the prompt
    period_names = {
        'month': "the current month",
        'ytd': "year-to-date",
        'full_year': "full year forecast"
    }
    
    # Format drivers data for the prompt
    drivers_text = ""
    for driver in drivers_data:
        drivers_text += f"- {driver['subcategory']} ({driver['category']}): {driver['variance_formatted']} {driver['direction']}\n"
    
    # Create a more focused prompt with detailed data
    prompt = f"""<|system|>
You are a financial analyst assistant. You identify key drivers in financial data with specificity and precision.
<|user|>
Based on the following financial data for {period_names[period]}, identify the 3-5 most significant and specific drivers that
explain why Direct Expense is {metrics[period]['variance_direction']} budget by {metrics[period]['budget_variance']}.

Direct Expense: {metrics[period]['direct_expense']}
Variance: {metrics[period]['budget_variance']} ({metrics[period]['variance_direction']} budget)

Detailed subcategory drivers:
{drivers_text}

IMPORTANT: Be very specific with your drivers. For example, instead of just saying "higher Compensation", specify exact subcategories 
like "lower average FTE headcount (24)" or "higher Professional Services spending on cloud migration project".

Return ONLY a comma-separated list of specific drivers in this format:
"lower average FTE headcount (24), higher Professional Services spending on cloud migration, increased Vendor costs for new platform"

DO NOT include any explanation, additional numbers, or text beyond the specific drivers. Focus on the most significant subcategories that explain the overall variance.
<|assistant|>"""
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.2,  # Low temperature for consistency
                do_sample=True,
                top_p=0.95
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response (remove the prompt)
        response = response.split("<|assistant|>")[-1].strip()
        
        # Clean the response to ensure we only get the drivers
        if "," in response:
            drivers = [driver.strip() for driver in response.split(',')]
            return drivers[:5]  # Limit to top 5 drivers
        else:
            # If no commas found, try to parse as best as possible
            return [response.strip()]
    except Exception as e:
        st.error(f"Error generating drivers with Phi-3: {str(e)}")
        
        # Fallback with most significant drivers from the data
        fallback_drivers = []
        for i, driver in enumerate(drivers_data[:3]):
            direction = "lower" if driver['variance'] < 0 else "higher"
            fallback_drivers.append(f"{direction} {driver['subcategory']}")
        
        return fallback_drivers

def generate_commentary(metrics, drivers):
    """Generate commentary using templates with specific drivers"""
    
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

@st.cache_data(show_spinner=False)
def process_data_with_llm(summary_file_bytes, detailed_file_bytes, model, tokenizer):
    """Process data with LLM and cache the results"""
    
    # Load and process the summary data from bytes
    try:
        summary_df, current_month = load_and_process_summary_data(BytesIO(summary_file_bytes))
        
        if summary_df is None or current_month is None:
            return None, None, None, None, None, None, None
            
        # Extract metrics from summary data
        metrics = extract_metrics(summary_df, current_month)
        
        # Extract category metrics
        category_metrics = extract_category_metrics(summary_df, current_month)
        
        # Process detailed data if available
        detailed_pivot = None
        top_drivers_overall = None
        top_drivers_by_category = None
        
        if detailed_file_bytes:
            detailed_df = load_and_process_detailed_data(BytesIO(detailed_file_bytes))
            if detailed_df is not None:
                # Create pivot table with inferred categories
                detailed_pivot = create_expense_pivot(detailed_df)
                
                # Extract top drivers by category
                top_drivers_overall, top_drivers_by_category = get_top_drivers_by_category(detailed_pivot)
        
        # Process with LLM to get drivers
        drivers = {}
        
        # Process for each time period
        for period in ['month', 'ytd', 'full_year']:
            if period in metrics:
                # Prepare drivers data
                if detailed_pivot is not None and top_drivers_overall is not None:
                    drivers_data = prepare_drivers_for_llm(top_drivers_overall, top_drivers_by_category, category_metrics, period)
                    
                    # Get specific drivers from Phi-3
                    drivers[period] = get_specific_drivers_from_phi3(metrics, drivers_data, period, model, tokenizer)
                else:
                    # Fallback to high-level category analysis if no detailed data
                    drivers_data = []
                    for category, cat_metrics in category_metrics.items():
                        if period in cat_metrics:
                            drivers_data.append({
                                'category': category,
                                'subcategory': category,
                                'variance': cat_metrics[period]['variance'],
                                'variance_formatted': cat_metrics[period]['variance_formatted'],
                                'direction': cat_metrics[period]['direction']
                            })
                    
                    # Sort by absolute variance
                    drivers_data.sort(key=lambda x: abs(x['variance']), reverse=True)
                    
                    # Get specific drivers from Phi-3
                    drivers[period] = get_specific_drivers_from_phi3(metrics, drivers_data, period, model, tokenizer)
        
        # Generate commentary
        commentary = generate_commentary(metrics, drivers)
        
        return summary_df, current_month, metrics, category_metrics, drivers, commentary, detailed_pivot
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return None, None, None, None, None, None, None

def main():
    st.title("Executive Management Report Commentary Generator")
    st.subheader("Enhanced Subcategory Analysis with Phi-3-Mini")
    
    # Add model information
    st.info("This application uses Microsoft's Phi-3-Mini-128k model running locally via HuggingFace Transformers")
    
    # Load Phi-3 model
    with st.spinner("Initializing Phi-3-Mini model..."):
        model, tokenizer = load_phi3_model()
    
    if model is None or tokenizer is None:
        st.error("Failed to load Phi-3-Mini model. Please check your installation.")
        st.stop()
    
    # Add model status indicator
    st.success("✅ Phi-3-Mini model loaded successfully")
    
    # File upload section with tabs for required and optional files
    st.subheader("Upload Files")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Required: Summary Excel File**")
        summary_file = st.file_uploader("Upload summary Excel file", type=["xlsx", "xls"], key="summary")
        
    with col2:
        st.markdown("**Optional: Detailed Excel File**")
        st.markdown("*For more specific subcategory analysis*")
        detailed_file = st.file_uploader("Upload detailed Excel file", type=["xlsx", "xls"], key="detailed")
    
    if summary_file is not None:
        # Read files into memory
        summary_file_bytes = summary_file.getvalue()
        detailed_file_bytes = detailed_file.getvalue() if detailed_file is not None else None
        
        # Process data and cache results
        with st.spinner("Processing data and generating commentary..."):
            summary_df, current_month, metrics, category_metrics, drivers, commentary, detailed_pivot = process_data_with_llm(
                summary_file_bytes, detailed_file_bytes, model, tokenizer
            )
        
        if summary_df is not None and current_month is not None:
            # Success message based on which files were processed
            if detailed_file_bytes:
                st.success(f"✅ Analysis complete for {current_month} using both summary and detailed subcategory data")
            else:
                st.success(f"✅ Analysis complete for {current_month} using summary data only")
            
            # Display subcategory breakdown based on pivot table if available
            if detailed_pivot is not None:
                with st.expander("View Subcategory Analysis"):
                    st.subheader("Subcategory Breakdown")
                    
                    # Show the counts by category
                    categories = detailed_pivot.index.get_level_values(0).unique()
                    
                    # Create tabs for each category
                    category_tabs = st.tabs(categories)
                    
                    for i, category in enumerate(categories):
                        with category_tabs[i]:
                            # Display subcategories for this category
                            subcategories = detailed_pivot.loc[category].index.tolist()
                            st.write(f"**{len(subcategories)} subcategories found for {category}:**")
                            
                            # Display subcategory table
                            subcategory_df = detailed_pivot.loc[category].reset_index()
                            st.dataframe(
                                subcategory_df[['Expense Details', 'CY CM Actuals', 'CY CM Budget', 'Variance', 'Variance_Pct']],
                                use_container_width=True
                            )
            
            # Add toggle for data inspection
            with st.expander("View Extracted Data and Analysis"):
                st.subheader("Extracted Metrics")
                st.json(metrics)
                
                st.subheader("Category Metrics")
                st.json(category_metrics)
                
                st.subheader("Phi-3-Generated Drivers")
                st.json(drivers)
                
                # Show detailed pivot if available
                if detailed_pivot is not None:
                    st.subheader("Top Expense Detail Variances")
                    if isinstance(detailed_pivot.index, pd.MultiIndex):
                        # Display with categories
                        st.dataframe(
                            detailed_pivot.head(15)[['CY CM Actuals', 'CY CM Budget', 'Variance', 'Variance_Pct']],
                            use_container_width=True
                        )
                    else:
                        # Display without categories
                        st.dataframe(
                            detailed_pivot.head(15)[['CY CM Actuals', 'CY CM Budget', 'Variance', 'Variance_Pct']],
                            use_container_width=True
                        )
            
            # Commentary section
            st.subheader("Generated Commentary with Subcategory Details")
            
            # Create a container with background for better readability
            commentary_container = st.container(border=True)
            with commentary_container:
                st.markdown(f"```\n{commentary}\n```")
            
            # Add download button
            st.download_button(
                label="Download Commentary",
                data=commentary,
                file_name="commentary.txt",
                mime="text/plain"
            )
            
            # Compare with original example
            with st.expander("Compare with example commentary"):
                example_commentary = """Month:
Direct Expense of $12.6mm is $(56k) under budget mostly
driven by Compensation based on lower headcount, partly
offset by higher Vendor spend.

YTD:
Direct Expense of $24.8mm is ($675k)/(3%) favorable to
budget driven mostly by lower average FTE headcount (26),
partly offset by higher Professional Services & IC payout.

Full Year:
Direct Expense of $156.7mm is $2.5mm unfavorable driven by
higher Professional Services due to approved additional
resources & IC Payout."""
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Generated Commentary")
                    st.text(commentary)
                
                with col2:
                    st.subheader("Example Commentary")
                    st.text(example_commentary)
    else:
        # Display instructions when no files are uploaded
        st.info("Please upload the summary Excel file to generate commentary. For more detailed subcategory analysis, also upload the detailed Excel file.")
        
        with st.expander("View example file formats"):
            st.markdown("### Summary File Format (Required)")
            st.markdown("""
            The summary file should have categories as rows, with columns for:
            - Current Month (Month 'Year) Actuals
            - Current Month (Month 'Year) Budget
            - Current Month (Month 'Year) O/U
            - YTD 'Year Actuals
            - YTD 'Year Budget
            - YTD 'Year O/U
            - Full Year 'Year Actuals
            - Full Year 'Year Budget
            - Full Year 'Year O/U
            """)
            
            st.markdown("### Detailed File Format (Optional)")
            st.markdown("""
            The detailed file should have individual expense items with columns including:
            - Expense Details (subcategory)
            - CY CM Actuals (Current Year Current Month Actuals)
            - CY CM Budget (Current Year Current Month Budget)
            
            Adding this file will enable subcategory-level analysis for more specific commentary.
            """)

if __name__ == "__main__":
    main()
