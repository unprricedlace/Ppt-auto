import pandas as pd
import streamlit as st
from string import Template
import re
from io import BytesIO
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_and_process_data(file):
    """Load and preprocess the Excel data"""
    try:
        # First try to read as summary file
        df_summary = pd.read_excel(file, index_col=0)
        
        # Find current month from column names
        month_pattern = r"Current Month \((.*?)\)"
        current_month = None
        for col in df_summary.columns:
            match = re.search(month_pattern, col)
            if match:
                current_month = match.group(1)
                break
        
        return df_summary, current_month, None
    
    except Exception as e:
        st.warning(f"Could not read as summary file, trying raw format: {str(e)}")
        try:
            # Try to read as raw file
            df_raw = pd.read_excel(file)
            
            # Check if it has the expected raw columns
            required_columns = ['Expense Details', 'Current Month', 'CY CM Actuals', 
                              'CY CM Budget', 'Full Year PY Actuals', 'Full Year CY Outlook',
                              'YTD PY Actuals', 'YTD CY Outlook']
            
            if all(col in df_raw.columns for col in required_columns):
                # Create pivot table from raw data
                pivot = pd.pivot_table(df_raw, 
                                     index='Expense Details',
                                     values=['Current Month', 'CY CM Actuals', 'CY CM Budget',
                                            'Full Year PY Actuals', 'Full Year CY Outlook',
                                            'YTD PY Actuals', 'YTD CY Outlook'],
                                     aggfunc='sum')
                
                # Try to get current month from data (might need adjustment based on actual data)
                current_month = "February '2025"  # Placeholder - adjust as needed
                
                return None, current_month, pivot
            else:
                st.error("Uploaded file doesn't match expected raw or summary format")
                return None, None, None
                
        except Exception as e:
            st.error(f"Error processing the Excel file: {str(e)}")
            return None, None, None

def extract_metrics(df_summary, df_raw_pivot, current_month):
    """Extract key metrics from either summary or raw data"""
    metrics = {}
    
    if df_summary is not None:
        # Process summary data
        time_periods = {
            'month': f"Current Month ({current_month})",
            'ytd': "YTD '25",
            'full_year': "Full Year '25"
        }
        
        for period, prefix in time_periods.items():
            try:
                # Extract core metrics from summary
                direct_expense_actual = df_summary.loc['Direct Expense', f"{prefix} Actuals"]
                direct_expense_budget = df_summary.loc['Direct Expense', f"{prefix} Budget"]
                variance = df_summary.loc['Direct Expense', f"{prefix} O/U"]
                variance_pct = df_summary.loc['Direct Expense', f"{prefix} O/U(%)"] if f"{prefix} O/U(%)" in df_summary.columns else None
                
                # Format for display
                if period == 'full_year':
                    direct_expense_formatted = f"{direct_expense_actual/1000:.1f}mm"
                    variance_formatted = f"{abs(variance)/1000:.1f}mm"
                else:
                    direct_expense_formatted = f"{direct_expense_actual/1000:.1f}mm"
                    variance_formatted = f"{abs(variance):.0f}k"
                
                direction = "favorable" if variance < 0 else "unfavorable"
                
                metrics[period] = {
                    'direct_expense': direct_expense_formatted,
                    'direct_expense_raw': direct_expense_actual,
                    'budget_variance': variance_formatted,
                    'budget_variance_raw': variance,
                    'variance_direction': direction,
                    'data_source': 'summary'
                }
                
                if period == 'ytd' and variance_pct is not None:
                    metrics[period]['budget_variance_pct'] = f"{abs(variance_pct):.0f}%"
                    
            except Exception as e:
                st.warning(f"Could not extract all metrics for {period} from summary: {str(e)}")
    
    elif df_raw_pivot is not None:
        # Process raw data pivot
        time_periods = {
            'month': ('CY CM Actuals', 'CY CM Budget'),
            'ytd': ('YTD CY Outlook', 'YTD PY Actuals'),  # Adjust based on actual mapping
            'full_year': ('Full Year CY Outlook', 'Full Year PY Actuals')  # Adjust based on actual mapping
        }
        
        for period, (actual_col, budget_col) in time_periods.items():
            try:
                # Calculate direct expense as sum of all expense details
                direct_expense_actual = df_raw_pivot[actual_col].sum()
                direct_expense_budget = df_raw_pivot[budget_col].sum()
                variance = direct_expense_actual - direct_expense_budget
                
                # Format for display
                if period == 'full_year':
                    direct_expense_formatted = f"{direct_expense_actual/1000:.1f}mm"
                    variance_formatted = f"{abs(variance)/1000:.1f}mm"
                else:
                    direct_expense_formatted = f"{direct_expense_actual/1000:.1f}mm"
                    variance_formatted = f"{abs(variance):.0f}k"
                
                direction = "favorable" if variance < 0 else "unfavorable"
                
                metrics[period] = {
                    'direct_expense': direct_expense_formatted,
                    'direct_expense_raw': direct_expense_actual,
                    'budget_variance': variance_formatted,
                    'budget_variance_raw': variance,
                    'variance_direction': direction,
                    'data_source': 'raw'
                }
                
                # For YTD, calculate percentage if possible
                if period == 'ytd' and direct_expense_budget != 0:
                    variance_pct = (abs(variance) / direct_expense_budget) * 100
                    metrics[period]['budget_variance_pct'] = f"{variance_pct:.0f}%"
                    
            except Exception as e:
                st.warning(f"Could not extract all metrics for {period} from raw data: {str(e)}")
    
    return metrics

def prepare_llm_input(df_summary, df_raw_pivot, metrics, current_month, period):
    """Prepare the data for LLM driver analysis from either source"""
    if df_summary is not None:
        # Prepare from summary data
        prefix = {
            'month': f"Current Month ({current_month})",
            'ytd': "YTD '25",
            'full_year': "Full Year '25"
        }[period]
        
        summary = {
            'direct_expense': {
                'actual': df_summary.loc['Direct Expense', f"{prefix} Actuals"],
                'budget': df_summary.loc['Direct Expense', f"{prefix} Budget"],
                'variance': df_summary.loc['Direct Expense', f"{prefix} O/U"],
                'variance_pct': df_summary.loc['Direct Expense', f"{prefix} O/U(%)"] if f"{prefix} O/U(%)" in df_summary.columns else None
            }
        }
        
        categories = ['Compensation', 'Non-Compensation', 'Allocations', 'Headcount', 'Employees', 'Contractors']
        category_data = {}
        
        for category in categories:
            if category in df_summary.index:
                try:
                    category_data[category] = {
                        'actual': df_summary.loc[category, f"{prefix} Actuals"],
                        'budget': df_summary.loc[category, f"{prefix} Budget"],
                        'variance': df_summary.loc[category, f"{prefix} O/U"],
                        'variance_pct': df_summary.loc[category, f"{prefix} O/U(%)"] if f"{prefix} O/U(%)" in df_summary.columns else None
                    }
                except:
                    pass
        
        return {
            'summary': summary,
            'categories': category_data,
            'metrics': metrics[period],
            'data_source': 'summary'
        }
    
    elif df_raw_pivot is not None:
        # Prepare from raw data pivot
        col_mapping = {
            'month': ('CY CM Actuals', 'CY CM Budget'),
            'ytd': ('YTD CY Outlook', 'YTD PY Actuals'),
            'full_year': ('Full Year CY Outlook', 'Full Year PY Actuals')
        }
        
        actual_col, budget_col = col_mapping[period]
        
        # Get top contributors to variance
        df_variance = df_raw_pivot.copy()
        df_variance['variance'] = df_variance[actual_col] - df_variance[budget_col]
        df_variance['abs_variance'] = df_variance['variance'].abs()
        
        # Get top 5 contributors by absolute variance
        top_contributors = df_variance.nlargest(5, 'abs_variance')
        
        category_data = {}
        for category, row in top_contributors.iterrows():
            category_data[category] = {
                'actual': row[actual_col],
                'budget': row[budget_col],
                'variance': row['variance'],
                'variance_pct': (row['variance'] / row[budget_col]) * 100 if row[budget_col] != 0 else None
            }
        
        return {
            'summary': {
                'direct_expense': {
                    'actual': df_raw_pivot[actual_col].sum(),
                    'budget': df_raw_pivot[budget_col].sum(),
                    'variance': metrics[period]['budget_variance_raw'],
                    'variance_pct': float(metrics[period]['budget_variance_pct'].strip('%')) if 'budget_variance_pct' in metrics[period] else None
                }
            },
            'categories': category_data,
            'metrics': metrics[period],
            'data_source': 'raw',
            'top_contributors': top_contributors
        }

def get_detailed_drivers_from_phi3(llm_input, period, model, tokenizer):
    """Get detailed drivers from Phi-3-Mini using raw data"""
    # Format the detailed category data for the prompt
    categories_text = []
    for category, values in llm_input['categories'].items():
        if values.get('variance') is not None:
            direction = "under budget" if values['variance'] < 0 else "over budget"
            variance_text = ""
            
            if abs(values['variance']) < 1000:
                variance_text = f"{values['variance']:.0f}k"
            elif abs(values['variance']) < 1000000:
                variance_text = f"{values['variance']/1000:.1f}k"
            else:
                variance_text = f"{values['variance']/1000000:.2f}mm"
                
            categories_text.append(f"- {category}: {variance_text} {direction}")
    
    category_data_str = "\n".join(categories_text)
    
    # Format period names for the prompt
    period_names = {
        'month': "the current month",
        'ytd': "year-to-date",
        'full_year': "full year forecast"
    }
    
    # Create a more detailed prompt for raw data
    prompt = f"""<|system|>
You are a financial analyst assistant that identifies key drivers in financial data with specific details.
</|System|>
<|user|>
Based on the following detailed financial data for {period_names[period]}, identify the 2-3 most significant drivers that
explain why Direct Expense is {llm_input['metrics']['variance_direction']} budget by {llm_input['metrics']['budget_variance']}.

Direct Expense: {llm_input['metrics']['direct_expense']}
Variance: {llm_input['metrics']['budget_variance']} ({llm_input['metrics']['variance_direction']} budget)

Detailed expense categories and their variances:
{category_data_str}

IMPORTANT: Return ONLY a comma-separated list of specific, detailed drivers in this format:
"lower average FTE headcount (26), higher Professional Services due to IC payout, increased Vendor spend in IT services"

Key requirements:
1. Be specific with numbers where available
2. Mention sub-categories when relevant
3. Keep each driver concise but detailed
4. Return ONLY the comma-separated list, no additional text
</|user|>
<|assistant|>"""
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.2,
                do_sample=True,
                top_p=0.9
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("</|user|>")[-1].strip()
        
        # Extract drivers and clean up
        if "," in response:
            drivers = [driver.strip() for driver in response.split(',')]
            return drivers[:3]
        else:
            return [response.strip()]
    except Exception as e:
        st.error(f"Error generating detailed drivers: {str(e)}")
        return ["lower headcount", "higher Professional Services"]  # Fallback

def generate_detailed_commentary(metrics, drivers):
    """Generate more detailed commentary using templates"""
    templates = {
        'month': Template("Month:\nDirect Expense of $$${direct_expense} is $$(${budget_variance}) ${variance_direction} budget driven by:\n- ${driver1}\n- ${driver2}${driver3}"),
        
        'ytd': Template("YTD:\nDirect Expense of $$${direct_expense} is $$(${budget_variance})/(${budget_variance_pct}) ${variance_direction} to budget driven by:\n- ${driver1}\n- ${driver2}${driver3}"),
        
        'full_year': Template("Full Year:\nDirect Expense of $$${direct_expense} is $$${budget_variance}mm ${variance_direction} driven by:\n- ${driver1}\n- ${driver2}${driver3}")
    }
    
    commentary = {}
    
    for period in ['month', 'ytd', 'full_year']:
        if period not in metrics or period not in drivers:
            commentary[period] = f"{period.upper()}: Data not available."
            continue
            
        # Fill driver placeholders
        driver_vars = {
            'driver1': drivers[period][0] if len(drivers[period]) > 0 else "",
            'driver2': drivers[period][1] + "\n- " if len(drivers[period]) > 1 else "",
            'driver3': drivers[period][2] if len(drivers[period]) > 2 else ""
        }
        
        # Generate commentary using template
        try:
            template_data = {
                'direct_expense': metrics[period]['direct_expense'],
                'budget_variance': metrics[period]['budget_variance'],
                'variance_direction': metrics[period]['variance_direction'],
                **driver_vars
            }
            
            if period == 'ytd' and 'budget_variance_pct' in metrics[period]:
                template_data['budget_variance_pct'] = metrics[period]['budget_variance_pct']
            
            commentary[period] = templates[period].substitute(template_data)
        except KeyError as e:
            st.warning(f"Missing data for {period} commentary: {str(e)}")
            commentary[period] = f"{period.upper()}: Insufficient data."
    
    # Combine all sections with appropriate spacing
    full_commentary = ""
    for period in ['month', 'ytd', 'full_year']:
        if period in commentary:
            full_commentary += commentary[period] + "\n\n"
    
    return full_commentary.strip()

@st.cache_data(show_spinner=False)
def process_data_with_llm(file_bytes, model, tokenizer):
    """Process data with LLM and cache the results"""
    try:
        # Load data from bytes
        with BytesIO(file_bytes) as bio:
            # First try to read as summary file
            try:
                df_summary = pd.read_excel(bio, index_col=0)
                bio.seek(0)
                
                # Find current month from column names
                month_pattern = r"Current Month \((.*?)\)"
                current_month = None
                for col in df_summary.columns:
                    match = re.search(month_pattern, col)
                    if match:
                        current_month = match.group(1)
                        break
                
                df_raw_pivot = None
                
            except:
                # If summary read fails, try as raw file
                bio.seek(0)
                df_raw = pd.read_excel(bio)
                df_summary = None
                
                # Check for required columns
                required_columns = ['Expense Details', 'Current Month', 'CY CM Actuals', 
                                  'CY CM Budget', 'Full Year PY Actuals', 'Full Year CY Outlook',
                                  'YTD PY Actuals', 'YTD CY Outlook']
                
                if all(col in df_raw.columns for col in required_columns):
                    # Create pivot table
                    pivot = pd.pivot_table(df_raw, 
                                         index='Expense Details',
                                         values=['CY CM Actuals', 'CY CM Budget',
                                                'Full Year PY Actuals', 'Full Year CY Outlook',
                                                'YTD PY Actuals', 'YTD CY Outlook'],
                                         aggfunc='sum')
                    
                    current_month = "February '2025"  # Placeholder - adjust as needed
                    df_raw_pivot = pivot
                else:
                    st.error("Uploaded file doesn't match expected format")
                    return None, None, None, None, None
        
        # Extract metrics
        metrics = extract_metrics(df_summary, df_raw_pivot, current_month)
        
        # Process with LLM to get drivers
        drivers = {}
        
        for period in ['month', 'ytd', 'full_year']:
            if period in metrics:
                llm_input = prepare_llm_input(df_summary, df_raw_pivot, metrics, current_month, period)
                
                if llm_input['data_source'] == 'raw':
                    # Use detailed driver analysis for raw data
                    drivers[period] = get_detailed_drivers_from_phi3(llm_input, period, model, tokenizer)
                else:
                    # Use standard driver analysis for summary data
                    drivers[period] = get_drivers_from_phi3(llm_input, period, model, tokenizer)
        
        # Generate appropriate commentary based on data source
        if df_raw_pivot is not None:
            commentary = generate_detailed_commentary(metrics, drivers)
        else:
            commentary = generate_commentary(metrics, drivers)
        
        return df_summary, current_month, metrics, drivers, commentary, df_raw_pivot
    
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return None, None, None, None, None, None

def main():
    st.title("Enhanced EMR Commentary Generator")
    st.subheader("Now with Detailed Driver Analysis")
    
    # Load Phi-3 model
    with st.spinner("Initializing Phi-3-Mini model..."):
        model, tokenizer = load_phi3_model()
    
    if model is None or tokenizer is None:
        st.error("Failed to load Phi-3-Mini model. Please check your installation.")
        st.stop()
    
    st.success("✅ Phi-3-Mini model loaded successfully")
    
    # File upload section
    uploaded_file = st.file_uploader("Upload Excel file (summary or raw format)", type=["xlsx", "xls"])
    
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        
        with st.spinner("Processing data and generating commentary..."):
            df_summary, current_month, metrics, drivers, commentary, df_raw_pivot = process_data_with_llm(
                file_bytes, model, tokenizer)
        
        if current_month is not None:
            st.success(f"✅ Analysis complete for {current_month}")
            
            # Show data source info
            if df_raw_pivot is not None:
                st.info("Using detailed raw data for granular driver analysis")
                if st.checkbox("Show raw data pivot"):
                    st.dataframe(df_raw_pivot)
            else:
                st.info("Using summary data for analysis")
                if st.checkbox("Show summary data"):
                    st.dataframe(df_summary)
            
            # Display commentary
            st.subheader("Generated Commentary")
            st.text_area("Commentary", commentary, height=400)
            
            # Download button
            st.download_button(
                label="Download Commentary",
                data=commentary,
                file_name="detailed_commentary.txt",
                mime="text/plain"
            )
            
            # Compare with example
            if st.checkbox("Compare with example detailed commentary"):
                example = """Month:
Direct Expense of $12.6mm is $(56k) under budget driven by:
- Compensation savings from 26 fewer FTEs than planned
- Lower Facilities spend due to delayed office renovations

YTD:
Direct Expense of $24.8mm is ($675k)/(3%) favorable to budget driven by:
- Lower average FTE headcount (26 under plan)
- Reduced Marketing spend ($150k) from delayed campaigns
- Savings in Technology licenses ($120k)

Full Year:
Direct Expense of $156.7mm is $2.5mm unfavorable driven by:
- Higher Professional Services ($1.8mm) due to IC payout
- Increased Vendor spend in IT ($700k) for cloud migration"""
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Generated")
                    st.text(commentary)
                with col2:
                    st.subheader("Example")
                    st.text(example)

if __name__ == "__main__":
    main()
