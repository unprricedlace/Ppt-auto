Generated Drivers
MONTH Drivers

lower headcount, higher Professional Services

YTD Drivers

lower headcount, higher Vendor spend

FULL_YEAR Drivers

higher Compensation costs, higher Contractor spend

Subcategory Groupings
{
"Other":{
"actual":-637.4597900000035
"budget":-1053.3547600000174
"variance":415.8949700000067
}
"Non-Compensation":{
"actual":540.04069
"budget":548.89224
"variance":-8.851549999999975
}
}


YTD:
Direct Expense of $24.8mm is ($675)/(3%) favorable to budget
driven mostly by lower headcount

Full Year:
Direct Expense of $156.7mm is $2.5mm unfavorable driven by 


"Compensation", "Non-Compensation", "Direct Expense", "Direct Exp w/o Severance",
        "Allocations", "Total Expense", "Employees", "Contractors", "Total Headcount"





















import pandas as pd
import streamlit as st
from string import Template
import re
from io import BytesIO
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict

# Configuration - EDIT THIS TO MATCH YOUR CATEGORIES
CATEGORY_MAPPING = {
    'Professional Services': 'Non-Compensation',
    'IC Payout': 'Compensation',
    'IT Vendor Spend': 'Non-Compensation',
    'Facilities': 'Non-Compensation',
    'Employee Bonuses': 'Compensation',
    'Recruitment Fees': 'Compensation',
    'Technology Licenses': 'Non-Compensation',
    'Marketing Events': 'Non-Compensation',
    'Travel & Entertainment': 'Non-Compensation',
    'Training Costs': 'Compensation'
}

def load_and_process_summary(file):
    """Load and process summary Excel file"""
    try:
        df = pd.read_excel(file, index_col=0)
        month_pattern = r"Current Month \((.*?)\)"
        current_month = None
        for col in df.columns:
            match = re.search(month_pattern, col)
            if match:
                current_month = match.group(1)
                break
        return df, current_month
    except Exception as e:
        st.error(f"Error processing summary file: {str(e)}")
        return None, None

def process_raw_file(file):
    """Process raw detailed Excel file"""
    try:
        df = pd.read_excel(file)
        required_columns = ['Expense Details', 'Current Month', 'CY CM Actuals', 
                          'CY CM Budget', 'Full Year PY Actuals', 'Full Year CY Outlook',
                          'YTD PY Actuals', 'YTD CY Outlook']
        
        if not all(col in df.columns for col in required_columns):
            st.error("Raw file missing required columns")
            return None
        
        pivot = pd.pivot_table(df, 
                             index='Expense Details',
                             values=['CY CM Actuals', 'CY CM Budget',
                                    'Full Year PY Actuals', 'Full Year CY Outlook',
                                    'YTD PY Actuals', 'YTD CY Outlook'],
                             aggfunc='sum')
        return pivot
    except Exception as e:
        st.error(f"Error processing raw file: {str(e)}")
        return None

def group_subcategories(raw_pivot, period_prefix):
    """Group granular expenses into main categories"""
    grouped = defaultdict(lambda: {'actual': 0, 'budget': 0, 'variance': 0})
    
    if period_prefix == 'month':
        actual_col, budget_col = 'CY CM Actuals', 'CY CM Budget'
    elif period_prefix == 'ytd':
        actual_col, budget_col = 'YTD CY Outlook', 'YTD PY Actuals'
    else:  # full_year
        actual_col, budget_col = 'Full Year CY Outlook', 'Full Year PY Actuals'
    
    for subcategory, row in raw_pivot.iterrows():
        main_category = CATEGORY_MAPPING.get(subcategory, 'Other')
        grouped[main_category]['actual'] += row[actual_col]
        grouped[main_category]['budget'] += row[budget_col]
        grouped[main_category]['variance'] += (row[actual_col] - row[budget_col])
    
    return grouped

def extract_metrics(df_summary, current_month):
    """Extract key metrics from summary dataframe"""
    metrics = {}
    time_periods = {
        'month': f"Current Month ({current_month})",
        'ytd': "YTD '25",
        'full_year': "Full Year '25"
    }
    
    for period, prefix in time_periods.items():
        try:
            direct_expense_actual = df_summary.loc['Direct Expense', f"{prefix} Actuals"]
            direct_expense_budget = df_summary.loc['Direct Expense', f"{prefix} Budget"]
            variance = df_summary.loc['Direct Expense', f"{prefix} O/U"]
            variance_pct = df_summary.loc['Direct Expense', f"{prefix} O/U(%)"] if f"{prefix} O/U(%)" in df_summary.columns else None
            
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
            st.warning(f"Could not extract metrics for {period}: {str(e)}")
    
    return metrics

def prepare_llm_input(df_summary, metrics, current_month, period):
    """Prepare data for LLM analysis"""
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
    
    categories = ['Compensation', 'Non-Compensation', 'Allocations', 'Employees', 'Contractors']
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
        'metrics': metrics[period]
    }

@st.cache_resource
def load_phi3_model():
    """Load the Phi-3-Mini model"""
    try:
        with st.spinner("Loading Phi-3-Mini model..."):
            model_name = "microsoft/phi-3-mini-128k-instruct"
            tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                local_files_only=True
            )
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def get_drivers_from_phi3(llm_input, period, model, tokenizer):
    """Get drivers from Phi-3 model"""
    categories_text = []
    for category, values in llm_input['categories'].items():
        if values.get('variance') is not None:
            direction = "under budget" if values['variance'] < 0 else "over budget"
            variance_text = f"{values['variance']:.0f}k" if abs(values['variance']) < 1000 else f"{values['variance']/1000:.1f}mm"
            categories_text.append(f"- {category}: {variance_text} {direction}")
    
    category_data_str = "\n".join(categories_text)
    
    period_names = {
        'month': "the current month",
        'ytd': "year-to-date",
        'full_year': "full year forecast"
    }
    
    prompt = f"""<|system|>
You are a financial analyst assistant. Identify key budget drivers.
</|system|>
<|user|>
For {period_names[period]}, identify 2-3 main drivers why Direct Expense is {llm_input['metrics']['variance_direction']} budget by {llm_input['metrics']['budget_variance']}.

Direct Expense: {llm_input['metrics']['direct_expense']}
Variance: {llm_input['metrics']['budget_variance']} ({llm_input['metrics']['variance_direction']} budget)

Categories:
{category_data_str}

Return ONLY a comma-separated list like:
"lower headcount, higher Vendor spend, higher Professional Services"
</|user|>
<|assistant|>"""
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.1,
            do_sample=True,
            top_p=0.95
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("</|user|>")[-1].strip()
        
        if "," in response:
            drivers = [driver.strip() for driver in response.split(',')]
            return drivers[:3]
        else:
            return [response.strip()]
    except Exception as e:
        st.error(f"Error generating drivers: {str(e)}")
        return ["lower headcount", "higher Professional Services"]

def generate_hybrid_commentary(metrics, drivers, grouped_details):
    """Generate complete commentary with all periods and proper driver inclusion"""
    commentary = []
    
    # Month Section
    if 'month' in metrics and 'month' in drivers:
        month_template = """Month:
Direct Expense of ${direct_expense} is (${variance_abs}) ${direction} budget
driven by:
{drivers_list}"""
        
        drivers_list = []
        for driver in drivers['month']:
            if 'lower' in driver:
                drivers_list.append(f"- {driver.capitalize()}")
            else:
                # Find matching subcategories
                for subcat, data in grouped_details.get('month', {}).items():
                    if data['variance'] > 0 and subcat.lower() in driver.lower():
                        amount = f"${abs(data['variance']):.0f}k" if abs(data['variance']) < 1000 else f"${abs(data['variance'])/1000:.1f}K"
                        drivers_list.append(f"- {subcat} ({amount})")
        
        month_section = month_template.format(
            direct_expense=metrics['month']['direct_expense'],
            variance_abs=metrics['month']['budget_variance'].replace('k', ''),
            direction=metrics['month']['variance_direction'],
            drivers_list='\n'.join(drivers_list) if drivers_list else "No significant drivers"
        )
        commentary.append(month_section)
    
    # YTD Section
    if 'ytd' in metrics and 'ytd' in drivers:
        ytd_template = """YTD:
Direct Expense of ${direct_expense} is (${variance_abs})/({variance_pct}) {direction} to budget
driven by:
{drivers_list}"""
        
        drivers_list = []
        for driver in drivers['ytd']:
            if 'lower' in driver:
                drivers_list.append(f"- {driver.capitalize()}")
            else:
                # Find matching subcategories
                for subcat, data in grouped_details.get('ytd', {}).items():
                    if data['variance'] > 0 and subcat.lower() in driver.lower():
                        amount = f"${abs(data['variance']):.0f}k" if abs(data['variance']) < 1000 else f"${abs(data['variance'])/1000:.1f}K"
                        drivers_list.append(f"- {subcat} ({amount})")
        
        ytd_section = ytd_template.format(
            direct_expense=metrics['ytd']['direct_expense'],
            variance_abs=metrics['ytd']['budget_variance'].replace('k', ''),
            variance_pct=metrics['ytd'].get('budget_variance_pct', 'N/A'),
            direction=metrics['ytd']['variance_direction'],
            drivers_list='\n'.join(drivers_list) if drivers_list else "No significant drivers"
        )
        commentary.append(ytd_section)
    
    # Full Year Section
    if 'full_year' in metrics and 'full_year' in drivers:
        fy_template = """Full Year:
Direct Expense of ${direct_expense} is ${variance_abs}mm {direction} driven by:
{drivers_list}"""
        
        drivers_list = []
        for driver in drivers['full_year']:
            if 'higher' in driver:
                # Find matching subcategories
                for subcat, data in grouped_details.get('full_year', {}).items():
                    if data['variance'] > 0 and subcat.lower() in driver.lower():
                        amount = f"${abs(data['variance'])/1000000:.2f}mm"
                        drivers_list.append(f"- {subcat} ({amount})")
            else:
                drivers_list.append(f"- {driver.capitalize()}")
        
        fy_section = fy_template.format(
            direct_expense=metrics['full_year']['direct_expense'],
            variance_abs=metrics['full_year']['budget_variance'].replace('mm', ''),
            direction=metrics['full_year']['variance_direction'],
            drivers_list='\n'.join(drivers_list) if drivers_list else "No significant drivers"
        )
        commentary.append(fy_section)
    
    return '\n\n'.join(commentary)

def generate_commentary(metrics, drivers):
    """Fallback commentary for summary-only data"""
    templates = {
        'month': Template("Month:\nDirect Expense of $$${direct_expense} is $$(${budget_variance}) ${variance_direction} budget mostly driven by ${drivers}."),
        'ytd': Template("YTD:\nDirect Expense of $$${direct_expense} is $$(${budget_variance})/(${budget_variance_pct}) ${variance_direction} to budget driven mostly by ${drivers}."),
        'full_year': Template("Full Year:\nDirect Expense of $$${direct_expense} is $$${budget_variance}mm ${variance_direction} driven by ${drivers}.")
    }
    
    commentary = {}
    for period in ['month', 'ytd', 'full_year']:
        if period in metrics and period in drivers:
            drivers_text = ", ".join(drivers[period][:3])  # Top 3 drivers
            
            template_data = {
                'direct_expense': metrics[period]['direct_expense'],
                'budget_variance': metrics[period]['budget_variance'],
                'variance_direction': metrics[period]['variance_direction'],
                'drivers': drivers_text
            }
            
            if period == 'ytd' and 'budget_variance_pct' in metrics[period]:
                template_data['budget_variance_pct'] = metrics[period]['budget_variance_pct']
            
            commentary[period] = templates[period].substitute(template_data)
    
    return "\n\n".join([c for c in commentary.values() if c])

def format_metrics_for_display(metrics):
    """Convert metrics to display-friendly format"""
    display_metrics = {}
    for period, data in metrics.items():
        display_data = data.copy()
        # Remove raw values for cleaner display
        display_data.pop('direct_expense_raw', None)
        display_data.pop('budget_variance_raw', None)
        display_data.pop('data_source', None)
        display_metrics[period] = display_data
    return display_metrics

def main():
    st.set_page_config(layout="wide")
    st.title("📊 Advanced EMR Commentary Generator")
    st.caption("Combine summary and detailed expense data for richer insights")
    
    # Load model
    model, tokenizer = load_phi3_model()
    if model is None:
        st.stop()
    
    # File upload section
    st.sidebar.header("Upload Files")
    summary_file = st.sidebar.file_uploader("1. Summary File (Required)", type=["xlsx"])
    raw_file = st.sidebar.file_uploader("2. Raw Data File (Optional)", type=["xlsx"])
    
    if summary_file:
        # Process files
        with st.spinner("Processing data..."):
            df_summary, current_month = load_and_process_summary(summary_file)
            
            if df_summary is None:
                st.error("Failed to process summary file")
                st.stop()
            
            metrics = extract_metrics(df_summary, current_month)
            
            # Process raw file if provided
            raw_pivot = None
            grouped_details = {}
            if raw_file:
                raw_pivot = process_raw_file(raw_file)
                if raw_pivot is not None:
                    for period in ['month', 'ytd', 'full_year']:
                        if period in metrics:
                            prefix = 'month' if period == 'month' else 'ytd' if period == 'ytd' else 'full_year'
                            grouped_details[period] = group_subcategories(raw_pivot, prefix)
            
            # Get drivers from LLM
            with st.spinner("Analyzing drivers..."):
                drivers = {}
                for period in ['month', 'ytd', 'full_year']:
                    if period in metrics:
                        llm_input = prepare_llm_input(df_summary, metrics, current_month, period)
                        drivers[period] = get_drivers_from_phi3(llm_input, period, model, tokenizer)
        
        # Display results
        st.success("Analysis complete!")
        
        # Metrics and Drivers Section
        with st.expander("📈 Metrics & Drivers Analysis", expanded=True):
            tab1, tab2 = st.tabs(["Metrics", "Drivers"])
            
            with tab1:
                st.subheader("Extracted Metrics")
                display_metrics = format_metrics_for_display(metrics)
                st.json(display_metrics)
                
                if raw_pivot is not None:
                    st.subheader("Raw Data Summary")
                    st.dataframe(raw_pivot.describe())
            
            with tab2:
                st.subheader("Generated Drivers")
                for period in ['month', 'ytd', 'full_year']:
                    if period in drivers:
                        st.markdown(f"**{period.upper()} Drivers**")
                        st.write(", ".join(drivers[period]))
                
                if grouped_details:
                    st.subheader("Subcategory Groupings")
                    st.json(grouped_details.get('ytd', {}))
        
        # Generate and display commentary
        st.subheader("📝 Generated Commentary")
        if raw_file and grouped_details:
            commentary = generate_hybrid_commentary(metrics, drivers, grouped_details)
            st.success("Using enhanced commentary with detailed subcategories")
        else:
            commentary = generate_commentary(metrics, drivers)
            st.info("Using standard commentary (upload raw file for detailed analysis)")
        
        st.text_area("Commentary", commentary, height=200)
        
        # Add download buttons
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="Download Commentary",
                data=commentary,
                file_name="emr_commentary.txt",
                mime="text/plain"
            )
        with col2:
            csv = pd.DataFrame.from_dict(metrics, orient='index').to_csv()
            st.download_button(
                label="Download Metrics (CSV)",
                data=csv,
                file_name="emr_metrics.csv",
                mime="text/csv"
            )
        
        # Data previews
        with st.expander("🔍 Data Previews"):
            st.subheader("Summary Data Preview")
            st.dataframe(df_summary.head())
            
            if raw_pivot is not None:
                st.subheader("Raw Data Pivot Preview")
                st.dataframe(raw_pivot.head())

if __name__ == "__main__":
    main()
