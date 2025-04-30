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

def prepare_llm_input(df, metrics, current_month, period):
    """Prepare the data for LLM driver analysis"""
    prefix = {
        'month': f"Current Month ({current_month})",
        'ytd': "YTD '25",
        'full_year': "Full Year '25"
    }[period]
    
    # Create a summary of key metrics for the period
    summary = {
        'direct_expense': {
            'actual': df.loc['Direct Expense', f"{prefix} Actuals"],
            'budget': df.loc['Direct Expense', f"{prefix} Budget"],
            'variance': df.loc['Direct Expense', f"{prefix} O/U"],
            'variance_pct': df.loc['Direct Expense', f"{prefix} O/U(%)"] if f"{prefix} O/U(%)" in df.columns else None
        }
    }
    
    # Add data for all relevant categories
    categories = ['Compensation', 'Non-Compensation', 'Allocations', 'Headcount', 'Employees', 'Contractors']
    category_data = {}
    
    for category in categories:
        if category in df.index:
            try:
                category_data[category] = {
                    'actual': df.loc[category, f"{prefix} Actuals"],
                    'budget': df.loc[category, f"{prefix} Budget"],
                    'variance': df.loc[category, f"{prefix} O/U"],
                    'variance_pct': df.loc[category, f"{prefix} O/U(%)"] if f"{prefix} O/U(%)" in df.columns else None
                }
            except:
                pass  # Skip categories with missing data
    
    return {
        'summary': summary,
        'categories': category_data,
        'metrics': metrics[period]
    }

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

def get_drivers_from_phi3(llm_input, period, model, tokenizer):
    """Get drivers from Phi-3-Mini with a constrained prompt"""
    # Format the category data for the prompt
    categories_text = []
    for category, values in llm_input['categories'].items():
        if values.get('variance') is not None:
            direction = "under budget" if values['variance'] < 0 else "over budget"
            variance_text = f"{values['variance']:.0f}k" if abs(values['variance']) < 1000 else f"{values['variance']/1000:.1f}mm"
            categories_text.append(f"- {category}: {variance_text} {direction}")
    
    category_data_str = "\n".join(categories_text)
    
    # Format period names for the prompt
    period_names = {
        'month': "the current month",
        'ytd': "year-to-date",
        'full_year': "full year forecast"
    }
    
    # Create a constrained prompt
    prompt = f"""<|system|>
You are a financial analyst assistant. You identify key drivers in financial data.
<|user|>
Based ONLY on the following financial data for {period_names[period]}, identify the 2-3 most significant drivers that
explain why Direct Expense is {llm_input['metrics']['variance_direction']} budget by {llm_input['metrics']['budget_variance']}.

Direct Expense: {llm_input['metrics']['direct_expense']}
Variance: {llm_input['metrics']['budget_variance']} ({llm_input['metrics']['variance_direction']} budget)

Individual categories:
{category_data_str}

IMPORTANT: Return ONLY a comma-separated list of specific drivers in this format:
"lower headcount, higher Vendor spend, higher Professional Services"

DO NOT include any explanation, numbers, or additional text beyond the drivers.
<|assistant|>"""
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.1,  # Low temperature for consistency
                do_sample=True,
                top_p=0.95
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response (remove the prompt)
        response = response.split("<|assistant|>")[-1].strip()
        
        # Clean the response to ensure we only get the drivers
        if "," in response:
            drivers = [driver.strip() for driver in response.split(',')]
            return drivers[:3]  # Limit to top 3 drivers
        else:
            # If no commas found, try to parse as best as possible
            return [response.strip()]
    except Exception as e:
        st.error(f"Error generating drivers with Phi-3: {str(e)}")
        return ["lower headcount", "higher Professional Services"]  # Fallback

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

@st.cache_data(show_spinner=False)
def process_data_with_llm(file_bytes, model, tokenizer):
    """Process data with LLM and cache the results
    This function takes the file bytes to ensure caching works correctly with uploaded files"""
    
    # Load and process the data from bytes
    try:
        df = pd.read_excel(BytesIO(file_bytes), index_col=0)
        
        # Find current month from column names
        month_pattern = r"Current Month \((.*?)\)"
        current_month = None
        for col in df.columns:
            match = re.search(month_pattern, col)
            if match:
                current_month = match.group(1)
                break
        
        if current_month is None:
            return None, None, None, None
        
        # Extract metrics
        metrics = extract_metrics(df, current_month)
        
        # Process with LLM to get drivers
        drivers = {}
        
        # Process for each time period
        for period in ['month', 'ytd', 'full_year']:
            if period in metrics:
                # Prepare input for LLM
                llm_input = prepare_llm_input(df, metrics, current_month, period)
                
                # Get drivers from Phi-3
                drivers[period] = get_drivers_from_phi3(llm_input, period, model, tokenizer)
        
        # Generate commentary
        commentary = generate_commentary(metrics, drivers)
        
        return df, current_month, metrics, drivers, commentary
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return None, None, None, None, None

def main():
    st.title("Executive Management Report Commentary Generator")
    st.subheader("Phi-3-Mini + Template Approach")
    
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
    
    # File upload section
    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])
    
    if uploaded_file is not None:
        # Read file into memory
        file_bytes = uploaded_file.getvalue()
        
        # Process data and cache results
        with st.spinner("Processing data and generating commentary..."):
            df, current_month, metrics, drivers, commentary = process_data_with_llm(file_bytes, model, tokenizer)
        
        if df is not None and current_month is not None:
            st.success(f"✅ Analysis complete for {current_month}")
            
            # Add toggle for data inspection - no longer triggers reprocessing
            if st.checkbox("Show extracted data"):
                st.subheader("Extracted Metrics")
                st.json(metrics)
                
                st.subheader("Phi-3-Generated Drivers")
                st.json(drivers)
            
            st.subheader("Generated Commentary")
            st.text_area("Commentary", commentary, height=300)
            
            # Add download button
            st.download_button(
                label="Download Commentary",
                data=commentary,
                file_name="commentary.txt",
                mime="text/plain"
            )
            
            # Compare with original - no longer triggers reprocessing
            if st.checkbox("Compare with example commentary"):
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

if __name__ == "__main__":
    main()
