"""
utility.py - Excel Commentary Generator Utility Functions

This module contains the core functionality for the Excel Commentary Generator,
implemented using functional programming principles.
"""

import pandas as pd
import numpy as np
import json
import logging
import sys
import torch
from datetime import datetime
from typing import Dict, List, Any, Union
from functools import partial
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("excel_commentary.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Model ID - hardcoded since we're only using one model
MODEL_ID = "microsoft/phi-3-mini-4k-instruct"

# Global variables for model and tokenizer caching
model = None
tokenizer = None

def load_excel_file(file_path: str) -> Dict[str, pd.DataFrame]:
    """
    Load all sheets from an Excel file into dataframes.
    
    Args:
        file_path: Path to the Excel file
        
    Returns:
        Dictionary of sheet names to dataframes
    """
    logger.info(f"Loading Excel file: {file_path}")
    try:
        with pd.ExcelFile(file_path) as excel_file:
            # Use dictionary comprehension for a more functional approach
            sheets = {
                sheet_name: pd.read_excel(excel_file, sheet_name=sheet_name)
                for sheet_name in excel_file.sheet_names
            }
        
        # Log information
        for sheet_name, df in sheets.items():
            logger.info(f"Loaded sheet '{sheet_name}' with shape {df.shape}")
            
        return sheets
    except Exception as e:
        logger.error(f"Error loading Excel file: {e}")
        raise

def calculate_statistics(df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Any]:
    """Calculate basic statistics for numeric columns."""
    return {
        "summary": df[numeric_cols].describe().to_dict(),
    }

def calculate_period_changes(df: pd.DataFrame, numeric_cols: List[str], date_col: str) -> Dict[str, Any]:
    """Calculate changes between periods for time-series data."""
    # Create a new dataframe to avoid modifying the original
    df_with_changes = df.copy()
    df_with_changes[date_col] = pd.to_datetime(df_with_changes[date_col])
    df_with_changes = df_with_changes.sort_values(by=date_col)
    
    # Calculate changes
    for col in numeric_cols:
        df_with_changes[f"{col}_change"] = df_with_changes[col].pct_change() * 100
        df_with_changes[f"{col}_abs_change"] = df_with_changes[col].diff()
    
    # Get last and previous rows
    last_row = df_with_changes.iloc[-1].to_dict()
    prev_row = df_with_changes.iloc[-2].to_dict() if len(df_with_changes) > 1 else None
    
    result = {
        "latest_period": {
            "date": last_row[date_col].strftime("%Y-%m-%d") if not pd.isna(last_row[date_col]) else None,
            "values": {col: last_row[col] for col in numeric_cols if col in last_row and not pd.isna(last_row[col])},
            "changes": {col: last_row.get(f"{col}_change") for col in numeric_cols 
                      if f"{col}_change" in last_row and not pd.isna(last_row.get(f"{col}_change"))}
        }
    }
    
    if prev_row:
        result["previous_period"] = {
            "date": prev_row[date_col].strftime("%Y-%m-%d") if not pd.isna(prev_row[date_col]) else None,
            "values": {col: prev_row[col] for col in numeric_cols if col in prev_row and not pd.isna(prev_row[col])}
        }
    
    return result

def find_extrema(df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Any]:
    """Find maximum and minimum values in the dataframe."""
    return {
        "maxima": {col: {"value": df[col].max(), "row": df.loc[df[col].idxmax()].to_dict()} 
                 for col in numeric_cols if not df[col].isna().all()},
        "minima": {col: {"value": df[col].min(), "row": df.loc[df[col].idxmin()].to_dict()} 
                 for col in numeric_cols if not df[col].isna().all()}
    }

def calculate_trends(df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Any]:
    """Calculate growth trends for numeric columns."""
    if len(df) <= 1:
        return {"trends": {}}
    
    trends = {}
    for col in numeric_cols:
        start_val = df[col].iloc[0]
        end_val = df[col].iloc[-1]
        
        if start_val != 0 and not pd.isna(start_val) and not pd.isna(end_val):
            total_growth = ((end_val - start_val) / start_val) * 100
            trends[col] = {
                "start_value": start_val,
                "end_value": end_val,
                "total_growth_percent": total_growth,
                "is_positive": total_growth > 0
            }
    
    return {"trends": trends}

def find_outliers(df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Any]:
    """Identify notable outliers (values > 2 std devs from mean)."""
    outliers = {}
    for col in numeric_cols:
        if not df[col].isna().all():
            mean = df[col].mean()
            std = df[col].std()
            outlier_indices = df[df[col] > mean + 2*std].index.tolist() + df[df[col] < mean - 2*std].index.tolist()
            
            if outlier_indices:
                outliers[col] = {
                    "indices": outlier_indices,
                    "values": df.loc[outlier_indices, col].tolist(),
                    "rows": df.loc[outlier_indices].to_dict('records')
                }
    
    return {"outliers": outliers}

def analyze_financial_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze financial data to extract key insights.
    
    Args:
        df: DataFrame containing financial data
        
    Returns:
        Dictionary of analysis results
    """
    logger.info("Analyzing financial data")
    
    try:
        # Find numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            return {}
        
        # Apply analysis functions and merge results
        analysis = calculate_statistics(df, numeric_cols)
        
        # Calculate period changes if date column exists
        date_col = next((col for col in ["date", "Date"] if col in df.columns), None)
        if date_col:
            analysis.update(calculate_period_changes(df, numeric_cols, date_col))
        
        # Apply other analysis functions
        analysis.update(find_extrema(df, numeric_cols))
        analysis.update(calculate_trends(df, numeric_cols))
        analysis.update(find_outliers(df, numeric_cols))
        
        logger.info(f"Analysis complete, identified {len(analysis.keys())} key areas")
        return analysis
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        return {"error": str(e)}

def clean_for_json(obj: Any) -> Any:
    """
    Clean an object to make it JSON-serializable.
    
    Args:
        obj: Any Python object
        
    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    elif isinstance(obj, (pd.Timestamp, datetime)):
        return obj.strftime("%Y-%m-%d %H:%M:%S")
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    elif pd.isna(obj):
        return None
    else:
        return str(obj)

def create_prompt(sheet_name: str, df: pd.DataFrame, analysis: Dict[str, Any]) -> str:
    """
    Create a prompt for the LLM based on the analysis.
    
    Args:
        sheet_name: Name of the Excel sheet
        df: DataFrame containing the sheet's data
        analysis: Analysis results from analyze_financial_data
        
    Returns:
        Prompt text for the LLM
    """
    # Get column names and first few rows for context
    columns = df.columns.tolist()
    sample_data = df.head(3).to_dict('records')
    
    # Create a JSON-serializable version of the analysis
    clean_analysis = clean_for_json(analysis)
    
    prompt = f"""
    <|system|>
    You are a financial analyst tasked with writing insightful commentary for executive reports based on Excel data analysis.
    </|system|>
    
    <|user|>
    Write an executive summary commentary for a financial report sheet named "{sheet_name}".
    
    The data has the following columns: {columns}
    
    Here are some sample rows from the data:
    {json.dumps(sample_data, indent=2, default=str)}
    
    Based on our analysis, here are the key insights:
    {json.dumps(clean_analysis, indent=2, default=str)}
    
    Please generate a professional executive commentary that:
    1. Summarizes the overall performance shown in this data
    2. Highlights key trends and notable changes
    3. Points out any significant outliers or anomalies
    4. Provides context for the most important metrics
    5. Offers brief forward-looking statements when appropriate
    
    Keep the tone professional but accessible. Use specific numbers from the analysis to support your points.
    Format the commentary in paragraphs suitable for inclusion in an executive presentation.
    </|user|>
    
    <|assistant|>
    """
    
    return prompt

def load_model_and_tokenizer():
    """
    Load the model and tokenizer if they haven't been loaded yet.
    
    Returns:
        tuple: (model, tokenizer)
    """
    global model, tokenizer
    
    if model is None or tokenizer is None:
        try:
            logger.info(f"Loading model and tokenizer: {MODEL_ID}")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            
            # Load model with optimizations
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,  # Use half precision for efficiency
                device_map="auto"  # Use GPU if available
            )
            
            logger.info("Model and tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    return model, tokenizer

def generate_commentary(model_param: str, sheet_name: str, df: pd.DataFrame, analysis: Dict[str, Any], 
                       temperature: float = 0.7, max_tokens: int = 1000) -> str:
    """
    Generate commentary for a sheet based on analysis results using AutoModelForCausalLM.
    
    Args:
        model_param: Placeholder parameter for compatibility (ignored)
        sheet_name: Name of the Excel sheet
        df: DataFrame containing the sheet's data
        analysis: Analysis results from analyze_financial_data
        temperature: Temperature parameter for generation
        max_tokens: Maximum tokens to generate
        
    Returns:
        Generated commentary text
    """
    logger.info(f"Generating commentary for sheet: {sheet_name}")
    
    # Prepare input for the LLM
    prompt = create_prompt(sheet_name, df, analysis)
    
    try:
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer()
        
        # Tokenize prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate text
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode the generated tokens
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the assistant's response
        commentary = generated_text.split("<|assistant|>")[-1].strip()
        
        logger.info(f"Successfully generated commentary ({len(commentary)} chars)")
        return commentary
        
    except Exception as e:
        logger.error(f"Error generating commentary: {e}")
        return f"Error generating commentary: {str(e)}"

def process_sheet(model_param: str, sheet_name: str, df: pd.DataFrame, temperature: float = 0.7, max_tokens: int = 1000) -> str:
    """Process a single sheet and generate commentary."""
    # Skip sheets with no data
    if df.empty:
        return "No data available in this sheet."
        
    # Analyze the data
    analysis = analyze_financial_data(df)
    
    # Generate commentary (model_param parameter is kept for compatibility)
    return generate_commentary(model_param, sheet_name, df, analysis, temperature, max_tokens)

def process_excel_file(model_param: str, file_path: str, temperature: float = 0.7, max_tokens: int = 1000) -> Dict[str, str]:
    """
    Process an Excel file and generate commentary for all sheets.
    
    Args:
        model_param: Model name parameter (kept for compatibility)
        file_path: Path to the Excel file
        temperature: Temperature parameter for generation
        max_tokens: Maximum tokens to generate
        
    Returns:
        Dictionary mapping sheet names to commentaries
    """
    logger.info(f"Processing Excel file: {file_path}")
    
    # Load the Excel file
    sheets = load_excel_file(file_path)
    
    # Create a partially applied function for processing sheets
    process_sheet_with_params = partial(process_sheet, model_param, temperature=temperature, max_tokens=max_tokens)
    
    # Generate commentary for each sheet using dictionary comprehension
    commentaries = {
        sheet_name: process_sheet_with_params(sheet_name, df)
        for sheet_name, df in sheets.items()
    }
    
    return commentaries

def save_commentaries(commentaries: Dict[str, str], output_path: str) -> None:
    """
    Save generated commentaries to a file.
    
    Args:
        commentaries: Dictionary mapping sheet names to commentaries
        output_path: Path to save the commentaries
    """
    logger.info(f"Saving commentaries to: {output_path}")
    
    with open(output_path, 'w') as f:
        json.dump(commentaries, f, indent=2)
        
    logger.info(f"Commentaries saved successfully")
