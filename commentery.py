"""
Excel Commentary Generator

This script automates the generation of commentary for executive management reports
based on Excel data analysis.
"""

import pandas as pd
import numpy as np
import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Union
import openai  # For OpenAI's API
# Alternative: import anthropic  # For Anthropic's API

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

class ExcelCommentaryGenerator:
    """Main class for generating commentary from Excel files."""
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        """
        Initialize the commentary generator.
        
        Args:
            api_key: API key for the LLM service
            model: Model name to use (default: gpt-4)
        """
        self.api_key = api_key
        self.model = model
        openai.api_key = api_key
        logger.info(f"Initialized ExcelCommentaryGenerator with model: {model}")
        
    def load_excel_file(self, file_path: str) -> Dict[str, pd.DataFrame]:
        """
        Load all sheets from an Excel file into dataframes.
        
        Args:
            file_path: Path to the Excel file
            
        Returns:
            Dictionary of sheet names to dataframes
        """
        logger.info(f"Loading Excel file: {file_path}")
        try:
            excel_file = pd.ExcelFile(file_path)
            sheets = {}
            
            for sheet_name in excel_file.sheet_names:
                sheets[sheet_name] = pd.read_excel(excel_file, sheet_name=sheet_name)
                logger.info(f"Loaded sheet '{sheet_name}' with shape {sheets[sheet_name].shape}")
                
            return sheets
        except Exception as e:
            logger.error(f"Error loading Excel file: {e}")
            raise
    
    def analyze_financial_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze financial data to extract key insights.
        
        Args:
            df: DataFrame containing financial data
            
        Returns:
            Dictionary of analysis results
        """
        logger.info("Analyzing financial data")
        analysis = {}
        
        # Example analyses - customize based on your specific needs
        try:
            # Find numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols:
                # Calculate basic statistics
                analysis["statistics"] = {
                    "summary": df[numeric_cols].describe().to_dict(),
                }
                
                # Calculate month-over-month or year-over-year changes
                if "date" in df.columns or "Date" in df.columns:
                    date_col = "date" if "date" in df.columns else "Date"
                    df[date_col] = pd.to_datetime(df[date_col])
                    df = df.sort_values(by=date_col)
                    
                    for col in numeric_cols:
                        df[f"{col}_change"] = df[col].pct_change() * 100
                        df[f"{col}_abs_change"] = df[col].diff()
                    
                    last_row = df.iloc[-1].to_dict()
                    prev_row = df.iloc[-2].to_dict() if len(df) > 1 else None
                    
                    analysis["latest_period"] = {
                        "date": last_row[date_col].strftime("%Y-%m-%d") if not pd.isna(last_row[date_col]) else None,
                        "values": {col: last_row[col] for col in numeric_cols if col in last_row and not pd.isna(last_row[col])},
                        "changes": {col: last_row.get(f"{col}_change") for col in numeric_cols 
                                  if f"{col}_change" in last_row and not pd.isna(last_row.get(f"{col}_change"))}
                    }
                    
                    if prev_row:
                        analysis["previous_period"] = {
                            "date": prev_row[date_col].strftime("%Y-%m-%d") if not pd.isna(prev_row[date_col]) else None,
                            "values": {col: prev_row[col] for col in numeric_cols if col in prev_row and not pd.isna(prev_row[col])}
                        }
                
                # Find maximum and minimum values
                analysis["maxima"] = {col: {"value": df[col].max(), "row": df.loc[df[col].idxmax()].to_dict()} 
                                    for col in numeric_cols if not df[col].isna().all()}
                analysis["minima"] = {col: {"value": df[col].min(), "row": df.loc[df[col].idxmin()].to_dict()} 
                                    for col in numeric_cols if not df[col].isna().all()}
                
                # Calculate growth trends
                if len(df) > 1:
                    analysis["trends"] = {}
                    for col in numeric_cols:
                        if df[col].iloc[0] != 0 and not pd.isna(df[col].iloc[0]) and not pd.isna(df[col].iloc[-1]):
                            start_val = df[col].iloc[0]
                            end_val = df[col].iloc[-1]
                            total_growth = ((end_val - start_val) / start_val) * 100
                            analysis["trends"][col] = {
                                "start_value": start_val,
                                "end_value": end_val,
                                "total_growth_percent": total_growth,
                                "is_positive": total_growth > 0
                            }
            
            # Identify notable outliers (values > 2 std devs from mean)
            analysis["outliers"] = {}
            for col in numeric_cols:
                if not df[col].isna().all():
                    mean = df[col].mean()
                    std = df[col].std()
                    outliers = df[df[col] > mean + 2*std].index.tolist() + df[df[col] < mean - 2*std].index.tolist()
                    if outliers:
                        analysis["outliers"][col] = {
                            "indices": outliers,
                            "values": df.loc[outliers, col].tolist(),
                            "rows": df.loc[outliers].to_dict('records')
                        }
                        
            logger.info(f"Analysis complete, identified {len(analysis.keys())} key areas")
            return analysis
            
        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            return {"error": str(e)}
    
    def generate_commentary(self, sheet_name: str, df: pd.DataFrame, analysis: Dict[str, Any]) -> str:
        """
        Generate commentary for a sheet based on analysis results.
        
        Args:
            sheet_name: Name of the Excel sheet
            df: DataFrame containing the sheet's data
            analysis: Analysis results from analyze_financial_data
            
        Returns:
            Generated commentary text
        """
        logger.info(f"Generating commentary for sheet: {sheet_name}")
        
        # Prepare input for the LLM
        prompt = self._create_prompt(sheet_name, df, analysis)
        
        try:
            # Call the LLM API
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a financial analyst tasked with writing insightful commentary for executive reports based on Excel data analysis."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            commentary = response.choices[0].message.content
            logger.info(f"Successfully generated commentary ({len(commentary)} chars)")
            return commentary
            
        except Exception as e:
            logger.error(f"Error generating commentary: {e}")
            return f"Error generating commentary: {str(e)}"
    
    def _create_prompt(self, sheet_name: str, df: pd.DataFrame, analysis: Dict[str, Any]) -> str:
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
        clean_analysis = self._clean_for_json(analysis)
        
        prompt = f"""
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
        """
        
        return prompt
    
    def _clean_for_json(self, obj: Any) -> Any:
        """
        Clean an object to make it JSON-serializable.
        
        Args:
            obj: Any Python object
            
        Returns:
            JSON-serializable version of the object
        """
        if isinstance(obj, dict):
            return {k: self._clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_for_json(item) for item in obj]
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif pd.isna(obj):
            return None
        else:
            return str(obj)
    
    def process_excel_file(self, file_path: str) -> Dict[str, str]:
        """
        Process an Excel file and generate commentary for all sheets.
        
        Args:
            file_path: Path to the Excel file
            
        Returns:
            Dictionary mapping sheet names to commentaries
        """
        logger.info(f"Processing Excel file: {file_path}")
        
        # Load the Excel file
        sheets = self.load_excel_file(file_path)
        
        # Generate commentary for each sheet
        commentaries = {}
        for sheet_name, df in sheets.items():
            # Skip sheets with no data
            if df.empty:
                commentaries[sheet_name] = "No data available in this sheet."
                continue
                
            # Analyze the data
            analysis = self.analyze_financial_data(df)
            
            # Generate commentary
            commentary = self.generate_commentary(sheet_name, df, analysis)
            commentaries[sheet_name] = commentary
        
        return commentaries
    
    def save_commentaries(self, commentaries: Dict[str, str], output_path: str) -> None:
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


def main():
    """Main function to run the commentary generator."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate commentary from Excel files')
    parser.add_argument('--excel_file', required=True, help='Path to the Excel file')
    parser.add_argument('--output_file', default='commentaries.json', help='Path to save commentaries')
    parser.add_argument('--api_key', required=True, help='API key for the LLM service')
    parser.add_argument('--model', default='gpt-4', help='LLM model to use')
    
    args = parser.parse_args()
    
    # Create the commentary generator
    generator = ExcelCommentaryGenerator(api_key=args.api_key, model=args.model)
    
    # Process the Excel file
    commentaries = generator.process_excel_file(args.excel_file)
    
    # Save the commentaries
    generator.save_commentaries(commentaries, args.output_file)
    
    print(f"Commentaries generated and saved to {args.output_file}")


if __name__ == "__main__":
    main()
