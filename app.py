"""
app.py - Excel Commentary Generator Streamlit Interface

This script provides a web interface for the Excel Commentary Generator using Streamlit,
with drag-and-drop functionality for Excel files.
"""

import streamlit as st
import pandas as pd
import json
import os
import tempfile
from datetime import datetime


# Import functions from utility.py
from utility import (
    load_excel_file,
    analyze_financial_data,
    generate_commentary,
    process_excel_file,
    save_commentaries
)

# Set page configuration
st.set_page_config(
    page_title="Excel Commentary Generator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e6f3ff;
        font-weight: bold;
    }
    .drag-container {
        border: 2px dashed #aaa;
        border-radius: 10px;
        padding: 30px;
        text-align: center;
        margin-bottom: 20px;
        background-color: #f8f9fa;
    }
    .card {
        padding: 20px;
        border-radius: 10px;
        background-color: #ffffff;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""
if 'model' not in st.session_state:
    st.session_state.model = "gpt-4"
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'sheets_data' not in st.session_state:
    st.session_state.sheets_data = {}
if 'commentaries' not in st.session_state:
    st.session_state.commentaries = {}

def main():
    """Main Streamlit application"""
    
    # Create sidebar
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input("OpenAI API Key", value=st.session_state.api_key, type="password")
        if api_key != st.session_state.api_key:
            st.session_state.api_key = api_key
        
        # Model selection
        model = st.selectbox(
            "Select LLM Model",
            ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
            index=1
        )
        if model != st.session_state.model:
            st.session_state.model = model
        
        # Advanced options expander
        with st.expander("Advanced Options"):
            temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
            max_tokens = st.slider("Max Tokens", min_value=100, max_value=2000, value=1000, step=100)
            
        # Information
        st.divider()
        st.markdown("### About")
        st.info(
            "This tool generates executive commentary from Excel data using AI. "
            "Upload an Excel file, and get professional insights for each sheet."
        )
    
    # Main content area
    st.title("📊 Excel Commentary Generator")
    st.write("Generate professional commentaries for your Excel financial data")
    
    # File uploader with drag & drop
    uploaded_file = st.file_uploader(
        "Drag and drop your Excel file here",
        type=["xlsx", "xls"],
        key="excel_uploader",
        help="Upload an Excel file containing financial data"
    )
    
    # Process uploaded file
    if uploaded_file is not None and (st.session_state.uploaded_file is None or 
                                     uploaded_file.name != st.session_state.uploaded_file.name):
        st.session_state.uploaded_file = uploaded_file
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_filepath = tmp_file.name
        
        try:
            # Load sheets from the Excel file
            with st.spinner("Loading Excel data..."):
                st.session_state.sheets_data = load_excel_file(tmp_filepath)
                st.success(f"Successfully loaded {len(st.session_state.sheets_data)} sheets from {uploaded_file.name}")
        except Exception as e:
            st.error(f"Error loading Excel file: {str(e)}")
            st.session_state.sheets_data = {}
        finally:

            try:
                import time
                time.sleep(1)
                # Clean up the temporary file
                if os.path.exists(tmp_filepath):
                    os.unlink(tmp_filepath)
            except PermissionError:
                pass
    
    # If we have sheets data, display them
    if st.session_state.sheets_data:
        # Create tabs for each sheet
        sheet_names = list(st.session_state.sheets_data.keys())
        tabs = st.tabs(sheet_names)
        
        # Display each sheet in its own tab
        for i, (sheet_name, df) in enumerate(st.session_state.sheets_data.items()):
            with tabs[i]:
                st.markdown(f"### Sheet: {sheet_name}")
                
                # Display the dataframe
                with st.expander("View Data", expanded=False):
                    st.dataframe(df, use_container_width=True)
                
                # Generate commentary button
                if st.button(f"Generate Commentary for {sheet_name}", key=f"gen_btn_{i}"):
                    if not st.session_state.api_key:
                        st.error("Please enter your OpenAI API key in the sidebar")
                    else:
                        with st.spinner(f"Analyzing data and generating commentary for {sheet_name}..."):
                            try:
                                # Analyze data
                                analysis = analyze_financial_data(df)
                                
                                # Generate commentary
                                commentary = generate_commentary(
                                    api_key=st.session_state.api_key,
                                    model=st.session_state.model,
                                    sheet_name=sheet_name,
                                    df=df,
                                    analysis=analysis
                                )
                                
                                # Store the commentary
                                if 'commentaries' not in st.session_state:
                                    st.session_state.commentaries = {}
                                st.session_state.commentaries[sheet_name] = commentary
                                
                                st.success("Commentary generated successfully!")
                            except Exception as e:
                                st.error(f"Error generating commentary: {str(e)}")
                
                # Display commentary if available
                if sheet_name in st.session_state.commentaries:
                    st.markdown("### Generated Commentary")
                    st.markdown(f"<div class='card'>{st.session_state.commentaries[sheet_name]}</div>", unsafe_allow_html=True)
        
        # Generate all commentaries button
        if st.button("Generate All Commentaries"):
            if not st.session_state.api_key:
                st.error("Please enter your OpenAI API key in the sidebar")
            else:
                with st.spinner("Generating commentaries for all sheets..."):
                    try:
                        # Create a temporary file again
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
                            tmp_file.write(st.session_state.uploaded_file.getvalue())
                            tmp_filepath = tmp_file.name
                        
                        # Process the Excel file
                        commentaries = process_excel_file(
                            api_key=st.session_state.api_key,
                            model=st.session_state.model,
                            file_path=tmp_filepath
                        )
                        
                        # Store the commentaries
                        st.session_state.commentaries = commentaries
                        st.success("All commentaries generated successfully!")
                        
                        try:
                            import time 
                            time.sleep(1)
                            # Clean up the temporary file
                            if os.path.exists(tmp_filepath):
                                os.unlink(tmp_filepath)
                        except PermissionError:
                            pass

                    except Exception as e:
                        st.error(f"Error generating commentaries: {str(e)}")
        
        # Download commentaries button
        if st.session_state.commentaries:
            commentaries_json = json.dumps(st.session_state.commentaries, indent=2)
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.download_button(
                    label="Download Commentaries (JSON)",
                    data=commentaries_json,
                    file_name=f"commentaries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            with col2:
                # Create a markdown version for download
                markdown_content = ""
                for sheet_name, commentary in st.session_state.commentaries.items():
                    markdown_content += f"# Commentary for {sheet_name}\n\n"
                    markdown_content += commentary + "\n\n---\n\n"
                
                st.download_button(
                    label="Download Commentaries (Markdown)",
                    data=markdown_content,
                    file_name=f"commentaries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
    
    else:
        # Display sample/placeholder when no file is uploaded
        st.markdown("""
        <div class="drag-container">
            <h3>Upload an Excel file to get started</h3>
            <p>Drag and drop your Excel file above or click to browse</p>
            <p>The file should contain financial data for analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Example of what the app does
        st.markdown("### Example Process")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="card">
                <h3>1. Upload Excel</h3>
                <p>Upload your financial spreadsheet with data to analyze</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="card">
                <h3>2. AI Analysis</h3>
                <p>Our AI analyzes trends, outliers, and key metrics</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="card">
                <h3>3. Get Commentary</h3>
                <p>Receive professional commentary ready for reports</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
