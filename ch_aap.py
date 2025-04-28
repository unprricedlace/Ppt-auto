import warnings
warnings.filterwarnings('ignore')
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import logging
from datetime import date, timedelta, datetime
import time
import json
import os
from utility import *
import requests
from requests.auth import HTTPBasicAuth
import uuid

init_app()

# Page configuration
st.set_page_config(
    page_title="FnBM-BOT",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Additional custom CSS for enhanced UI (inline to avoid changing files)
st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 1rem;
        text-align: center;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e57373;
    }
    
    /* Subtitle styling */
    .subtitle {
        font-size: 1.1rem;
        font-weight: 400;
        color: #7f8c8d;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    /* User message styling */
    .user-message {
        background-color: #f0f7ff;
        border-left: 5px solid #3498db;
        padding: 12px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    
    /* Assistant message styling */
    .assistant-message {
        background-color: #f9f9f9;
        border-left: 5px solid #e57373;
        padding: 12px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    
    /* Data preview container */
    .data-preview-container {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
        background-color: #ffffff;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #2c3e50;
        margin: 15px 0 10px 0;
        padding-bottom: 5px;
        border-bottom: 1px solid #ddd;
    }
    
    /* Summary box */
    .summary-box {
        background-color: #e8f4f8;
        border-left: 5px solid #3498db;
        padding: 12px;
        border-radius: 5px;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

st.title = "FnBM-BOT"

def chatbot():
    # Add a custom title with better styling
    st.markdown('<h1 class="main-title">FnBM Analytics Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-Powered Data Insights Tool</p>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown('<div class="section-header">Dashboard Controls</div>', unsafe_allow_html=True)
        
        if st.button("Clear Chat History", use_container_width=True):
            st.session_state['chat_history'] = []
            st.session_state['last_question'] = None
            st.session_state['graph_button'] = None
            st.session_state['summary_button'] = None
            st.session_state['file_path'] = None
            st.rerun()

        st.markdown('<div class="section-header">Data Source</div>', unsafe_allow_html=True)
        option = st.selectbox("**Select an option:**", ["Fnbm Risk-Data", "temp_data2"])
        selected_database = option
        st.session_state['selected_database'] = selected_database
        
        try:
            rate_data = pd.read_csv("H:/app/annualised_rate_data.csv")
            mds_ccor=pd.read_excel("H:/app/MDS_CCOR.xlsx")

            #### Data Preview with better styling
            st.markdown('<div class="section-header">Data Preview</div>', unsafe_allow_html=True)
            if selected_database == "Fnbm Risk-Data":
                with st.expander('Rate Data'):
                    st.markdown('<div class="data-preview-container">', unsafe_allow_html=True)
                    st.dataframe(rate_data.head(), use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                with st.expander('MDS Data'):
                    st.markdown('<div class="data-preview-container">', unsafe_allow_html=True)
                    st.dataframe(mds_ccor.head(), use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            elif selected_database == "temp_data2":
                pass
            else:
                pass
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")

    if 'username' in st.session_state:
        st.markdown(f"<div style='text-align: right; color: #666; margin-bottom: 15px;'>Hello {st.session_state['username']}</div>", unsafe_allow_html=True)

    # Display chat history with improved styling
    if 'chat_history' in st.session_state:
        for entry in st.session_state['chat_history']:
            # User message
            st.markdown(f"""
            <div class="user-message">
                <strong>Question:</strong><br>
                {entry['question']}
            </div>
            """, unsafe_allow_html=True)
            
            # Assistant response
            st.markdown(f"""
            <div class="assistant-message">
                <strong>SQL Query:</strong>
                <pre>{entry['sql_query']}</pre>
                <strong>Result:</strong>
                <div style="margin-top: 10px;">
                    {entry['result'] if isinstance(entry['result'], str) else "Data retrieved successfully"}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Display chart if available
            if entry['chart']:
                st.plotly_chart(entry['chart'], use_container_width=True, key=f"{uuid.uuid1()}")
            
            # Display summary if available
            if entry['summary']:
                st.markdown(f"""
                <div class="summary-box">
                    <strong>Analysis Summary:</strong><br>
                    {entry['summary']}
                </div>
                """, unsafe_allow_html=True)

    ### Placeholders - keeping original logic
    user_input_placeholder = st.empty()
    sql_response_placeholder = st.empty()
    data_response_placeholder = st.empty()
    error_placeholder = st.empty()
    graph_placeholder = st.empty()
    summary_placeholder = st.empty()
    button_column = st.columns([1, 1, 1, 1, 2, 2, 2])

    with button_column[5]:
        if st.session_state['graph_button'] is not None:
            if st.button('Graph', use_container_width=True):
                result_df = pd.read_csv(st.session_state.last_df, compression='zip')
                user_input = st.session_state.last_question

                rs = graph(user_input, result_df, graph_placeholder)

                if rs:
                    st.session_state['graph_button'] = None
                    st.rerun()
                else:
                    pass

                time.sleep(1)
                st.rerun()

    # Add a divider before the chat input
    st.markdown("<hr style='margin: 20px 0;'>", unsafe_allow_html=True)
    
    # Chat input with better styling
    st.markdown('<div class="section-header">Ask a Question</div>', unsafe_allow_html=True)
    user_input = st.chat_input("Enter your question about the data")

    if user_input is not None:
        # Show a spinner while processing
        with st.spinner("Processing your query..."):
            sql_response = process_question(user_input, user_input_placeholder, error_placeholder, selected_database)

            if sql_response is not None:
                run_database(sql_response, sql_response_placeholder, selected_database, data_response_placeholder, summary_placeholder, error_placeholder)
                result_df = pd.read_csv(st.session_state.last_df, compression='zip')
                generate_summary(user_input, result_df, summary_placeholder)
            else:
                pass

chatbot()
