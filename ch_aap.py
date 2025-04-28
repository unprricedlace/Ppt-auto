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

# Load custom CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title = "FnBM-BOT"

def chatbot():
    with st.sidebar:
        st.markdown("**AI-Powered Data Insights Tool**")
        if st.button("Clear Chat History", use_container_width=True):
            st.session_state['chat_history'] = []
            st.session_state['last_question'] = None
            st.session_state['graph_button'] = None
            st.session_state['summary_button'] = None
            st.session_state['file_path'] = None
            st.rerun()

        option = st.selectbox("**Select an option:**", ["Fnbm Risk-Data", "temp_data2"])
        selected_database = option
        st.session_state['selected_database'] = selected_database
        rate_data = pd.read_csv("H:/app/annualised_rate_data.csv")
        mds_ccor=pd.read_excel("H:/app/MDS_CCOR.xlsx")

        #### Data Preview
        st.markdown('## Data Preview:')
        if selected_database == "Fnbm Risk-Data":
            with st.popover('#### Rate Data:'):
                st.markdown('<div class="data-preview">', unsafe_allow_html=True)
                st.write(rate_data.head())
                st.markdown('</div>', unsafe_allow_html=True)
            with st.popover('#### MDS Data:'):
                st.markdown('<div class="data-preview">', unsafe_allow_html=True)
                st.write(mds_ccor.head())
                st.markdown('</div>', unsafe_allow_html=True)
        elif selected_database == "temp_data2":
            pass
        else:
            pass

    if 'username' in st.session_state:
        st.markdown(f"Hello {st.session_state['username']}")

    for entry in st.session_state['chat_history']:
        with st.chat_message(name='user', avatar='user'):
            st.write(entry['question'])
        with st.chat_message(name='assistant', avatar='assistant'):
            st.code(entry['sql_query'], language='sql')
            st.write(entry['result'])
            if entry['chart']:
                st.plotly_chart(entry['chart'], use_container_width=True, key=f"{uuid.uuid1()}")
            if entry['summary']:
                st.info(entry['summary'])

    ### Placeholders
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

    ### Main Content
    user_input = st.chat_input("Enter your question about the data")

    if user_input is not None:
        sql_response = process_question(user_input, user_input_placeholder, error_placeholder, selected_database)

        if sql_response is not None:
            run_database(sql_response, sql_response_placeholder, selected_database, data_response_placeholder, summary_placeholder, error_placeholder)
            result_df = pd.read_csv(st.session_state.last_df, compression='zip')
            generate_summary(user_input, result_df, summary_placeholder)
        else:
            pass

chatbot()
