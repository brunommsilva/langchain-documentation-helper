#!/usr/bin/env python3

import streamlit as st
from dotenv import load_dotenv

from classes.agent import Jarvis

load_dotenv()

if "agent" not in st.session_state:
    st.session_state.agent = Jarvis()

if "messages" not in st.session_state:
    st.session_state.messages = []

st.header("Chat with Jarvis")
clear_button = st.button("Clear Chat")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Enter your question:")

if clear_button:
    st.session_state.messages = []
    st.session_state.agent.clear_messages()
    st.rerun()

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("Getting answer from Jarvis..."):
        response = st.session_state.agent.answer_question(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
