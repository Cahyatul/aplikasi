import streamlit as st
import requests

# Initialize the summarization pipeline
summarizer = pipeline("summarization")

# Title of the Streamlit app
st.title("Text Summarization App")

# Input text box
input_text = st.text_area("Enter text to summarize", height=200)

# Summarize button
if st.button("Summarize"):
    if input_text:
        # Generate summary
        summary = summarizer(input_text, max_length=130, min_length=30, do_sample=False)
        st.subheader("Summary")
        st.write(summary[0]['summary_text'])
    else:
        st.write("Please enter some text to summarize.")
