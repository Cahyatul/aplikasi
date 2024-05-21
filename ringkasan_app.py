import streamlit as st
from PyPDF2 import PdfFileReader
from docx import Document
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import heapq
import re
import spacy

nltk.download('punkt')
nltk.download('stopwords')

# Load SpaCy model
nlp = spacy.load('en_core_web_sm')

def read_pdf(file):
    pdf_reader = PdfFileReader(file)
    text = ""
    for page_num in range(pdf_reader.getNumPages()):
        text += pdf_reader.getPage(page_num).extract_text()
    return text

def read_docx(file):
    doc = Document(file)
    text = [p.text for p in doc.paragraphs]
    return "\n".join(text)

def text_rank_summarize(text, top_n=5):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.isalnum() and word not in stop_words]

    word_freq = {}
    for word in words:
        if word not in word_freq:
            word_freq[word] = 1
        else:
            word_freq[word] += 1

    max_freq = max(word_freq.values())
    for word in word_freq:
        word_freq[word] = word_freq[word] / max_freq

    sentence_scores = {}
    for sent in sentences:
        for word in word_tokenize(sent.lower()):
            if word in word_freq:
                if sent not in sentence_scores:
                    sentence_scores[sent] = word_freq[word]
                else:
                    sentence_scores[sent] += word_freq[word]

    summary_sentences = heapq.nlargest(top_n, sentence_scores, key=sentence_scores.get)
    summary = ' '.join(summary_sentences)
    return summary

st.title("Article Summarization App")

uploaded_file = st.file_uploader("Choose a file (text, PDF, or DOCX)", type=["txt", "pdf", "docx"])

if uploaded_file:
    if uploaded_file.type == "text/plain":
        content = str(uploaded_file.read(), "utf-8")
    elif uploaded_file.type == "application/pdf":
        content = read_pdf(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        content = read_docx(uploaded_file)
    
    st.subheader("Original Content")
    st.text_area("Content", content, height=300)
    
    if st.button("Summarize"):
        summary = text_rank_summarize(content)
        st.subheader("Summary")
        st.write(summary)
