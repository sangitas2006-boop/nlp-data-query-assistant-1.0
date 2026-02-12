import streamlit as st
import pandas as pd
import plotly.express as px
import json
import streamlit.components.v1 as components
import speech_recognition as sr
import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import tempfile



# Initialize session state safely (only once)
if "df" not in st.session_state:
    st.session_state.df = None

if "question_input" not in st.session_state:
    st.session_state.question_input = ""

if "whisper_model" not in st.session_state:
    st.session_state.whisper_model = whisper.load_model("base")

# ------------------ Voice Recording Function ------------------
def record_audio(duration=5, fs=16000):
    st.info("🎙 Recording... Speak now")

    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wav.write(temp_file.name, fs, audio)

    return temp_file.name
def transcribe_audio(file_path):
    model = st.session_state.whisper_model
    result = model.transcribe(file_path)
    return result["text"]

# ------------------ Voice Recording Functions ------------------

def record_audio(duration=5, fs=16000):
    st.info("🎙 Recording... Speak now")

    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wav.write(temp_file.name, fs, audio)

    return temp_file.name


def transcribe_audio(file_path):
    model = st.session_state.whisper_model
    result = model.transcribe(file_path)
    return result["text"]


# ------------------ Page Config ------------------
st.set_page_config(
    page_title="NLP Data Query Assistant",
    page_icon="📊",
    layout="wide"
)

# ------------------ GLOBAL STYLING ------------------
st.markdown("""
<style>

/* Import Modern Font */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
}

/* Beautiful Gradient Background */
body {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 40%, #1f2937 100%);
}

/* Remove top padding */
.block-container {
    padding-top: 2rem;
}

/* Header Styling */
h1 {
    font-size: 48px !important;
    font-weight: 700 !important;
    letter-spacing: -1px;
}

h2 {
    font-size: 32px !important;
    font-weight: 600 !important;
}

h3 {
    font-size: 22px !important;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    border-right: 1px solid rgba(255,255,255,0.05);
}

/* Section Card */
.section-card {
    background: rgba(30,41,59,0.7);
    backdrop-filter: blur(10px);
    padding: 30px;
    border-radius: 20px;
    margin-bottom: 35px;
    border: 1px solid rgba(255,255,255,0.05);
}

/* GLOW CARD */
.glow-card {
    box-shadow: 0 10px 30px rgba(0,0,0,0.5);
}

/* 3D Floating Buttons */
.stButton > button {
    background: linear-gradient(145deg, #6366f1, #4f46e5);
    color: white;
    border-radius: 12px;
    padding: 12px 20px;
    border: none;
    font-weight: 600;
    box-shadow: 0 6px 0 #3730a3,
                0 10px 20px rgba(0,0,0,0.4);
    transition: all 0.15s ease-in-out;
}

/* Button Hover */
.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 9px 0 #3730a3,
                0 15px 25px rgba(0,0,0,0.5);
}

/* Button Click Effect */
.stButton > button:active {
    transform: translateY(3px);
    box-shadow: 0 2px 0 #3730a3,
                0 5px 10px rgba(0,0,0,0.4);
}

/* Input Field Premium Look */
input[type="text"] {
    background: rgba(30,41,59,0.8) !important;
    border-radius: 14px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    padding: 12px !important;
    color: white !important;
    font-size: 16px !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: rgba(30,41,59,0.7);
    padding: 20px;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.05);
}

/* Sidebar Card */
.sidebar-card {
    background: rgba(30,41,59,0.6);
    padding: 18px;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.05);
}

/* Smooth Fade Animation */
@keyframes fadeUp {
    from {opacity:0; transform: translateY(15px);}
    to {opacity:1; transform: translateY(0);}
}

.section-card {
    animation: fadeUp 0.4s ease;
}

</style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.markdown("""
<div style="padding:30px 0 10px 0;">
    <h1 style="font-size:42px; font-weight:700; margin-bottom:5px;">
        📊 NLP Data Query Assistant
    </h1>
    <p style="color:#9ca3af; font-size:18px;">
        Upload your dataset. Ask questions in natural language. Visualize instantly.
    </p>
</div>
""", unsafe_allow_html=True)

# ------------------ Sidebar ------------------
with st.sidebar:
    st.markdown("## 🤖 AI Data Assistant")
    st.markdown("---")

    if st.session_state.df is not None:
        df = st.session_state.df

        st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
        st.markdown("### 📁 Dataset Status: 🟢 Loaded")
        st.markdown(f"Rows: {df.shape[0]}")
        st.markdown(f"Columns: {df.shape[1]}")
        st.markdown(f"Missing Values: {df.isnull().sum().sum()}")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
        st.markdown("### 📁 Dataset Status: 🔴 Not Loaded")
        st.markdown("Upload a dataset to begin.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.write("• Ask for averages")
    st.write("• Ask for counts")
    st.write("• Ask for top values")

# ------------------ File Upload ------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.header("Step 1: Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload CSV, Excel, or JSON",
    type=["csv", "xlsx", "json"]
)

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith(".json"):
        df = pd.read_json(uploaded_file)

    st.session_state.df = df

    st.success("Dataset loaded successfully!")

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())

    st.subheader("Preview")
    st.dataframe(df.head())

st.markdown('</div>', unsafe_allow_html=True)

# ------------------ Ask Section ------------------
st.markdown('<div class="section-card glow-card">', unsafe_allow_html=True)
st.header("Step 2: Ask Questions")

df = st.session_state.df
disabled_state = df is None

col_input, col_mic = st.columns([6, 1])

with col_mic:
    if st.button("🎤 Record"):
        audio_file = record_audio()
        text = transcribe_audio(audio_file)

        st.session_state.question_input = text
        st.success("Voice captured successfully!")

with col_input:
    query = st.text_input(
        "Type your question here:",
        key="question_input",
        placeholder="Example: Show average sales by region"
    )


if df is None:
    st.markdown("""
<style>

/* Modern Font */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

/* Animated Gradient Background */
body {
    background: linear-gradient(-45deg, #1e1b4b, #312e81, #0f172a, #4f46e5);
    background-size: 400% 400%;
    animation: gradientMove 12s ease infinite;
    color: white;
}

@keyframes gradientMove {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Floating Orbs */
body::before, body::after {
    content: "";
    position: fixed;
    width: 400px;
    height: 400px;
    border-radius: 50%;
    filter: blur(120px);
    z-index: -1;
    animation: float 15s infinite alternate ease-in-out;
}

body::before {
    background: #6366f1;
    top: -100px;
    left: -100px;
}

body::after {
    background: #ec4899;
    bottom: -120px;
    right: -120px;
}

@keyframes float {
    0% { transform: translateY(0px) }
    100% { transform: translateY(40px) }
}

/* Container spacing */
.block-container {
    padding-top: 2rem;
}

/* Section Card - Glass Effect */
.section-card {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(20px);
    padding: 30px;
    border-radius: 20px;
    margin-bottom: 35px;
    border: 1px solid rgba(255,255,255,0.15);
    box-shadow: 0 20px 40px rgba(0,0,0,0.4);
    animation: fadeUp 0.6s ease;
}

@keyframes fadeUp {
    from { opacity:0; transform: translateY(30px); }
    to { opacity:1; transform: translateY(0); }
}

/* Headers */
h1 {
    font-size: 48px !important;
    font-weight: 700 !important;
    letter-spacing: -1px;
}

h2 {
    font-size: 30px !important;
    font-weight: 600 !important;
}

/* 3D Glowing Buttons */
.stButton > button {
    background: linear-gradient(145deg, #6366f1, #ec4899);
    border: none;
    color: white;
    font-weight: 600;
    border-radius: 14px;
    padding: 12px 22px;
    box-shadow:
        0 8px 20px rgba(99,102,241,0.5),
        0 0 15px rgba(236,72,153,0.6);
    transition: all 0.25s ease;
}

.stButton > button:hover {
    transform: translateY(-4px) scale(1.03);
    box-shadow:
        0 12px 30px rgba(99,102,241,0.6),
        0 0 25px rgba(236,72,153,0.9);
}

.stButton > button:active {
    transform: translateY(2px);
}

/* Input Field - Glass */
input[type="text"] {
    background: rgba(255,255,255,0.1) !important;
    border-radius: 14px !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    padding: 12px !important;
    color: white !important;
    font-size: 16px !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(15,23,42,0.95);
}

/* Metrics */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.08);
    padding: 18px;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.1);
}

</style>
""", unsafe_allow_html=True)


# ------------------ NLP Logic (UNCHANGED) ------------------
def process_query(query, df):
    query = query.lower()
    cols = df.columns.tolist()
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

    def find_col(column_list, user_query):
        for c in column_list:
            if c.lower() in user_query:
                return c
        return None

    target_num = find_col(numeric_cols, query)
    target_cat = find_col(categorical_cols, query)

    if any(word in query for word in ["average", "mean", "avg"]):
        if target_num and target_cat:
            result = df.groupby(target_cat)[target_num].mean().reset_index()
            return result, "bar", target_cat, target_num
        elif target_num:
            val = df[target_num].mean()
            return pd.DataFrame({"Metric": [f"Average {target_num}"], "Value": [val]}), "metric", None, "Value"

    if "count" in query or "total" in query:
        if target_cat:
            result = df[target_cat].value_counts().reset_index()
            result.columns = [target_cat, "count"]
            return result, "bar", target_cat, "count"

    if "top" in query or "highest" in query:
        if target_num:
            import re
            match = re.search(r'\d+', query)
            n = int(match.group()) if match else 5
            result = df.sort_values(target_num, ascending=False).head(n)
            label_col = target_cat if target_cat else df.columns[0]
            return result, "bar", label_col, target_num

    return None, None, None, None

# ------------------ Query Execution (UNCHANGED) ------------------
analyze = st.button("🔎 Analyze Query")

if analyze:
    if df is None:
        st.warning("Please upload a dataset first.")
    elif not st.session_state.question_input:
        st.warning("Please enter a query.")
    else:
        result, chart_type, x, y = process_query(st.session_state.question_input, df)

        if result is not None:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("<h3>📈 Analysis Result</h3>", unsafe_allow_html=True)
            st.dataframe(result)

            if chart_type == "bar" and x is not None:
                fig = px.bar(result, x=x, y=y)
                st.plotly_chart(fig, use_container_width=True)
            elif chart_type == "bar":
                fig = px.bar(result, y=y)
                st.plotly_chart(fig, use_container_width=True)

            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("Could not understand the query. Try another phrasing.")

st.markdown('</div>', unsafe_allow_html=True)