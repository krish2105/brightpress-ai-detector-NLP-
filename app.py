# app.py — BrightPress AI Text Detector
# Run with: streamlit run app.py --server.fileWatcherType none

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time
import re
import plotly.graph_objects as go
from datetime import datetime

MODEL_DIR = "detector"
MAX_LENGTH = 200

# Set page config FIRST
st.set_page_config(
    page_title="BrightPress AI Text Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State for History
if 'history' not in st.session_state:
    st.session_state.history = []

# ----------------- UI / THEME CONFIGURATION -----------------
st.sidebar.title("⚙️ Settings")
theme = st.sidebar.radio("Theme Mode", ["Dark", "Light"], index=0)

if theme == "Dark":
    bg_color = "#0f172a"
    card_bg = "rgba(30, 41, 59, 0.7)"
    text_color = "#f8fafc"
    accent_color = "#3b82f6"
    border_color = "rgba(255,255,255,0.1)"
    plotly_font_color = "#ffffff"
else:
    bg_color = "#f8fafc"
    card_bg = "rgba(255, 255, 255, 0.8)"
    text_color = "#0f172a"
    accent_color = "#2563eb"
    border_color = "rgba(0,0,0,0.1)"
    plotly_font_color = "#000000"

# Inject custom CSS
custom_css = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    .stApp {{
        background-color: {bg_color};
        color: {text_color};
        font-family: 'Inter', sans-serif;
        transition: background-color 0.3s ease;
    }}
    
    /* Global Text Fix for Native Dark Mode Override */
    p, span, label, div.stMarkdown, div.stText {{
        color: {text_color};
    }}
    
    .glass-card {{
        background: {card_bg};
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid {border_color};
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    .glass-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.15);
    }}
    
    /* Custom Alerts */
    .custom-warning {{
        background-color: rgba(234, 179, 8, 0.2);
        border-left: 5px solid #eab308;
        padding: 16px;
        border-radius: 4px;
        color: {text_color};
        margin-bottom: 24px;
    }}
    
    h1, h2, h3, h4, h5, h6 {{
        color: {text_color} !important;
        font-family: 'Inter', sans-serif;
    }}
    
    .gradient-text {{
        background: linear-gradient(90deg, {accent_color}, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.5rem;
    }}
    
    .stButton>button {{
        background: linear-gradient(90deg, {accent_color}, #8b5cf6);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }}
    .stButton>button:hover {{
        transform: scale(1.02);
        box-shadow: 0 0 15px rgba(139, 92, 246, 0.5);
    }}
    
    .stTextArea textarea {{
        background-color: {card_bg} !important;
        color: {text_color} !important;
        border-radius: 8px;
        border: 1px solid {border_color};
    }}
    .stTextArea textarea:focus {{
        border-color: {accent_color};
        box-shadow: 0 0 0 1px {accent_color};
    }}
    .stTextArea textarea::placeholder {{
        color: {text_color};
        opacity: 0.5;
    }}
    
    /* Radio buttons */
    div[role="radiogroup"] label {{
        color: {text_color} !important;
    }}
    div[role="radiogroup"] div {{
        color: {text_color} !important;
    }}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ----------------- BACKEND LOGIC -----------------
@st.cache_resource
def load_detector():
    """Load the saved Round 2 detector model and tokenizer."""
    import os
    if os.path.exists(MODEL_DIR):
        model_path = MODEL_DIR
    else:
        # Fallback for Streamlit Cloud demo
        model_path = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
    model.eval()
    return tokenizer, model

def analyze_statistics(text):
    """Calculates basic linguistic statistics."""
    words = re.findall(r'\b\w+\b', text)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s for s in sentences if s.strip()]
    
    word_count = len(words)
    sentence_count = len(sentences)
    avg_word_length = sum(len(w) for w in words) / word_count if word_count > 0 else 0
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
    unique_words = len(set(w.lower() for w in words))
    lexical_diversity = unique_words / word_count if word_count > 0 else 0
    
    return {
        "Word Count": word_count,
        "Sentence Count": sentence_count,
        "Avg Word Length": round(avg_word_length, 2),
        "Avg Sentence Length (Words)": round(avg_sentence_length, 1),
        "Lexical Diversity": f"{lexical_diversity:.0%}"
    }

def classify_text(text, tokenizer, model):
    """Classify text as HUMAN-WRITTEN or AI-GENERATED."""
    encoded = tokenizer(
        [text],
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    with torch.no_grad():
        logits = model(**encoded).logits
        probs = torch.softmax(logits, dim=-1)[0]

    human_prob = float(probs[0])
    ai_prob = float(probs[1])

    if ai_prob > human_prob:
        verdict = "AI-GENERATED"
        confidence = ai_prob
        color = "#ef4444" # Red
        icon = "🤖"
    else:
        verdict = "HUMAN-WRITTEN"
        confidence = human_prob
        color = "#10b981" # Green
        icon = "✍️"

    return verdict, confidence, human_prob, ai_prob, color, icon

def create_gauge_chart(value, title, color):
    """Creates a beautiful Plotly gauge chart."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value * 100,
        number = {'suffix': "%"},
        title = {'text': title, 'font': {'size': 20, 'color': plotly_font_color}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': plotly_font_color},
            'bar': {'color': color},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': border_color,
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=250,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    return fig

# ----------------- SIDEBAR HISTORY -----------------
st.sidebar.markdown("---")
st.sidebar.title("📚 Audit Log")
if not st.session_state.history:
    st.sidebar.info("No scans yet in this session.")
else:
    for item in reversed(st.session_state.history):
        st.sidebar.markdown(
            f"""
            <div style="background:{card_bg}; padding: 10px; border-radius: 8px; margin-bottom: 8px; border: 1px solid {border_color};">
                <small style="color:{item['color']}; font-weight:bold;">{item['icon']} {item['verdict']} ({item['confidence']:.0%})</small><br/>
                <span style="font-size:0.8rem; opacity:0.8;">{item['snippet']}</span>
            </div>
            """, unsafe_allow_html=True
        )
    if st.sidebar.button("Clear Log"):
        st.session_state.history = []
        st.rerun()

# ----------------- APP LAYOUT -----------------

# Header
st.markdown('<h1 class="gradient-text">✨ BrightPress AI Text Detector</h1>', unsafe_allow_html=True)
st.markdown(
    f"""
    <div class="glass-card">
        <p style="font-size: 1.1rem; opacity: 0.9;">
        This prototype classifies text as <strong>HUMAN-WRITTEN</strong> or <strong>AI-GENERATED</strong>.
        Trained through a GAN-style adversarial loop: <em>detector → attack → retrain → fresh attack</em>.
        </p>
    </div>
    """, unsafe_allow_html=True
)

st.markdown("""
    <div class="custom-warning">
        ⚠️ <strong>Responsible-Use Policy:</strong> This tool has a measured false-accusation rate. Never use it as sole evidence against a person.
    </div>
""", unsafe_allow_html=True)

# Mode Selection
app_mode = st.radio("Select Tool Mode:", ["Single Scan (Standard)", "Adversarial Face-Off (Compare Two Texts)"], horizontal=True)

try:
    tokenizer, model = load_detector()
except Exception as e:
    st.error("The detector model could not be loaded. Please ensure the Round 2 model was saved in the 'detector' folder.")
    st.stop()

if app_mode == "Single Scan (Standard)":
    # Main Columns
    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        st.markdown('<h3>Input Text</h3>', unsafe_allow_html=True)
        text_input = st.text_area(
            "Paste text here:",
            height=300,
            placeholder="Paste an essay, opinion piece, reader letter, or review here to analyse..."
        )
        analyse = st.button("🔍 Analyse Text")

    with col2:
        st.markdown('<h3>Results Dashboard</h3>', unsafe_allow_html=True)
        results_container = st.empty()
        
        if analyse:
            if not text_input.strip():
                results_container.error("Please paste some text before analysing.")
            else:
                with st.spinner("Neural network analyzing linguistic patterns..."):
                    time.sleep(0.5) 
                    verdict, confidence, human_prob, ai_prob, color, icon = classify_text(text_input, tokenizer, model)
                    stats = analyze_statistics(text_input)
                    
                    # Save to history
                    st.session_state.history.append({
                        "verdict": verdict,
                        "confidence": confidence,
                        "color": color,
                        "icon": icon,
                        "snippet": text_input[:50] + "..."
                    })
                    
                # Show Results Card
                results_container.markdown(
                    f"""
                    <div class="glass-card" style="border-left: 5px solid {color}; text-align:center;">
                        <h2 style="margin-top:0; color:{color} !important;">{icon} {verdict}</h2>
                        <p style="font-size:1.2rem; margin-bottom: 5px;">Overall Confidence: <strong>{confidence:.1%}</strong></p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                # Show Gauge Chart
                st.plotly_chart(create_gauge_chart(ai_prob, "AI Probability", "#ef4444"), use_container_width=True)
                
                # Show Stats Expander
                with st.expander("📊 View Linguistic Statistics"):
                    cols = st.columns(3)
                    idx = 0
                    for k, v in stats.items():
                        cols[idx % 3].metric(k, str(v))
                        idx += 1
                        
                # Download Report
                report_content = f"""BrightPress Forensic NLP Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
--------------------------------------------------
Verdict: {verdict}
Confidence: {confidence:.2%}
AI Probability: {ai_prob:.2%}
Human Probability: {human_prob:.2%}

Linguistic Statistics:
{str(stats)}

--------------------------------------------------
Text Sample Analyzed:
"{text_input[:500]}..."

--------------------------------------------------
WARNING: This tool has a measured false-accusation rate. 
Do not use as sole evidence against an individual.
"""
                st.download_button("📥 Download Forensic Report", data=report_content, file_name=f"BrightPress_Report_{int(time.time())}.txt")
                
        else:
            results_container.markdown(
                f"""
                <div class="glass-card" style="text-align: center; opacity: 0.7;">
                    <h1 style="font-size: 3rem; margin:0;">⏳</h1>
                    <p>Waiting for text input...</p>
                </div>
                """, 
                unsafe_allow_html=True
            )

else:
    # Adversarial Face-Off Mode
    st.markdown("### 🤺 Adversarial Face-Off")
    st.markdown("Paste a known human text and a suspected AI text to see how the model compares them.")
    
    col_a, col_b = st.columns(2)
    with col_a:
        text_a = st.text_area("Text A", height=250, placeholder="Paste first text here...")
    with col_b:
        text_b = st.text_area("Text B", height=250, placeholder="Paste second text here...")
        
    if st.button("⚔️ Initiate Face-Off Analysis"):
        if text_a.strip() and text_b.strip():
            with st.spinner("Analyzing both texts simultaneously..."):
                time.sleep(0.5)
                v_a, c_a, h_a, ai_a, col_a_color, icon_a = classify_text(text_a, tokenizer, model)
                v_b, c_b, h_b, ai_b, col_b_color, icon_b = classify_text(text_b, tokenizer, model)
            
            c1, c2 = st.columns(2)
            c1.markdown(
                f"""
                <div class="glass-card" style="border-top: 5px solid {col_a_color}; text-align:center;">
                    <h3 style="color:{col_a_color} !important;">{icon_a} Text A: {v_a}</h3>
                    <p>AI Probability: {ai_a:.1%}</p>
                </div>
                """, unsafe_allow_html=True
            )
            c1.plotly_chart(create_gauge_chart(ai_a, "AI Score", col_a_color), use_container_width=True)
            
            c2.markdown(
                f"""
                <div class="glass-card" style="border-top: 5px solid {col_b_color}; text-align:center;">
                    <h3 style="color:{col_b_color} !important;">{icon_b} Text B: {v_b}</h3>
                    <p>AI Probability: {ai_b:.1%}</p>
                </div>
                """, unsafe_allow_html=True
            )
            c2.plotly_chart(create_gauge_chart(ai_b, "AI Score", col_b_color), use_container_width=True)
        else:
            st.error("Please provide both Text A and Text B to compare.")

# Footer
st.markdown("---")
st.markdown(
    f"""
    <div style="text-align: center; opacity: 0.6; font-size: 0.9rem;">
        BrightPress Academy | NLP Engineering Team | GAN-Style Adversarial Build
    </div>
    """,
    unsafe_allow_html=True
)
