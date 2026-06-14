import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import sqlite3
import time
from datetime import datetime
import base64

st.set_page_config(page_title="DSS Dashboard", layout="wide")

with open("static/background.jpg", "rb") as f:
    bg_img = base64.b64encode(f.read()).decode()

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: linear-gradient(
            rgba(0,0,0,0.6),
            rgba(0,0,0,0.6)
        ),
        url("data:image/jpeg;base64,{bg_img}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    #MainMenu, footer, header {{
        visibility: hidden;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

def init_feedback_db():
    conn = sqlite3.connect('feedback.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            prediction TEXT,
            rating INTEGER,
            comments TEXT,
            user_age INTEGER
        )
    ''')
    try:
        c.execute('ALTER TABLE feedback ADD COLUMN user_gender TEXT')
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()

def save_feedback(prediction, rating, comments, age, gender):
    conn = sqlite3.connect('feedback.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO feedback (timestamp, prediction, rating, comments, user_age, user_gender)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (datetime.now().isoformat(), prediction, rating, comments, age, gender))
    conn.commit()
    conn.close()

init_feedback_db()

@st.cache_resource
def load_model():
    try:
        return joblib.load('decision_support_model.joblib')
    except Exception as e:
        st.error("Model not found. Please verify the file path.")
        st.stop()

model = load_model()

st.title("VADER: The Digital Wellbeing DSS")
st.markdown("Decision Support System for Behavioral Screen-Time Analysis")
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Demographics & Sessions")
    
    subcol1, subcol2 = st.columns(2)
    with subcol1:
        age = st.number_input("Age", min_value=10, max_value=100, value=22, step=1)
    with subcol2:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        
    session_all_per_day = st.slider("Daily App Sessions", min_value=1, max_value=100, value=10)
    category_count_unique = st.slider("Unique Categories Viewed", min_value=1, max_value=200, value=15)

with col2:
    st.subheader("Daytime Watch (Minutes)")
    morning_wt = st.number_input("Morning (06:00 - 11:59)", min_value=0, max_value=400, value=30)
    noon_wt = st.number_input("Noon (12:00 - 14:59)", min_value=0, max_value=400, value=20)
    afternoon_wt = st.number_input("Afternoon (15:00 - 17:59)", min_value=0, max_value=400, value=45)

with col3:
    st.subheader("Evening & Late Night (Minutes)")
    evening_wt = st.number_input("Evening (18:00 - 23:59)", min_value=0, max_value=500, value=60)
    midnight_wt_total = st.number_input("Late Night (00:00 - 05:59)", min_value=0, max_value=500, value=0)
    
    st.write("") 
    
    if st.button("Generate Diagnostic Report", type="primary", width="stretch"):
        st.session_state['show_report'] = True

if st.session_state.get('show_report', False):
    st.divider()
    
    total_watch_time = morning_wt + noon_wt + afternoon_wt + evening_wt + midnight_wt_total
    
    midnight_ratio = midnight_wt_total / (total_watch_time + 1)
    
    input_data = pd.DataFrame([[
        morning_wt, 
        noon_wt, 
        afternoon_wt, 
        evening_wt, 
        midnight_wt_total, 
        total_watch_time, 
        session_all_per_day, 
        category_count_unique, 
        age,
        midnight_ratio
    ]], columns=[
        'morning_wt', 'noon_wt', 'afternoon_wt', 'evening_wt', 
        'midnight_wt_total', 'total_wt', 'session_all_per_day', 
        'category_count_unique', 'age', 'midnight_ratio'
    ])
    
    start_time = time.time()
    
    # Pure Machine Learning Inference
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    classes = list(model.classes_)
    
    latency_ms = (time.time() - start_time) * 1000
    
    st.subheader(f"Diagnostic Result: {str(prediction).upper()}")
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Watch Time", f"{total_watch_time} min")
    kpi2.metric("Daily Sessions", f"{session_all_per_day}")
    kpi3.metric("Late Night Watch", f"{midnight_wt_total} min", delta="High Risk" if midnight_wt_total > 60 else "Normal", delta_color="inverse")
    kpi4.metric("Content Breadth", f"{category_count_unique} topics")

    st.write("---")

    chart_col, notes_col = st.columns([2, 1])
    
    with chart_col:
        st.markdown("**Confidence Probability Matrix**")
        prob_df = pd.DataFrame({'Category': classes, 'Probability': [p * 100 for p in probabilities]})
        prob_df = prob_df.sort_values(by='Probability', ascending=True)
        
        fig = px.bar(prob_df, x='Probability', y='Category', orientation='h', text_auto='.1f')
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, showticklabels=False, title="", range=[0, 100]),
            yaxis=dict(title=""),
            margin=dict(l=0, r=0, t=10, b=0),
            height=250
        )
        fig.update_traces(textfont_size=14, textangle=0, textposition="outside", cliponaxis=False)
        st.plotly_chart(fig, width="stretch", config={'displayModeBar': False})
        
        st.caption(f"⏱️ Model Inference Latency: {latency_ms:.2f} ms")
        
    with notes_col:
        st.markdown("**Clinical Notes**")
        if midnight_wt_total > 60:
            st.error("High Late-Night Watch Time. Primary indicator of behavioral addiction.")
        if session_all_per_day > 30:
            st.warning("High Session Count. Pattern indicates compulsive checking.")
        if midnight_wt_total <= 60 and session_all_per_day <= 30:
            st.success("Telemetry indicates standard behavioral patterns.")

    st.write("---")
    st.markdown("**Help Us Improve! 🌱**")
    
    with st.form("feedback_form", clear_on_submit=True):
        user_rating = st.radio(
            "Accuracy Rating:",
            options=["1 ⭐️", "2 ⭐️", "3 ⭐️", "4 ⭐️", "5 ⭐️"],
            horizontal=True
        )
        comments = st.text_area("Corrections? (Optional)")
        submitted = st.form_submit_button("Submit Feedback")
        
        if submitted:
            db_rating = int(user_rating[0])
            save_feedback(prediction, db_rating, comments, age, gender)
            st.success(f"✅ Securely logged.")

    st.write("---")
    with st.expander("🛠️ Admin: View Validation Database"):
        admin_password = st.text_input("Enter Admin Credentials", type="password")
        if admin_password == "VADER4LYFE":
            try:
                conn = sqlite3.connect('feedback.db')
                feedback_data = pd.read_sql_query("SELECT * FROM feedback", conn)
                st.dataframe(feedback_data, width="stretch")
                conn.close()
            except Exception as e:
                st.error("Database error.")
        elif admin_password != "":
            st.error("Unauthorized access.")