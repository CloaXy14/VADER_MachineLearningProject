# 🌿 VADER: Video Addiction Diagnostic & Evaluation Routine

**A Machine Learning Decision Support System (DSS) for Digital Wellbeing.**

VADER is an end-to-end, classical machine learning application designed to classify problematic short-video usage patterns from proxy behavioral telemetry. Instead of merely tracking screen time, this system analyzes nuanced habits—such as midnight watch spikes, daily session frequency, and content breadth—to output actionable, probability-based diagnostics.

### 🚀 Key Engineering Highlights
* **Advanced Leakage Prevention:** Implements `GroupShuffleSplit` to isolate repeated monthly observations across 10,000+ unique users, ensuring zero row-level data leakage during model evaluation.
* **Imbalance Optimization:** Utilizes an optimized Random Forest ensemble with dynamic `class_weight='balanced'` handling to successfully capture minority-class boundaries (Mildly Addicted users).
* **Sub-100ms Inference:** Achieves ~9ms real-time inference latency using Streamlit resource caching (`@st.cache_resource`), making the deployment fast and scalable.
* **Human-in-the-Loop (HITL):** Features a SQLite-backed feedback loop allowing users to rate and correct diagnostic outputs for future model retraining.

### 🛠️ Tech Stack
* **Machine Learning:** `scikit-learn`, `pandas`, `numpy`
* **Deployment:** `streamlit`
* **Database:** `sqlite3`
* **Language:** `Python 3.12`
