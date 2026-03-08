import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="AI Academic Intelligence Platform", layout="wide")

# ------------------ GEMINI CONFIG ----------------
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("models/gemini-pro-latest")

# ------------------ AI INTERVENTION FUNCTION ----------------
def fallback_intervention_insight(row):
    reasons = []
    actions = []
    risks = []

    # --- Risk reasoning ---
    if row["Avg_Score"] < 50:
        reasons.append("overall academic performance is critically low")
        actions.append("daily remedial classes with core subject focus")
        risks.append("high probability of academic failure")

    if row["ABI"] < 90:
        reasons.append("high inconsistency across subjects")
        actions.append("balanced subject-wise mentoring plan")
        risks.append("uneven learning gaps may widen")

    if row["Lunch Type"].lower() == "free/reduced":
        reasons.append("possible socio-economic learning constraints")
        actions.append("school-supported academic resources")
        risks.append("external factors impacting performance")

    if row["Risk_Level"] == "High Risk":
        actions.append("parent-teacher counseling and weekly monitoring")
        risks.append("drop in confidence and engagement")

    # --- Safe defaults ---
    if not reasons:
        reasons.append("stable academic performance")
        actions.append("performance tracking and enrichment activities")
        risks.append("risk escalation unlikely")

    return f"""
### 📌 Risk Explanation
This student is identified at risk because **{' and '.join(reasons)}**.

### 🎯 Recommended Intervention
• {'; '.join(set(actions))}

### ⚠️ If No Action Is Taken
• {'; '.join(set(risks))}
"""
# ------------------ AI GENERATION FUNCTION ----------------
def generate_ai_intervention(row):
    prompt = f"""
You are an academic intervention expert.

Student Profile:
- Gender: {row['Gender']}
- Lunch Type: {row['Lunch Type']}
- Parental Education: {row['Parental Education']}
- Average Score: {round(row['Avg_Score'], 2)}
- Academic Balance Index (ABI): {round(row['ABI'], 2)}
- Risk Level: {row['Risk_Level']}

Tasks:
1. Explain why the student is at this risk level.
2. Recommend 2–3 targeted academic interventions.
3. Explain potential consequences if no action is taken.

Tone: clear, professional, student-support focused.
Format using headings and bullet points.
"""

    response = model.generate_content(prompt)
    return response.text

# ------------------ LOAD DATA --------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Students_Data.csv")
    df = df.reset_index(drop=True)
    df["Student_ID"] = df.index + 1
    return df

df = load_data()

# ------------------ FEATURE ENGINEERING ----------
df["Avg_Score"] = df[["Maths Score", "Reading Score", "Writing Score"]].mean(axis=1)
df["ABI"] = 100 - df[["Maths Score", "Reading Score", "Writing Score"]].std(axis=1)

# ------------------ RISK CLASSIFICATION ----------
def classify_risk(row):
    if row["Avg_Score"] < 50 or row["ABI"] < 90:
        return "High Risk"
    elif row["Avg_Score"] < 70 or row["ABI"] < 95:
        return "Medium Risk"
    else:
        return "Low Risk"

df["Risk_Level"] = df.apply(classify_risk, axis=1)

# ------------------ INTERVENTION ENGINE ----------
def intervention_plan(row):
    if row["Risk_Level"] == "High Risk":
        return "Immediate remedial classes, parental counseling, weekly monitoring"
    elif row["Risk_Level"] == "Medium Risk":
        return "Subject-specific tutoring and structured test preparation"
    else:
        return "Advanced enrichment and performance tracking"

df["Intervention"] = df.apply(intervention_plan, axis=1)

# ------------------- PREDICTION ----------------------
def predict_future_risk(row):
    if row["Risk_Level"] == "Medium Risk" and (row["Avg_Score"] < 65 or row["ABI"] < 93):
        return "High Risk (Predicted)"
    elif row["Risk_Level"] == "Low Risk" and (row["Avg_Score"] < 70 or row["ABI"] < 95):
        return "Medium Risk (Predicted)"
    else:
        return "Stable"

df["Predicted_Risk"] = df.apply(predict_future_risk, axis=1)

# ------------------ SIDEBAR ----------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Analysis", "Interventions","Prediction", "AI Summary"]
)
st.sidebar.markdown("---")
st.sidebar.subheader("Filters")

risk_filter = st.sidebar.multiselect(
    "Risk Level",
    options=df["Risk_Level"].unique(),
    default=df["Risk_Level"].unique()
)

gender_filter = st.sidebar.multiselect(
    "Gender",
    options=df["Gender"].unique(),
    default=df["Gender"].unique()
)

lunch_filter = st.sidebar.multiselect(
    "Lunch Type",
    options=df["Lunch Type"].unique(),
    default=df["Lunch Type"].unique()
)

pred_filter = st.sidebar.multiselect(
    "Predicted Risk",
    options=df["Predicted_Risk"].unique(),
    default=df["Predicted_Risk"].unique()
)

# ------------------ FILTER DATAFRAME ----------------------
filtered_df = df[
    (df["Risk_Level"].isin(risk_filter)) &
    (df["Gender"].isin(gender_filter)) &
    (df["Lunch Type"].isin(lunch_filter)) &
    (df["Predicted_Risk"].isin(pred_filter))
]

if filtered_df.empty:
    st.warning("⚠️ No students match the selected filters.")
    st.stop()

def export_csv(df):
    return df.to_csv(index=False).encode("utf-8")

# ================== OVERVIEW =====================
if page == "Overview":
    st.title("📊 Student Performance Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Average Score", round(filtered_df["Avg_Score"].mean(), 2))
    c2.metric("Average ABI", round(filtered_df["ABI"].mean(), 2))
    c3.metric("Total Students", len(filtered_df))
    c4.metric(
        "High-Risk Students (%)",
        f"{round((filtered_df['Risk_Level'] == 'High Risk').mean() * 100, 1)}%"
    )

    fig = px.histogram(filtered_df, x="Avg_Score", title="Average Score Distribution")
    st.plotly_chart(fig, use_container_width=True)

# ================== ANALYSIS =====================
elif page == "Analysis":
    st.title("📈 Academic & Socio-Economic Analysis")

    fig1 = px.box(
        filtered_df,
        x="Test Preparation Course",
        y="Avg_Score",
        title="Impact of Test Preparation Course"
    )
    st.plotly_chart(fig1, use_container_width=True)

    risk_counts = filtered_df["Risk_Level"].value_counts().reset_index()
    risk_counts.columns = ["Risk Level", "Students"]

    fig2 = px.bar(
        risk_counts,
        x="Risk Level",
        y="Students",
        color="Risk Level",
        title="Academic Risk Classification"
    )
    st.plotly_chart(fig2, use_container_width=True)

# ================== INTERVENTIONS =================
elif page == "Interventions":
    st.title("🎯 Academic Intervention Engine")

    st.markdown(
        "This module converts academic risk signals into "
        "**actionable intervention strategies** for leadership."
    )

    # KPIs
    c1, c2, c3 = st.columns(3)
    c1.metric("High Risk Students", (filtered_df["Risk_Level"] == "High Risk").sum())
    c2.metric("Medium Risk Students", (filtered_df["Risk_Level"] == "Medium Risk").sum())
    c3.metric("Low Risk Students", (filtered_df["Risk_Level"] == "Low Risk").sum())

    intervention_counts = (
        filtered_df["Intervention"]
        .value_counts()
        .rename_axis("Intervention Strategy")
        .reset_index(name="Students")
    )

    fig = px.bar(
        intervention_counts,
        x="Intervention Strategy",
        y="Students",
        color="Intervention Strategy",
        title="Recommended Academic Intervention Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 Student-Level Intervention Plan")
    st.dataframe(
        filtered_df[
            [
                "Student_ID",
                "Gender",
                "Lunch Type",
                "Parental Education",
                "Avg_Score",
                "ABI",
                "Risk_Level",
                "Intervention"
            ]
        ],
        use_container_width=True
    )

    st.download_button(
        label="⬇️ Download Intervention Report (CSV)",
        data=export_csv(
            filtered_df[
                [
                    "Student_ID",
                    "Gender",
                    "Lunch Type",
                    "Parental Education",
                    "Avg_Score",
                    "ABI",
                    "Risk_Level",
                    "Intervention"
                ]
            ]
        ),
        file_name="intervention_report.csv",
        mime="text/csv"
    )

    # 🔽 🔽 🔽 MOVE THIS INSIDE INTERVENTIONS 🔽 🔽 🔽
    st.markdown("---")
    st.subheader("🧠 AI-Powered Student Intervention Insight")

    selected_student = st.selectbox(
        "Select Student ID",
        filtered_df["Student_ID"].unique()
    )

    student_row = filtered_df[
        filtered_df["Student_ID"] == selected_student
    ].iloc[0]

    if st.button("Generate AI Intervention Insight"):
        with st.spinner("Analyzing student profile..."):
            try:
                insight = generate_ai_intervention(student_row)
                if "quota" in insight.lower():
                    insight = fallback_intervention_insight(student_row)
            except Exception:
                insight = fallback_intervention_insight(student_row)

        st.success("AI Intervention Insight Generated")
        st.markdown(insight)

# ================== PREDICTIONS ==================
elif page == "Prediction":
    st.title("🔮 Academic Risk Prediction Engine")

    st.markdown(
        "This module forecasts **future academic risk trajectories** "
        "using academic consistency and performance signals."
    )

    # KPIs
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Predicted High Risk",
        (filtered_df["Predicted_Risk"] == "High Risk (Predicted)").sum()
    )
    c2.metric(
        "Predicted Medium Risk",
        (filtered_df["Predicted_Risk"] == "Medium Risk (Predicted)").sum()
    )
    c3.metric(
        "Stable Students",
        (filtered_df["Predicted_Risk"] == "Stable").sum()
    )

    #  SAFE AGGREGATION
    pred_counts = (
        filtered_df["Predicted_Risk"]
        .value_counts()
        .rename_axis("Predicted Risk")
        .reset_index(name="Students")
    )

    fig = px.bar(
        pred_counts,
        x="Predicted Risk",
        y="Students",
        color="Predicted Risk",
        title="Predicted Academic Risk Distribution (Next Term)"
    )
    st.plotly_chart(fig, use_container_width=True)

    #  FULL STUDENT FORECAST TABLE
    st.subheader("📋 Student-Level Risk Forecast")
    st.dataframe(
        filtered_df[
            [
                "Student_ID",
                "Gender",
                "Lunch Type",
                "Avg_Score",
                "ABI",
                "Risk_Level",
                "Predicted_Risk",
                "Intervention"
            ]
        ],
        use_container_width=True
    )

    st.download_button(
        label="⬇️ Download Risk Prediction Report (CSV)",
        data=export_csv(
            filtered_df[
                [
                    "Student_ID",
                    "Gender",
                    "Lunch Type",
                    "Avg_Score",
                    "ABI",
                    "Risk_Level",
                    "Predicted_Risk",
                    "Intervention"
                ]
            ]
        ),
        file_name="risk_prediction_report.csv",
        mime="text/csv"
    )

# ================== AI SUMMARY ===================
elif page == "AI Summary":
    st.title("🧠 AI Executive Academic Intelligence")

    st.markdown(
        "AI-generated executive insights combining academic performance, "
        "risk analytics, and intervention outcomes."
    )

    if st.button("Generate AI Insights"):
        total_students = len(filtered_df)
        avg_score = filtered_df["Avg_Score"].mean()
        avg_abi = filtered_df["ABI"].mean()

        prep_effect = (
            filtered_df[filtered_df["Test Preparation Course"] == "completed"]["Avg_Score"].mean()
            - filtered_df[filtered_df["Test Preparation Course"] == "none"]["Avg_Score"].mean()
        )

        high_risk = (filtered_df["Risk_Level"] == "High Risk").sum()
        medium_risk = (filtered_df["Risk_Level"] == "Medium Risk").sum()

        prompt = f"""
You are an Academic Intelligence Consultant advising school leadership.

Key Intelligence:
- Total students: {total_students}
- High-risk students: {high_risk}
- Medium-risk students: {medium_risk}
- Average Score: {round(avg_score, 2)}
- Average ABI: {round(avg_abi, 2)}
- Test preparation uplift: {round(prep_effect, 2)} points

Tasks:
1. Interpret academic risk distribution.
2. Recommend tiered intervention strategies.
3. Suggest one policy-level decision.

Tone: strategic, executive, outcome-focused.
"""

        try:
            with st.spinner("Generating AI insights..."):
                response = model.generate_content(prompt)
            st.success("AI Insights Generated")
            st.markdown(response.text)

        except Exception:
            st.warning("Gemini quota reached. Showing simulated executive insights.")
            st.markdown("""
**Executive AI Summary (Simulated)**

• ~10% students require immediate academic intervention  
• Test preparation shows measurable performance uplift  
• ABI effectively highlights hidden inconsistency risks  

**Recommended Actions**
1. Prioritize high-risk students with targeted support  
3. Monitor ABI as an early-warning indicator
""")
