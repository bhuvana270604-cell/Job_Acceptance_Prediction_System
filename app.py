import streamlit as st
import pandas as pd
import joblib

# Load model and feature columns
model = joblib.load("job_acceptance_random_forest.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# Load dataset
df = pd.read_csv("Job_Acceptance_Cleaned_Dataset.csv")

# Page configuration
st.set_page_config(
    page_title="Job Acceptance Prediction System",
    page_icon="💼",
    layout="wide"
)

# Sidebar
st.sidebar.title("💼 Project Menu")

st.sidebar.write("### About Project")

st.sidebar.info(
    "This project analyzes candidate placement data "
    "and predicts job placement using Machine Learning."
)

st.sidebar.write("### Technologies")

st.sidebar.write(
    "🐍 Python\n"
    "📊 Pandas\n"
    "🤖 Machine Learning\n"
    "🌲 Random Forest\n"
    "📈 Streamlit"
)
# Title
st.title("💼 Job Acceptance Prediction System")
st.write(
    "Analyze candidate placement data and predict job acceptance."
)

# Dashboard KPIs
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Candidates", len(df))

with col2:
    placement_rate = (df["Placement_Status"].eq("Placed").mean() * 100)
    st.metric("Placement Rate", f"{placement_rate:.2f}%")

with col3:
    avg_interview = df["Interview_Score"].mean()
    st.metric("Average Interview Score", f"{avg_interview:.2f}")

with col4:
    avg_skills = df["Skills_Match_Score"].mean()
    st.metric("Average Skills Match", f"{avg_skills:.2f}")

# Dataset preview
st.subheader("📊 Candidate Dataset")

st.dataframe(
    df.head(20),
    use_container_width=True
)

# Placement distribution
st.subheader("📈 Placement Status Distribution")

placement_counts = df["Placement_Status"].value_counts()

st.bar_chart(placement_counts)  

# Company Tier vs Placement

st.subheader("🏢 Company Tier vs Placement")

company_tier_data = pd.crosstab(
    df["Company_Tier"],
    df["Placement_Status"]
)

st.bar_chart(company_tier_data)  


# Skills Match vs Placement

st.subheader("🎯 Skills Match vs Placement")

skills_match_data = pd.crosstab(
    df["Skills_Match"],
    df["Placement_Status"]
)

st.bar_chart(skills_match_data) 

# Competition Level vs Placement

st.subheader("🏆 Competition Level vs Placement")

competition_data = pd.crosstab(
    df["Competition_Level"],
    df["Placement_Status"]
)

st.bar_chart(competition_data) 

# Interview Score vs Placement

st.subheader("📊 Interview Score vs Placement")

interview_data = df.groupby("Placement_Status")["Interview_Score"].mean()

st.bar_chart(interview_data)  

# Academic Performance vs Placement

st.subheader("🎓 Academic Performance vs Placement")

academic_data = df.groupby("Placement_Status")["Academic_Performance"].mean()

st.bar_chart(academic_data)



# Candidate Prediction 
st.divider()
st.subheader("🤖 Candidate Job Prediction")

st.write("Enter candidate details to predict placement status.")     



col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=60, value=25)
    
    academic = st.number_input(
        "Academic Performance",
        min_value=0.0,
        max_value=100.0,
        value=70.0
    )
    
    technical = st.number_input(
        "Technical Score",
        min_value=0.0,
        max_value=100.0,
        value=70.0
    )
    
    communication = st.number_input(
        "Communication Score",
        min_value=0.0,
        max_value=100.0,
        value=70.0
    )

with col2:
    aptitude = st.number_input(
        "Aptitude Score",
        min_value=0.0,
        max_value=100.0,
        value=70.0
    )
    
    coding = st.number_input(
        "Coding Score",
        min_value=0.0,
        max_value=100.0,
        value=70.0
    )
    
    interview = st.number_input(
        "Interview Score",
        min_value=0.0,
        max_value=100.0,
        value=70.0
    )
    
    skills_score = st.number_input(
        "Skills Match Score",
        min_value=0.0,
        max_value=100.0,
        value=70.0
    )  


# Additional Candidate Details

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    degree = st.selectbox(
        "Degree",
        sorted(df["Degree"].dropna().unique())
    )

    internship = st.number_input(
        "Internship Count",
        min_value=0,
        max_value=10,
        value=1
    )

    projects = st.number_input(
        "Projects Count",
        min_value=0,
        max_value=10,
        value=2
    )

    certification = st.selectbox(
        "Certification",
        ["No", "Yes"]
    )

    job_willingness = st.selectbox(
        "Job Willingness",
        ["Not Willing", "Willing"]
    )

with col2:
    job_relevance = st.selectbox(
        "Job Relevance",
        ["Not Relevant", "Relevant"]
    )

    company_tier = st.selectbox(
        "Company Tier",
        sorted(df["Company_Tier"].dropna().unique())
    )

    skills_match = st.selectbox(
        "Skills Match",
        ["Not Matched", "Matched"]
    )

    competition = st.selectbox(
        "Competition Level",
        ["Low", "Medium", "High"]
    )

    relocation = st.selectbox(
        "Relocation Required",
        ["Not Required", "Required"]
    )

    company_offer = st.selectbox(
        "Company Offer",
        ["No", "Yes"]
    )

    final_willingness = st.selectbox(
        "Final Willingness",
        ["Not Willing", "Willing"]
    )   

    offered_salary = st.number_input("Offered Salary", min_value=0.0, value=25000.0)
    expected_salary = st.number_input("Expected Salary", min_value=0.0, value=30000.0)
    


distance = st.number_input(
    "Distance From Home",
    min_value=0.0,
    value=5.0
)

job_offers = st.number_input(
    "Job Offers Count",
    min_value=0,
    max_value=10,
    value=1
)   


# Derived Placement Score

placement_score = (
    academic
    + technical
    + communication
    + aptitude
    + coding
    + interview
    + skills_score
) / 7


# Prediction Button

if st.button("🔮 Predict Job Placement"):

    input_data = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Academic_Performance": [academic],
        "Technical_Score": [technical],
        "Communication_Score": [communication],
        "Degree": [degree],
        "Aptitude_Score": [aptitude],
        "Coding_Score": [coding],
        "Interview_Score": [interview],
        "Skills_Match_Score": [skills_score],
        "Internship": [internship],
        "Certification": [certification],
        "Projects_Count": [projects],
        "Job_Willingness": [job_willingness],
        "Job_Relevance": [job_relevance],
        "Distance_From_Home": [distance],
        "Expected_Salary": [expected_salary],
        "Company_Tier": [company_tier],
        "Skills_Match": [skills_match],
        "Competition_Level": [competition],
        "Relocation_Required": [relocation],
        "Offered_Salary": [offered_salary],
        "Company_Offer": [company_offer],
        "Job_Offers_Count": [job_offers],
        "Final_Willingness": [final_willingness],
        "Academic_Performance_Band": ["Medium"],
        "Interview_Performance_Category": ["Medium"],
        "Skills_Match_Level": ["Medium"],
        "Placement_Probability_Score": [placement_score]
    })

    input_encoded = pd.get_dummies(input_data, drop_first=True)

    input_encoded = input_encoded.reindex(
        columns=feature_columns,
        fill_value=0
    )

    prediction = model.predict(input_encoded)[0]

    prediction_probability = model.predict_proba(input_encoded)[0]

    confidence = prediction_probability[prediction] * 100

    st.write(f"📊 Model Confidence: {confidence:.2f}%")

    if prediction == 1:
        st.success("🎉 Prediction: Placed")
        st.info("This candidate is predicted to be suitable for job placement.")
    else:
        st.error("❌ Prediction: Not Placed")
        st.warning("This candidate may need improvement in some placement-related factors.")