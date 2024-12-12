import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import streamlit as st

# Load the dataset
df = pd.read_csv("resume_dataset.csv")

# Step 1: Cluster the data
def cluster_data(df):
    features = df.drop(columns=["Name"])
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=4, random_state=42)
    df['Cluster'] = kmeans.fit_predict(scaled_features)

    cluster_names = {0: "Payroll", 1: "Absences", 2: "Compensation", 3: "Core HR"}
    df['Module'] = df['Cluster'].map(cluster_names)

    df.to_csv("clustered_resume_dataset.csv", index=False)
    return kmeans, scaler, cluster_names

# Step 2: Create the Streamlit app
def main():
    st.title("Resume Module Classification")

    kmeans, scaler, cluster_names = cluster_data(df)

    name = st.text_input("Name")
    years_of_experience = st.number_input("Years of Experience", min_value=0, max_value=50, step=1)
    skills = st.text_input("Skills")
    education = st.text_input("Education")
    previous_job_titles = st.text_input("Previous Job Titles")
    certifications = st.text_input("Certifications")
    project_management_experience = st.checkbox("Project Management Experience")
    hr_experience = st.checkbox("HR Experience")
    payroll_experience = st.checkbox("Payroll Experience")
    compensation_experience = st.checkbox("Compensation Experience")
    absences_management_experience = st.checkbox("Absences Management Experience")

    input_data = pd.DataFrame({
        "Years of Experience": [years_of_experience],
        "Skills": [skills],
        "Education": [education],
        "Previous Job Titles": [previous_job_titles],
        "Certifications": [certifications],
        "Project Management Experience": [project_management_experience],
        "HR Experience": [hr_experience],
        "Payroll Experience": [payroll_experience],
        "Compensation Experience": [compensation_experience],
        "Absences Management Experience": [absences_management_experience]
    })

    scaled_input_data = scaler.transform(input_data)
    cluster = kmeans.predict(scaled_input_data)[0]
    module = cluster_names[cluster]

    st.write(f"The resume belongs to the '{module}' module.")

if __name__ == "__main__":
    main()
