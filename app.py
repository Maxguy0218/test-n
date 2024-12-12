import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st

# Load the dataset
def load_data():
    data = pd.read_csv("/mnt/data/resume_dataset.csv")
    st.write("Columns in the dataset:", list(data.columns))  # Display column names
    return data

def preprocess_and_cluster(data):
    # Ensure the correct column is used for descriptions
    if 'Description' not in data.columns:
        st.error("The dataset does not contain a 'Description' column. Please check your dataset.")
        st.stop()

    # Preprocessing and feature extraction
    tfidf = TfidfVectorizer(stop_words='english', max_features=500)
    X = tfidf.fit_transform(data['Description'])

    # KMeans clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(X)

    # Assigning cluster labels to data
    data['Cluster'] = clusters

    return data, kmeans, tfidf

def train_knn(data, tfidf):
    X = tfidf.transform(data['Description'])
    y = data['Cluster']

    # Train a KNN classifier
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)
    return knn

def classify_record(knn, tfidf, input_text):
    input_features = tfidf.transform([input_text])
    prediction = knn.predict(input_features)[0]
    return prediction

# Streamlit App
st.title("Resume Module Classification App")

st.write("Dataset loaded successfully.")
data = load_data()

st.write("### Preview of Dataset")
st.dataframe(data.head())

data, kmeans, tfidf = preprocess_and_cluster(data)
knn = train_knn(data, tfidf)

st.write("### Clustering Complete")
st.write(data[['Description', 'Cluster']].head())

st.write("### Enter Resume Description")
input_text = st.text_area("Paste the description here:")

if st.button("Classify Module"):
    if input_text.strip():
        prediction = classify_record(knn, tfidf, input_text)
        cluster_names = {0: "Payroll", 1: "Absences", 2: "Compensation", 3: "Core HR"}
        st.write(f"The record belongs to the module: **{cluster_names[prediction]}**")
    else:
        st.write("Please enter a valid description.")
