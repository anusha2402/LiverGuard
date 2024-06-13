import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Initialize Spark session
spark = SparkSession.builder \
    .appName("LiverDiseasePrediction") \
    .getOrCreate()

# Load the dataset
@st.cache_data
def load_data():
    path = 'Indian_Liver_Patients_Dataset.csv'
    sdf = spark.read.csv(path, header=True, inferSchema=True)
    return sdf.toPandas()  # Convert to Pandas DataFrame

liver_df = load_data()

# Data preprocessing
liver_df['Gender_Male'] = liver_df['Gender'].apply(lambda x: 1 if x == 'Male' else 0)
liver_df['Gender_Female'] = liver_df['Gender'].apply(lambda x: 1 if x == 'Female' else 0)
liver_df['Albumin_and_Globulin_Ratio'].fillna(liver_df['Albumin_and_Globulin_Ratio'].mean(), inplace=True)
X = liver_df.drop(['Gender', 'Dataset'], axis=1)
y = liver_df['Dataset']

# Train the logistic regression model
logreg = LogisticRegression()
logreg.fit(X, y)

# Function to predict liver condition percentage
def predict_liver_condition_probability(age, total_bilirubin, direct_bilirubin, alkaline_phosphotase, alamine_aminotransferase,
                                        aspartate_aminotransferase, total_proteins, albumin, albumin_globulin_ratio, gender):
    input_data = pd.DataFrame({
        'Age': [age],
        'Total_Bilirubin': [total_bilirubin],
        'Direct_Bilirubin': [direct_bilirubin],
        'Alkaline_Phosphotase': [alkaline_phosphotase],
        'Alamine_Aminotransferase': [alamine_aminotransferase],
        'Aspartate_Aminotransferase': [aspartate_aminotransferase],
        'Total_Protiens': [total_proteins],
        'Albumin': [albumin],
        'Albumin_and_Globulin_Ratio': [albumin_globulin_ratio]
    })
    if gender.lower() == 'male':
        input_data['Gender_Male'] = 1
        input_data['Gender_Female'] = 0
    else:
        input_data['Gender_Male'] = 0
        input_data['Gender_Female'] = 1
    input_data = input_data[X.columns]
    probabilities = logreg.predict_proba(input_data)
    return probabilities[0][1] * 100

# Streamlit app
st.title('Liver Disease Prediction')
st.sidebar.title("Navigation")
option = st.sidebar.selectbox("Choose a page", ["EDA", "Prediction"])

if option == "EDA":
    st.header('Exploratory Data Analysis')
    st.write("Data Overview:")
    if st.checkbox('Show raw data'):
        st.write(liver_df.head())
    
    st.write("Summary statistics:")
    st.write(liver_df.describe())
    
    st.write("Distribution of Dataset (Liver Disease vs No Liver Disease):")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=liver_df, x='Dataset', ax=ax1)
    st.pyplot(fig1)
    
    st.write("Distribution of Gender:")
    fig2, ax2 = plt.subplots()
    sns.countplot(data=liver_df, x='Gender', ax=ax2)
    st.pyplot(fig2)
    
    st.write("Age vs Gender by Dataset:")
    fig3 = sns.catplot(x="Age", y="Gender", hue="Dataset", data=liver_df)
    st.pyplot(fig3)
    
    st.write("Correlation heatmap:")
    numeric_df = liver_df.drop(columns=['Gender'])
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax4)
    st.pyplot(fig4)

elif option == "Prediction":
    st.header('Predict Liver Disease')
    
    with st.form("prediction_form"):
        age = st.number_input('Age', min_value=0, max_value=100, value=0)
        total_bilirubin = st.number_input('Total Bilirubin', min_value=0.0, value=0.0)
        direct_bilirubin = st.number_input('Direct Bilirubin', min_value=0.0, value=0.0)
        alkaline_phosphotase = st.number_input('Alkaline Phosphotase', min_value=0, value=0)
        alamine_aminotransferase = st.number_input('Alamine Aminotransferase', min_value=0, value=0)
        aspartate_aminotransferase = st.number_input('Aspartate Aminotransferase', min_value=0, value=0)
        total_proteins = st.number_input('Total Proteins', min_value=0.0, value=0.0)
        albumin = st.number_input('Albumin', min_value=0.0, value=0.0)
        albumin_globulin_ratio = st.number_input('Albumin and Globulin Ratio', min_value=0.0, value=0.0)
        gender = st.selectbox('Gender', options=['Male', 'Female'])
        
        submitted = st.form_submit_button("Predict")
        if submitted:
            probability = predict_liver_condition_probability(age, total_bilirubin, direct_bilirubin, alkaline_phosphotase, alamine_aminotransferase,
                                                              aspartate_aminotransferase, total_proteins, albumin, albumin_globulin_ratio, gender)
            st.write(f'Predicted Percentage of Having Liver Disease: {probability:.2f}%')
