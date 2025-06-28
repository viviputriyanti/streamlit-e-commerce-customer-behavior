import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ------------------------
# Load Model dan Tools
# ------------------------
model = joblib.load("model_svm_linear.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder_membership.pkl")
df = pd.read_csv("E-commerce Customer Behavior.csv")
df.dropna(inplace=True)

# ------------------------
# Sidebar Navigasi
# ------------------------
menu = st.sidebar.radio("Pilih Menu", ["Dashboard", "Prediksi Kepuasan"])

# ------------------------
# MENU 1: DASHBOARD
# ------------------------
if menu == "Dashboard":
    st.title("📊 Dashboard Analisis Pelanggan")

    st.subheader("Distribusi Kepuasan Pelanggan")
    fig1 = px.histogram(df, x='Satisfaction Level', color='Satisfaction Level',
                        category_orders={'Satisfaction Level': [0, 1, 2]},
                        labels={'Satisfaction Level': 'Tingkat Kepuasan'})
    st.plotly_chart(fig1)

    st.subheader("Hubungan Rating dan Total Spend")
    fig2 = px.scatter(df, x='Average Rating', y='Total Spend',
                      color='Satisfaction Level', hover_data=['Membership Type'],
                      labels={'Average Rating': 'Rating', 'Total Spend': 'Total Belanja'})
    st.plotly_chart(fig2)

    st.subheader("Korelasi Antar Fitur Numerik")
    num_cols = ['Age', 'Total Spend', 'Items Purchased', 'Average Rating', 'Days Since Last Purchase']
    # st.dataframe(df[num_cols].corr())

# ------------------------
# MENU 2: PREDIKSI
# ------------------------
elif menu == "Prediksi Kepuasan":
    st.title("🧠 Prediksi Kepuasan Pelanggan")

    age = st.slider("Umur", 15, 80, 30)
    spend = st.number_input("Total Belanja", min_value=0.0)
    rating = st.slider("Rata-Rata Rating", 0.0, 5.0, 3.5)
    items = st.number_input("Jumlah Item Dibeli", min_value=0)
    days = st.number_input("Hari Sejak Pembelian Terakhir", min_value=0)

    membership = st.selectbox("Membership Type", df['Membership Type'].unique())
    gender = st.selectbox("Gender", df['Gender'].unique())

    if st.button("Prediksi Sekarang"):
        # Encode dan transform input
        encoded_membership = encoder.transform([membership])[0]
        encoded_gender = 1 if gender == 'Male' else 0

        X_input = pd.DataFrame([[
            age, spend, items, rating, days, encoded_membership, encoded_gender
        ]], columns=['Age', 'Total Spend', 'Items Purchased', 'Average Rating',
                     'Days Since Last Purchase', 'Membership Type', 'Gender'])

        X_scaled = scaler.transform(X_input)
        prediction = model.predict(X_scaled)[0]

        st.success(f"Hasil Prediksi: **{prediction}**")
