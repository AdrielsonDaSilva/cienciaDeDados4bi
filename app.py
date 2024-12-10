import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from treinarModelo import TreinarModelo
import seaborn as sns

data = pd.read_csv('diabetes_data.csv')

treinarModelo = TreinarModelo()
model = treinarModelo.train_model(data)

st.title("Predição do Nível de Glicose no Sangue")


peso = st.number_input("Peso (kg)", min_value=0.0, max_value=200.0, value=0.0)
altura = st.number_input("Altura (cm)", min_value=0, max_value=250, value=0)
horas_de_sono = st.number_input("Horas de Sono", min_value=0, max_value=24, value=0)

if st.button("Prever Nível de Glicose"):
    input_data = np.array([[peso, altura, horas_de_sono]])
    predicao_glicose = model.predict(input_data)
    st.success(f"O nível previsto de glicose no sangue é: {predicao_glicose[0]:.2f} mg/dL")

    st.subheader("Gráficos de Relação - Regressão Linear", divider= True)

    plt.figure(figsize=(10, 5))
    plt.scatter(peso, predicao_glicose[0], color='blue', alpha=0.5, s=1000)
    plt.title('Peso vs Nível de Glicose')
    plt.xlabel('Peso (kg)')
    plt.ylabel('Nível de Glicose (mg/dL)')
    plt.grid()
    st.pyplot(plt)


    plt.figure(figsize=(10, 5))
    plt.scatter(altura, predicao_glicose[0], color='green', alpha=0.5, s=1000)
    plt.title('Altura vs Nível de Glicose')
    plt.xlabel('Altura (cm)')
    plt.ylabel('Nível de Glicose (mg/dL)')
    plt.grid()
    st.pyplot(plt)


    plt.figure(figsize=(10, 5))
    plt.scatter(horas_de_sono, predicao_glicose[0], color='red', alpha=0.5, s=1000)
    plt.title('Horas de Sono vs Nível de Glicose')
    plt.xlabel('Horas de Sono')
    plt.ylabel('Nível de Glicose (mg/dL)')
    plt.grid()
    st.pyplot(plt)