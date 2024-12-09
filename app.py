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
plt.scatter(data['peso'], data['glicemia'], color='blue', alpha=0.5)
plt.title('Peso vs Nível de Glicose')
plt.xlabel('Peso (kg)')
plt.ylabel('Nível de Glicose (mg/dL)')
plt.grid()
st.pyplot(plt)


plt.figure(figsize=(10, 5))
plt.scatter(data['altura'], data['glicemia'], color='green', alpha=0.5)
plt.title('Altura vs Nível de Glicose')
plt.xlabel('Altura (cm)')
plt.ylabel('Nível de Glicose (mg/dL)')
plt.grid()
st.pyplot(plt)


plt.figure(figsize=(10, 5))
plt.scatter(data['horas_de_sono'], data['glicemia'], color='red', alpha=0.5)
plt.title('Horas de Sono vs Nível de Glicose')
plt.xlabel('Horas de Sono')
plt.ylabel('Nível de Glicose (mg/dL)')
plt.grid()
st.pyplot(plt)


plt.figure(figsize=(10, 5))
plt.hist(data['glicemia'], bins=20, color='purple', alpha=0.7)
plt.title('Distribuição do Nível de Glicose no Sangue')
plt.xlabel('Nível de Glicose (mg/dL)')
plt.ylabel('Frequência')
plt.grid(axis='y')
st.pyplot(plt)


plt.figure(figsize=(10, 5))
sns.boxplot(x='peso', y='glicemia', data=data)
plt.title('Box Plot do Nível de Glicose em Relação ao Peso')
plt.xlabel('Peso (kg)')
plt.ylabel('Nível de Glicose (mg/dL)')
plt.grid()
st.pyplot(plt)


bins = [0, 50, 70, 90, 110, 130, 150, 200]
labels = ['<50', '50-70', '70-90', '90-110', '110-130', '130-150', '>150']
data['weight_group'] = pd.cut(data['peso'], bins=bins, labels=labels)

mean_glucose_by_weight = data.groupby('weight_group')['glicemia'].mean().reset_index()


plt.figure(figsize=(10, 5))
plt.bar(mean_glucose_by_weight['weight_group'], mean_glucose_by_weight['glicemia'], color='orange', alpha=0.7)
plt.title('Média do Nível de Glicose por Grupo de Peso')
plt.xlabel('Grupo de Peso (kg)')
plt.ylabel('Média do Nível de Glicose (mg/dL)')
plt.xticks(rotation=45)
plt.grid(axis='y')
st.pyplot(plt)


bins_glucose = [0, 70, 100, 126, 200, 300]
labels_glucose = ['<70', '70-100', '100-126', '126-200', '>200']
data['glucose_group'] = pd.cut(data['glicemia'], bins=bins_glucose, labels=labels_glucose)


glucose_distribution = data['glucose_group'].value_counts()


plt.figure(figsize=(8, 8))
plt.pie(glucose_distribution, labels=glucose_distribution.index, autopct='%1.1f%%', startangle=140, colors=['lightblue', 'lightgreen', 'lightcoral', 'lightsalmon', 'lightyellow'])
plt.title('Distribuição do Nível de Glicose no Sangue')
plt.axis('equal') 
st.pyplot(plt)