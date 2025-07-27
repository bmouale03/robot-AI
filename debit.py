# fichier: streamlit_app.py

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

st.title("Prédiction du débit en fonction de la pluie")

# Même dataset synthétique que précédemment
time_steps = 3
jours = 200
np.random.seed(42)
pluie = np.random.poisson(2, jours).astype(float)
debit = np.zeros(jours)
for i in range(3, jours):
    debit[i] = 0.3*pluie[i] + 0.5*pluie[i-1] + 0.2*pluie[i-2] + np.random.normal(0, 0.2)

scaler_pluie = MinMaxScaler()
scaler_debit = MinMaxScaler()
pluie_scaled = scaler_pluie.fit_transform(pluie.reshape(-1,1))
debit_scaled = scaler_debit.fit_transform(debit.reshape(-1,1))

def create_sequences(X, y, time_steps=3):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

X_train, y_train = create_sequences(pluie_scaled, debit_scaled, time_steps)

model = Sequential([
    LSTM(50, activation='relu', input_shape=(time_steps,1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)

# Saisie utilisateur
input_pluie = []
for i in range(time_steps):
    val = st.number_input(f"Précipitation jour {-time_steps + i + 1} (mm)", min_value=0.0, max_value=50.0, step=0.1, value=1.0)
    input_pluie.append(val)

if st.button("Prédire le débit"):
    pluie_array = np.array(input_pluie).reshape(-1,1)
    pluie_scaled_input = scaler_pluie.transform(pluie_array).reshape(1, time_steps, 1)
    pred_scaled = model.predict(pluie_scaled_input)
    pred = scaler_debit.inverse_transform(pred_scaled)[0][0]
    st.write(f"Débit prédit : **{pred:.2f}** unités")
