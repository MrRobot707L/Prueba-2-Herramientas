import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os
import streamlit as st

# carga de datos
if os.path.exists('data/procesado.pkl'):
    df = pd.read_pickle('data/procesado.pkl')
else:
    st.error("error")
    st.stop()
# Crear etiquetas y limpiar
df['es_premium'] = (df['quality'] >= 6).astype(float)
X = df.drop(columns=['quality', 'es_premium']).values.astype(np.float32)
y = df['es_premium'].values.reshape(-1, 1).astype(np.float32)
# Normalización 
mean = X.mean(axis=0)
std = X.std(axis=0)
std[std == 0] = 1.0 
X_escalado = (X - mean) / std
# Crear Tensores y DataLoader
X_tensor = torch.tensor(X_escalado)
y_tensor = torch.tensor(y)
dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
# definicion del modelo MLPVinos
class MLPVinos(nn.Module):
    def __init__(self,input_size,hidden_sizes,output_size,dropout_rate=0.2):
        super(MLPVinos,self).__init__()
        layers=[]
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))

        for i in range(len(hidden_sizes)-1):
            layers.append(nn.Linear(hidden_sizes[i],hidden_sizes[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        layers.append(nn.Linear(hidden_sizes[-1],output_size))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)
#configuracion del modelo
input_size = 12
hidden_sizes = [64, 32]
output_size = 1
model=MLPVinos(input_size,hidden_sizes,output_size)
criterion=nn.BCELoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)

def train_model(model,train_loader,criterion,optimizer,num_epochs=100):
    train_losses=[]
    progreso_texto = st.empty()     # Reserva un espacio en blanco en la pantalla
    barra_progreso = st.progress(0) # Crea la barra vacía (al 0%)
    for epoch in range(num_epochs):
        model.train()
        running_loss=0.0
        for inputs,targets in train_loader:
            optimizer.zero_grad()
            outputs=model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()*inputs.size(0)
        epoch_loss=running_loss/len(train_loader.dataset)
        train_losses.append(epoch_loss)
        if (epoch+1)%10==0:
            progreso_texto.text(f'Entrenando... Época {epoch+1}/{num_epochs} - Error: {epoch_loss:.4f}')
            barra_progreso.progress((epoch + 1) / num_epochs)
    return train_losses

st.title("Entrenamiento del Modelo de Predicción de Vinos Premium")
if st.button("Iniciar Entrenamiento"):
    with st.spinner("Entrenando red neuronal. Por favor, espere..."):
        losses = train_model(model, train_loader, criterion, optimizer, num_epochs=100)
    st.success("Entrenamiento completado")
    st.subheader("Curva de Aprendizaje")
    st.line_chart(losses)