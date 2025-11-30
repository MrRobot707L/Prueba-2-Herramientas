import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os
import streamlit as st

def show():
    st.title("Entrenamiento del Sommelier AI")
    st.write("Esta sección permite entrenar el modelo de predicción de vinos premium utilizando una red neuronal.")
    pkl_path='data/procesado.pkl'
    if not os.path.exists(pkl_path):
        st.error("El archivo de datos procesados no se encuentra. Por favor, asegúrese de que 'data/procesado.pkl' existe.")
        return
    try:
        df = pd.read_pickle(pkl_path)
    except Exception as e:
        st.error(f"Error al cargar los datos: {e}")
        st.stop()
    feature_names = [
        'Fixed Acidity', 'Volatile Acidity', 'Citric Acid', 'Residual Sugar', 
        'Chlorides', 'Free Sulfur Dioxide', 'Total Sulfur Dioxide', 
        'Density', 'pH', 'Sulphates', 'Alcohol'
    ]
    # Mostrar información de los datos
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"Datos cargados: {len(df)} registros")
    if 'target' in df.columns:
        y_raw=df['target'].values
    elif 'es_premium' in df.columns:
        y_raw=df['es_premium'].values
    else:
        st.info("Generando etiquetas a partir de la calidad del vino.")
        y_raw = (df['quality'] >= 7).astype(float).values
    # Preparación de datos
    cols_to_drop = ['quality', 'target', 'es_premium']
    X_raw = df.drop(columns=cols_to_drop, errors='ignore').values.astype(np.float32)
    y_raw = y_raw.reshape(-1, 1).astype(np.float32)
    # Normalización
    mean = X_raw.mean(axis=0)
    std = X_raw.std(axis=0)
    std[std == 0] = 1.0
    X_escalado = (X_raw - mean) / std
    # Convertir a tensores pytorch
    X_tensor = torch.tensor(X_escalado)
    y_tensor = torch.tensor(y_raw)
    dataset = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    #modelo neuronal
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
    real_input_size = X_tensor.shape[1]
    # Mostrar número de características
    with col2:
        st.success(f"Número de características: {real_input_size}")
    hidden_sizes = [64, 32]
    output_size = 1
    model=MLPVinos(real_input_size,hidden_sizes,output_size)
    criterion=nn.BCELoss()
    optimizer=optim.Adam(model.parameters(),lr=0.001)
    # Función de entrenamiento con barra de progreso
# --- REEMPLAZA TU FUNCIÓN train_model POR ESTA VERSIÓN 2D RÁPIDA ---
# --- REEMPLAZA TU FUNCIÓN train_model POR ESTA ---
    def train_model(model, train_loader, criterion, optimizer, num_epochs=100):
        train_losses = []
        
        # 1. Crear los espacios vacíos en la interfaz
        col_progreso1, col_progreso2 = st.columns([1, 3])
        with col_progreso1:
            progreso_texto = st.empty()
            barra_progreso = st.progress(0)
        
        with col_progreso2:
            st.write("📉 **Evolución del Error (Loss) en Tiempo Real:**")
            grafico_placeholder = st.empty() # <--- AQUÍ OCURRE LA MAGIA
        
        # 2. Bucle de entrenamiento
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
            
            epoch_loss = running_loss / len(train_loader.dataset)
            train_losses.append(epoch_loss)
            
            # 3. ACTUALIZACIÓN EN TIEMPO REAL
            # En cada época, redibujamos el gráfico con los nuevos datos
            grafico_placeholder.line_chart(train_losses)
            
            # Actualizamos la barra y el texto
            progreso_texto.text(f'Época {epoch+1}/{num_epochs}\nLoss: {epoch_loss:.4f}')
            barra_progreso.progress((epoch + 1) / num_epochs)
                
        return train_losses
# --- 5. INTERFAZ DE USUARIO ---
    if 'modelo_entrenado' not in st.session_state:
        st.session_state.modelo_entrenado = None
    # Botón para iniciar el entrenamiento
    if st.button("Iniciar Entrenamiento"):
        with st.spinner("Entrenando el modelo, por favor espere..."):
           losses = train_model(model, train_loader, criterion, optimizer, num_epochs=150) 
           st.session_state.modelo_entrenado = model
           st.session_state.stats_mean = mean
           st.session_state.stats_std = std
           st.session_state.train_losses = losses
        st.success("¡Modelo entrenado exitosamente!")
    if st.session_state.modelo_entrenado is not None:
        st.subheader("Curva de Aprendizaje")
        st.line_chart(st.session_state.train_losses)
        st.caption("Una curva descendente indica que el modelo está aprendiendo.")
        st.markdown("---")
        st.header("Probador de Vinos (Inferencia)")
        with st.form("form_prediccion"):
            col_inp1, col_inp2, col_inp3 = st.columns(3)
            input_data = []
            for i, feature in enumerate(feature_names):
                default_val = float(st.session_state.stats_mean[i])
                if i % 3 == 0:
                    with col_inp1:
                        val = st.number_input(f"{feature}", value=default_val, format="%.4f")
                elif i % 3 == 1:
                    with col_inp2:
                        val = st.number_input(f"{feature}", value=default_val, format="%.4f")
                else:
                    with col_inp3:
                        val = st.number_input(f"{feature}", value=default_val, format="%.4f")
                input_data.append(val)
            submit_val = st.form_submit_button("Predecir")
        if submit_val:
            user_input = np.array(input_data, dtype=np.float32)
            user_input_norm = (user_input - st.session_state.stats_mean) / st.session_state.stats_std
            input_tensor = torch.tensor(user_input_norm).unsqueeze(0)
            model_eval = st.session_state.modelo_entrenado
            model_eval.eval()
            with torch.no_grad():
                prediction = model_eval(input_tensor)
                probabilidad = prediction.item()
            st.markdown("### Resultado:")
            col_res1, col_res2 = st.columns([1, 3])
            with col_res1:
                if probabilidad >= 0.5:
                    st.markdown("""
                                div style='background-color: #d4edda; padding: 20px; border-radius: 10px; text-align: center;'>
                                <h1 style='color: #155724; margin:0;'>🏆 ¡ES PREMIUM! 🌟</h1>
                                <p style='color: #155724; font-size: 20px; margin:0;'>Excelente calidad detectada</p>
                                </div>
                                """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                                <div style='background-color: #f8d7da; padding: 20px; border-radius: 10px; text-align: center;'>
                                <h1 style='color: #721c24; margin:0;'>🍷 Vino No Premium</h1>
                                <p style='color: #721c24; font-size: 20px; margin:0;'>Calidad estándar detectada</p>
                                </div>
                                """, unsafe_allow_html=True)
            with col_res2:
                if probabilidad >= 0.5:
                    st.success(f"**¡ES PREMIUM!**  (Prob: {probabilidad*100:.2f}%)")
                else:
                    st.error(f"**Es Estándar.**  (Prob: {probabilidad*100:.2f}%)")
if __name__ == "__main__":
    show()
                