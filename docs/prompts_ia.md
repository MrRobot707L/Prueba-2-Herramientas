# Prompts de IA utilizados

Este documento contiene los prompts y solicitudes realizadas a herramientas de Inteligencia Artificial (ChatGPT) durante el desarrollo del proyecto, tal como lo requiere la evaluación.

---

## Prompt 1
"Me puedes ayudar con la redacción de este texto para narrativa.md?”

**Respuesta:**  
ChatGPT generó la narrativa del proyecto basada en el dataset Wine Quality de UCI.

---

## Prompt 2
"¿Qué son las carpetas dentro de ui/?"

**Respuesta:**  
ChatGPT explicó la estructura modular de Streamlit y la función de cada archivo.

---

## Prompt 3
"Me puedes recopilar los prompts de este chat para la carpeta prompts_ia.md?"

**Respuesta:**  
ChatGPT generó los documentos completos para la carpeta /docs/.



---

## Prompt 4
"Actúa como un Senior Data Scientist y experto en Streamlit. Necesito reescribir el archivo ui/analisis_datos.py con un análisis profesional completo.

Contexto: Estamos preparando los datos para entrenar un Perceptrón Multicapa (Red Neuronal con PyTorch) para clasificar vinos de la "Viña CIA".

Genera el código completo para ui/analisis_datos.py que incluya lo siguiente:

Carga de Datos:

Usa la URL del repositorio UCI (Wine Quality Red).

Usa separador de punto y coma (;).

Cachea la función con @st.cache_data.

Sección 1: Exploración Inicial:

Muestra el head() del dataframe dentro de un st.expander.

Muestra estadísticas descriptivas (df.describe()).

Sección 2: Visualización de Distribuciones:

Genera un histograma o countplot de la variable quality (0-10) usando Seaborn.

Explica con st.markdown que las clases están desbalanceadas (hay mucho vino normal, poco excelente).

Sección 3: Transformación de Datos (DTS) - CRÍTICO PARA PYTORCH:

Crea una nueva columna target donde:

Si quality >= 7 -> 1 (Vino Premium).

Si quality < 7 -> 0 (Vino Estándar).

Muestra un gráfico de pastel o barras comparando la cantidad de vinos Premium vs Estándar resultante.

Nota importante: Agrega un texto explicando que esta binarización es necesaria para que el Perceptrón Multicapa aprenda a clasificar correctamente.

Sección 4: Análisis de Correlación:

Genera un Heatmap de correlación de Pearson.

Filtra y destaca cuáles son las 3 variables químicas que más se correlacionan con la calidad (generalmente Alcohol, Acidez Volátil y Sulfatos) para sugerir al usuario qué sliders mover en la predicción.

Estilo: Usa colores corporativos (rojos/vinos) para los gráficos."

**Respuesta:**
El agente de copilot generó el código completo para el módulo ui/analisis_datos.py, incluyendo la visualización avanzada con Seaborn, la matriz de correlación y la lógica de transformación de datos (DTS) necesaria para preparar el target del modelo de PyTorch.

**Intervención humana:** Se agregaron manualmente los colores corporativos a los gráficos y se revisó meticulosamente el procedimiento para verificar el correcto análisis de los datos.

---

# Nota
Este archivo será actualizado durante el desarrollo del proyecto con cada interacción relevante con IA.

## prompt 5
adaptame esta parte para un formato streamlit 
def train_model(model,train_loader,criterion,optimizer,epochs=100):
    train_losses=[]
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
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
    return train_losses

**Respuesta:**
# Definición de la función de entrenamiento
def train_model(model, train_loader, criterion, optimizer, num_epochs=100):
    train_losses = []
    
    # --- AGREGA ESTO AQUÍ (INICIALIZACIÓN) ---
    progreso_texto = st.empty()     # Reserva un espacio en blanco en la pantalla
    barra_progreso = st.progress(0) # Crea la barra vacía (al 0%)
    # -----------------------------------------
    
    for epoch in range(num_epochs): # Aquí empieza tu bucle
        model.train()
        running_loss = 0.0
        
        # ... (aquí va todo tu código de forward, backward, optimizer, etc.) ...
        
        # ... cálculo del loss ...

        # --- Y AQUÍ ACTUALIZAS LA BARRA (LO QUE PREGUNTASTE) ---
        if (epoch+1) % 10 == 0:
            # Actualiza el texto
            progreso_texto.text(f'Entrenando... Época {epoch+1}/{num_epochs} - Error: {epoch_loss:.4f}')
            # Mueve la barra azul (calcula el porcentaje matemático)
            barra_progreso.progress((epoch + 1) / num_epochs)
            
    return train_losses

## prompt 6
necesito agregarle una manera que el usuario ingrese datos para que el modelo prediga si es premiun o no

**Respuesta**
feature_names = [
        'Fixed Acidity', 'Volatile Acidity', 'Citric Acid', 'Residual Sugar', 
        'Chlorides', 'Free Sulfur Dioxide', 'Total Sulfur Dioxide', 
        'Density', 'pH', 'Sulphates', 'Alcohol'
    ]
with st.form("form_prediccion"):
            col_inp1, col_inp2, col_inp3 = st.columns(3)
            input_data = []
            
            # Crear inputs dinámicamente
            for i, feature in enumerate(feature_names):
                # Valor por defecto = la media (para que sea fácil probar)
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

            submit_val = st.form_submit_button("🔮 Predecir")

        # --- LO QUE TE FALTABA: LA LÓGICA DE PREDICCIÓN ---
        if submit_val:
            # 1. Convertir a numpy
            user_input = np.array(input_data, dtype=np.float32)
            
            # 2. NORMALIZAR (Usando las stats guardadas del entrenamiento)
            user_input_norm = (user_input - st.session_state.stats_mean) / st.session_state.stats_std
            
            # 3. Tensor y Predicción
            input_tensor = torch.tensor(user_input_norm).unsqueeze(0)
            
            model_eval = st.session_state.modelo_entrenado
            model_eval.eval()
            
            with torch.no_grad():
                prediction = model_eval(input_tensor)
                probabilidad = prediction.item()
            
            # 4. Mostrar resultado
            st.markdown("### Resultado:")
            col_res1, col_res2 = st.columns([1, 3])
            
            with col_res1:
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

