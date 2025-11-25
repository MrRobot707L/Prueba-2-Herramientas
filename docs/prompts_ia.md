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