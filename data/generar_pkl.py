"""
Script para generar el archivo procesado.pkl
Este archivo preprocesa el dataset de Wine Quality para el modelo MLP.

Ejecutar desde la carpeta data/:
    python generar_pkl.py
"""

import pandas as pd

# Cargar datos desde el archivo CSV local
df = pd.read_csv('dataset.csv', sep=';')

# Crear variable target para clasificacion binaria
# 1 = Premium (calidad >= 7)
# 0 = Estándar (calidad < 7)
df['target'] = df['quality'].apply(lambda x: 1 if x >= 7 else 0)

# Guardar como pkl
df.to_pickle('procesado.pkl')

print(f"Archivo procesado.pkl generado exitosamente")
print(f"Total de registros: {len(df)}")
print(f"Vinos Premium: {df['target'].sum()}")
print(f"Vinos Estándar: {len(df) - df['target'].sum()}")
