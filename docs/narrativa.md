# Narrativa del Proyecto

## Contexto y Motivación

La industria vitivinícola enfrenta desafíos para garantizar la calidad del producto final, ya que esta depende de diversos factores físico-químicos que deben ser controlados durante el proceso de elaboración. Evaluar la calidad del vino tradicionalmente requiere paneles sensoriales humanos, lo que implica tiempo, costo y variabilidad subjetiva.

En este proyecto se utiliza el dataset Wine Quality del repositorio UCI Machine Learning Repository, el cual contiene mediciones físico-químicas de vinos portugueses de la variedad Vinho Verde. Este dataset es ampliamente utilizado para demostrar conceptos de análisis, visualización y modelos predictivos en ciencia de datos.

El propósito del proyecto es construir una aplicación web interactiva que permita explorar estos datos y aplicar un modelo simple de aprendizaje automático que apoye el análisis de la calidad del vino.

## Descripción del Dataset

El dataset utilizado corresponde al publicado por Cortez et al. (2009) y consiste en dos conjuntos: vino tinto y vino blanco. Para este proyecto se selecciona uno de ellos (ej. vino tinto), compuesto por 4.898 muestras, cada una con 11 variables físico-químicas (features) y una variable objetivo (“quality”), evaluada en una escala del 0 al 10.

1) Variables físico-químicas (features):
2) Fixed acidity
3) Volatile acidity
4) Citric acid
5) Residual sugar
6) Chlorides
7) Free sulfur dioxide
8) Total sulfur dioxide
9) Density
10) pH
11) Sulphates
12) Alcohol
13) Variable objetivo (target):
14) quality: puntuación de calidad dada por catadores expertos.

Este dataset permite abordar el problema tanto como regresión (predecir un valor numérico de calidad) como clasificación (agrupar vinos según su nivel de calidad).

## Problema a Resolver

El problema principal consiste en predecir la calidad del vino a partir de sus características físico-químicas, utilizando modelos simples de aprendizaje automático.
Adicionalmente, se busca facilitar la exploración, análisis y visualización de los datos, con el fin de identificar patrones relevantes y comprender la relación entre las variables.

## Objetivos del Proyecto
Objetivo General

Desarrollar una aplicación web interactiva en Streamlit que permita analizar, visualizar y modelar la calidad del vino basado en sus propiedades físico-químicas.

Objetivos Específicos

Construir una interfaz modular con múltiples pantallas para:

Presentar la idea general del proyecto

Cargar y explorar el dataset

Visualizar relaciones entre variables

Aplicar transformaciones y limpieza de datos

Entrenar un modelo simple de ML (clasificación o regresión)

Mostrar métricas básicas del desempeño del modelo

Integrar herramientas y conceptos revisados durante el semestre, tales como:

Manipulación de datos con pandas

Visualización con matplotlib

Modelos de ML con scikit-learn

Navegación y estructura modular en Streamlit

Control de versiones mediante Git y GitHub

Documentación y estructura de proyecto profesional

## Justificación

Este proyecto refleja un caso realista de aplicación de ciencia de datos en la industria, donde las empresas buscan apoyo analítico para mejorar sus procesos. El análisis automatizado de la calidad del vino permite:

Reducir subjetividad humana

Detectar vinos de baja o alta calidad

Identificar qué variables influyen más en la calidad

Proporcionar herramientas accesibles para toma de decisiones

Además, el dataset es ideal para un contexto académico, ya que su estructura es limpia, de tamaño manejable y relevante para tareas clásicas de visualización, análisis exploratorio y modelado.