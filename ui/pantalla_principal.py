import streamlit as st

def show():
    st.title("Viñedo Viña CIA")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Narrativa del Caso")
        st.info(
            """
            **El Problema:**
            El Viñedo Viña CIA enfrenta un desafío crítico en su proceso de producción. 
            Actualmente, la certificación de calidad de las barricas se ha convertido en un 
            cuello de botella significativo, retrasando la distribución y afectando la eficiencia operativa.
            
            **La Solución:**
            Para abordar este problema, hemos desarrollado un "Sommelier Artificial". 
            Este sistema utiliza tecnologías avanzadas de Inteligencia Artificial, específicamente 
            redes neuronales, para clasificar la calidad de los vinos de manera rápida y precisa, 
            optimizando así el proceso de certificación.
            """
        )
    
    with col2:
        st.markdown("### Objetivo")
        st.write(
            """
            Clasificar la calidad de los vinos utilizando modelos de IA para apoyar la toma de decisiones 
            y mejorar la eficiencia en la certificación de barricas.
            """
        )