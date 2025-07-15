import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from fpdf import FPDF

# Configuración de la página
st.set_page_config(
    page_title="📊 Aprende Segmentación Jerárquica",
    page_icon="📊",
    layout="wide"
)

# Título principal
st.title("📊 Aprende Segmentación Jerárquica")

# Inicializar variables de sesión
if 'datos' not in st.session_state:
    st.session_state.datos = None
if 'variables_disponibles' not in st.session_state:
    st.session_state.variables_disponibles = []
if 'etapa' not in st.session_state:
    st.session_state.etapa = 0
if 'tab_actual' not in st.session_state:
    st.session_state.tab_actual = 0

# Función para cambiar de pestaña
def cambiar_tab(nueva_tab):
    st.session_state.tab_actual = nueva_tab
    st.session_state.etapa = nueva_tab
    st.rerun()

# Función para generar datos simulados
def generar_datos_simulados(n_obs, variables):
    np.random.seed(42)
    data = {}
    
    # Generar datos correlacionados para hacer más realista
    base_scores = np.random.normal(70, 15, n_obs)
    
    for var in variables:
        if var == "Matemática":
            data[var] = np.clip(base_scores + np.random.normal(0, 5, n_obs), 0, 100)
        elif var == "Lenguaje":
            data[var] = np.clip(base_scores + np.random.normal(0, 8, n_obs), 0, 100)
        elif var == "Ciencias":
            data[var] = np.clip(base_scores + np.random.normal(0, 6, n_obs), 0, 100)
    
    # Agregar nombres de estudiantes
    data['Estudiante'] = [f'Estudiante_{i+1}' for i in range(n_obs)]
    
    return pd.DataFrame(data)

# Función para generar dendrograma
def crear_dendrograma(datos, metrica, metodo_enlace):
    # Mapear nombres de métricas
    metricas_map = {
        'Euclidiana': 'euclidean',
        'Manhattan': 'manhattan',
        'Máxima': 'chebyshev'
    }
    
    # Mapear métodos de enlace
    metodos_map = {
        'Ward': 'ward',
        'Single': 'single',
        'Complete': 'complete',
        'Average': 'average'
    }
    
    # Preparar datos
    datos_numericos = datos.select_dtypes(include=[np.number])
    
    # Calcular linkage
    if metodo_enlace == 'Ward':
        linkage_matrix = linkage(datos_numericos, method='ward')
    else:
        distances = pdist(datos_numericos, metric=metricas_map[metrica])
        linkage_matrix = linkage(distances, method=metodos_map[metodo_enlace])
    
    return linkage_matrix

# Función para crear reporte PDF
def crear_reporte_pdf(datos, grupos, metrica, metodo_enlace):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    pdf.cell(200, 10, txt="Reporte de Segmentación Jerárquica", ln=1, align='C')
    pdf.ln(10)
    
    pdf.cell(200, 10, txt=f"Método de distancia: {metrica}", ln=1)
    pdf.cell(200, 10, txt=f"Método de enlace: {metodo_enlace}", ln=1)
    pdf.cell(200, 10, txt=f"Número de grupos formados: {len(np.unique(grupos))}", ln=1)
    
    return pdf.output(dest='S').encode('latin-1')

# Determinar qué pestaña mostrar
tab_index = st.session_state.tab_actual

# Crear pestañas con indicadores de estado
tab_names = [
    "1️⃣ Introducción" + (" ✅" if st.session_state.etapa > 0 else ""),
    "2️⃣ Ingreso de Datos" + (" ✅" if st.session_state.datos is not None else ""),
    "3️⃣ Visualizar Datos" + (" ✅" if hasattr(st.session_state, 'variables_analisis') else ""),
    "4️⃣ Análisis" + (" ✅" if hasattr(st.session_state, 'metrica_seleccionada') else ""),
    "5️⃣ Resultado"
]

# Interfaz de pestañas
tabs = st.tabs(tab_names)

# PESTAÑA 1: Introducción
with tabs[0]:
    st.header("¿Qué es la segmentación jerárquica?")
    st.write("La segmentación jerárquica es una técnica que agrupa elementos similares.")
    st.write("Se utiliza en áreas como la educación, la biología, y el análisis de mercado.")
    
    st.markdown("""
    - 🔹 Agrupa datos similares en clústeres.
    - 🔹 No necesita saber de antemano cuántos grupos hay.
    - 🔹 Se representa con un gráfico llamado dendrograma.
    """)
    
    if st.button("▶️ Continuar a los datos", key="ir_a_datos"):
        cambiar_tab(1)

# PESTAÑA 2: Elección de datos
with tabs[1]:
    st.header("¿Cómo quieres trabajar?")
    
    modo_datos = st.radio(
        "Selecciona el modo de ingreso de datos:",
        ["Simular datos aleatorios", "Subir un archivo CSV propio", "Insertar datos manualmente"],
        key="modo_datos"
    )
    
    if modo_datos == "Simular datos aleatorios":
        col1, col2 = st.columns(2)
        
        with col1:
            n_obs = st.number_input(
                "Número de estudiantes a generar:", 
                value=10, 
                min_value=5,
                key="n_obs"
            )
        
        with col2:
            vars_sim = st.multiselect(
                "Variables:",
                ["Matemática", "Lenguaje", "Ciencias"],
                default=["Matemática", "Lenguaje"],
                key="vars_sim"
            )
        
        if st.button("🎲 Generar datos simulados", key="generar"):
            if vars_sim:
                st.session_state.datos = generar_datos_simulados(n_obs, vars_sim)
                st.session_state.variables_disponibles = vars_sim
                st.success("¡Datos generados exitosamente!")
                st.dataframe(st.session_state.datos)
                # Botón para continuar
                if st.button("▶️ Continuar a visualización", key="continuar_despues_simular"):
                    cambiar_tab(2)
            else:
                st.error("Por favor selecciona al menos una variable.")
    
    elif modo_datos == "Subir un archivo CSV propio":
        uploaded_file = st.file_uploader("Sube tu archivo CSV", type=['csv'])
        header = st.checkbox("El archivo tiene encabezado", True)
        
        if uploaded_file is not None:
            try:
                if header:
                    st.session_state.datos = pd.read_csv(uploaded_file)
                else:
                    st.session_state.datos = pd.read_csv(uploaded_file, header=None)
                
                st.session_state.variables_disponibles = list(st.session_state.datos.select_dtypes(include=[np.number]).columns)
                
                st.success("¡Datos cargados exitosamente!")
                st.dataframe(st.session_state.datos)
                
                # Botón para continuar
                if st.button("▶️ Continuar a visualización", key="continuar_despues_csv"):
                    cambiar_tab(2)
                    
            except Exception as e:
                st.error(f"Error al cargar el archivo: {str(e)}")
    
    else:  # Insertar datos manualmente
        st.subheader("Insertar datos manualmente")
        
        # Configurar variables
        col1, col2 = st.columns(2)
        with col1:
            num_variables = st.number_input("Número de variables:", min_value=1, max_value=10, value=2, key="num_vars")
        with col2:
            num_observaciones = st.number_input("Número de observaciones:", min_value=3, max_value=50, value=5, key="num_obs_manual")
        
        # Nombres de variables
        st.subheader("Nombres de las variables:")
        nombres_variables = []
        cols = st.columns(min(num_variables, 3))
        for i in range(num_variables):
            with cols[i % 3]:
                nombre = st.text_input(f"Variable {i+1}:", value=f"Variable_{i+1}", key=f"var_name_{i}")
                nombres_variables.append(nombre)
        
        # Crear formulario para datos
        st.subheader("Ingresa los datos:")
        
        # Inicializar matriz de datos en session_state si no existe
        if 'datos_manuales' not in st.session_state:
            st.session_state.datos_manuales = {}
        
        # Crear grid para entrada de datos
        datos_ingresados = {}
        for j, var_name in enumerate(nombres_variables):
            st.write(f"**{var_name}:**")
            cols = st.columns(min(num_observaciones, 5))
            datos_ingresados[var_name] = []
            
            for i in range(num_observaciones):
                with cols[i % 5]:
                    valor = st.number_input(
                        f"Obs {i+1}:", 
                        value=0.0, 
                        key=f"dato_{j}_{i}",
                        step=0.1
                    )
                    datos_ingresados[var_name].append(valor)
        
        # Agregar columna de identificadores
        datos_ingresados['ID'] = [f'Obs_{i+1}' for i in range(num_observaciones)]
        
        # Mostrar vista previa
        if st.button("👁️ Vista previa de los datos", key="preview_manual"):
            df_preview = pd.DataFrame(datos_ingresados)
            st.dataframe(df_preview)
        
        # Confirmar datos
        if st.button("✅ Confirmar datos", key="confirmar_manual"):
            try:
                st.session_state.datos = pd.DataFrame(datos_ingresados)
                st.session_state.variables_disponibles = nombres_variables
                st.success("¡Datos ingresados exitosamente!")
                st.dataframe(st.session_state.datos)
                
                # Botón para continuar
                if st.button("▶️ Continuar a visualización", key="continuar_despues_manual"):
                    cambiar_tab(2)
                    
            except Exception as e:
                st.error(f"Error al procesar los datos: {str(e)}")

# PESTAÑA 3: Visualización de datos
with tabs[2]:
    st.header("Vista previa de los datos")
    
    if st.session_state.datos is not None:
        st.dataframe(st.session_state.datos)
        
        variables_seleccionadas = st.multiselect(
            "Selecciona variables para el análisis:",
            st.session_state.variables_disponibles,
            default=st.session_state.variables_disponibles[:2] if len(st.session_state.variables_disponibles) >= 2 else st.session_state.variables_disponibles,
            key="variables_seleccionadas"
        )
        
        if st.button("▶️ Continuar al análisis", key="ir_analisis"):
            if variables_seleccionadas:
                st.session_state.variables_analisis = variables_seleccionadas
                cambiar_tab(3)
                st.success("¡Continuemos con el análisis!")
            else:
                st.error("Por favor selecciona al menos una variable.")
    else:
        st.info("Primero genera o carga datos en la pestaña anterior.")

# PESTAÑA 4: Configuración del análisis
with tabs[3]:
    st.header("Configuración del Análisis Jerárquico")
    
    if st.session_state.datos is not None:
        # Distancia entre puntos
        metrica = st.selectbox(
            "📏 ¿Cómo medimos la distancia entre los puntos?",
            ["Euclidiana", "Manhattan", "Máxima"],
            key="metrica"
        )
        
        with st.expander("📘 ¿Qué significa esto?"):
            st.write("La 'distancia' nos dice qué tan diferentes son dos estudiantes según sus notas.")
            st.markdown("""
            - 🔹 **Euclidiana**: mide la distancia más directa entre dos puntos, como una línea recta.
            - 🔹 **Manhattan**: mide la distancia como si caminaras por calles en forma de cuadrícula.
            - 🔹 **Máxima**: solo toma en cuenta la mayor diferencia entre las variables (la más marcada).
            """)
        
        # Método de enlace
        metodo_enlace = st.selectbox(
            "🔗 ¿Cómo unimos los grupos de estudiantes?",
            ["Ward", "Single", "Complete", "Average"],
            key="metodo_enlace"
        )
        
        with st.expander("📘 ¿Qué significa esto?"):
            st.write("Los métodos de enlace deciden cómo se agrupan los estudiantes paso a paso.")
            st.markdown("""
            - 🔹 **Ward**: forma grupos tratando de que dentro de cada grupo las notas sean lo más parecidas posible.
            - 🔹 **Single (mínima distancia)**: une grupos si hay al menos un par de estudiantes muy cercanos.
            - 🔹 **Complete (máxima distancia)**: solo une grupos si todos sus estudiantes están cerca entre sí.
            - 🔹 **Average**: hace un balance usando el promedio de todas las distancias entre los grupos.
            """)
        
        if st.button("📈 Generar dendrograma", key="generar_dendro"):
            if hasattr(st.session_state, 'variables_analisis'):
                st.session_state.metrica_seleccionada = metrica
                st.session_state.metodo_enlace_seleccionado = metodo_enlace
                cambiar_tab(4)
                st.success("¡Dendrograma generado! Ve a la pestaña de resultados.")
            else:
                st.error("Por favor selecciona las variables en la pestaña anterior.")
    else:
        st.info("Primero genera o carga datos en las pestañas anteriores.")

# PESTAÑA 5: Dendrograma e interpretación
with tabs[4]:
    st.header("🌳 Dendrograma generado")
    
    if (st.session_state.datos is not None and 
        hasattr(st.session_state, 'variables_analisis') and 
        hasattr(st.session_state, 'metrica_seleccionada')):
        
        try:
            # Preparar datos para el análisis
            datos_analisis = st.session_state.datos[st.session_state.variables_analisis]
            
            # Crear dendrograma
            linkage_matrix = crear_dendrograma(
                datos_analisis, 
                st.session_state.metrica_seleccionada, 
                st.session_state.metodo_enlace_seleccionado
            )
            
            # Mostrar dendrograma
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Crear etiquetas si hay columna de estudiantes o ID
            if 'Estudiante' in st.session_state.datos.columns:
                labels = st.session_state.datos['Estudiante'].tolist()
            elif 'ID' in st.session_state.datos.columns:
                labels = st.session_state.datos['ID'].tolist()
            else:
                labels = [f'Obs_{i+1}' for i in range(len(datos_analisis))]
            
            dendrogram(linkage_matrix, labels=labels, ax=ax, orientation='top')
            ax.set_title(f'Dendrograma - {st.session_state.metrica_seleccionada} / {st.session_state.metodo_enlace_seleccionado}')
            ax.set_xlabel('Observaciones')
            ax.set_ylabel('Distancia')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            st.pyplot(fig)
            
            # Selector de número de grupos
            num_grupos = st.number_input(
                "¿Cuántos grupos quieres formar?", 
                value=2, 
                min_value=2,
                max_value=len(datos_analisis)-1,
                key="num_grupos"
            )
            
            if st.button("✂️ Cortar dendrograma", key="cortar"):
                # Formar grupos
                grupos = fcluster(linkage_matrix, num_grupos, criterion='maxclust')
                
                # Crear DataFrame con grupos
                resultado = st.session_state.datos.copy()
                resultado['Grupo'] = grupos
                
                # Mostrar resultados
                st.subheader("Resultados del agrupamiento:")
                st.dataframe(resultado)
                
                # Estadísticas por grupo
                st.subheader("Estadísticas por grupo:")
                for i in range(1, num_grupos + 1):
                    st.write(f"**Grupo {i}**: {len(resultado[resultado['Grupo'] == i])} observaciones")
                    grupo_stats = resultado[resultado['Grupo'] == i][st.session_state.variables_analisis].describe()
                    st.dataframe(grupo_stats)
                
                # Interpretación automática
                st.subheader("📋 Interpretación del Dendrograma:")
                
                # Análisis básico
                st.write("**Resumen del análisis:**")
                interpretacion = f"""
                Se han formado {num_grupos} grupos usando el método {st.session_state.metodo_enlace_seleccionado} 
                con distancia {st.session_state.metrica_seleccionada}.
                
                Distribución de observaciones por grupo:
                """
                
                for i in range(1, num_grupos + 1):
                    count = len(resultado[resultado['Grupo'] == i])
                    interpretacion += f"\n- Grupo {i}: {count} observaciones"
                
                st.text(interpretacion)
                
                # Análisis más detallado
                st.write("**¿Qué significa este dendrograma?**")
                
                # Calcular estadísticas para la interpretación
                medias_por_grupo = {}
                for i in range(1, num_grupos + 1):
                    grupo_data = resultado[resultado['Grupo'] == i][st.session_state.variables_analisis]
                    medias_por_grupo[i] = grupo_data.mean()
                
                # Interpretación basada en las medias
                st.markdown("### 🔍 Características de cada grupo:")
                for i in range(1, num_grupos + 1):
                    st.write(f"**Grupo {i}:**")
                    grupo_data = resultado[resultado['Grupo'] == i][st.session_state.variables_analisis]
                    
                    # Descripción del grupo
                    descripcion = f"- Contiene {len(grupo_data)} observaciones\n"
                    
                    # Analizar cada variable
                    for var in st.session_state.variables_analisis:
                        media = grupo_data[var].mean()
                        std = grupo_data[var].std()
                        descripcion += f"- {var}: promedio = {media:.2f} (±{std:.2f})\n"
                    
                    st.text(descripcion)
                
                # Explicación del método usado
                st.markdown("### 🛠️ Método utilizado:")
                metodo_explicacion = f"""
                **Distancia {st.session_state.metrica_seleccionada}:**
                """
                
                if st.session_state.metrica_seleccionada == "Euclidiana":
                    metodo_explicacion += """
                - Mide la distancia más directa entre dos puntos
                - Es la distancia "en línea recta" en el espacio multidimensional
                - Sensible a todas las variables por igual
                """
                elif st.session_state.metrica_seleccionada == "Manhattan":
                    metodo_explicacion += """
                - Mide la distancia como si caminaras por calles en cuadrícula
                - Suma las diferencias absolutas en cada variable
                - Menos sensible a valores extremos que la euclidiana
                """
                else:  # Máxima
                    metodo_explicacion += """
                - Solo considera la mayor diferencia entre variables
                - Útil cuando una variable domina el análisis
                - Ignora las diferencias pequeñas
                """
                
                metodo_explicacion += f"""
                
                **Enlace {st.session_state.metodo_enlace_seleccionado}:**
                """
                
                if st.session_state.metodo_enlace_seleccionado == "Ward":
                    metodo_explicacion += """
                - Minimiza la varianza dentro de cada grupo
                - Tiende a crear grupos compactos y de tamaño similar
                - Recomendado para la mayoría de aplicaciones
                """
                elif st.session_state.metodo_enlace_seleccionado == "Single":
                    metodo_explicacion += """
                - Une grupos basándose en los puntos más cercanos
                - Puede crear grupos alargados ("efecto cadena")
                - Útil para detectar formas irregulares
                """
                elif st.session_state.metodo_enlace_seleccionado == "Complete":
                    metodo_explicacion += """
                - Une grupos basándose en los puntos más lejanos
                - Tiende a crear grupos compactos y esféricos
                - Conservador, evita grupos muy dispersos
                """
                else:  # Average
                    metodo_explicacion += """
                - Considera el promedio de todas las distancias entre grupos
                - Balanceado entre single y complete
                - Robusto ante valores extremos
                """
                
                st.markdown(metodo_explicacion)
                
                # Cómo interpretar el dendrograma
                st.markdown("### 📊 Cómo leer el dendrograma:")
                dendro_explicacion = """
                **Elementos del dendrograma:**
                - **Eje horizontal**: Muestra las observaciones individuales
                - **Eje vertical**: Muestra la distancia a la que se unen los grupos
                - **Ramas**: Conectan observaciones/grupos similares
                - **Altura de unión**: Mientras más alta, más diferentes son los grupos
                
                **Interpretación:**
                - Observaciones que se unen a alturas bajas son muy similares
                - Observaciones que se unen a alturas altas son muy diferentes
                - Para formar grupos, "corta" el dendrograma horizontalmente
                - El número de ramas que cruza la línea = número de grupos
                """
                st.markdown(dendro_explicacion)
                
                # Recomendaciones
                st.markdown("### 💡 Recomendaciones:")
                
                # Analizar la calidad del agrupamiento
                if num_grupos == 2:
                    recomendacion = "Con 2 grupos, tienes una división básica. "
                elif num_grupos <= len(datos_analisis) // 3:
                    recomendacion = "El número de grupos parece apropiado para estos datos. "
                else:
                    recomendacion = "Tienes muchos grupos pequeños. Considera reducir el número. "
                
                # Análisis de la variabilidad
                variabilidad_total = np.var(datos_analisis.values)
                variabilidad_intra = 0
                for i in range(1, num_grupos + 1):
                    grupo_data = resultado[resultado['Grupo'] == i][st.session_state.variables_analisis].values
                    if len(grupo_data) > 1:
                        variabilidad_intra += np.var(grupo_data)
                
                if variabilidad_intra < variabilidad_total * 0.5:
                    recomendacion += "Los grupos formados son internamente homogéneos (buena separación). "
                else:
                    recomendacion += "Los grupos tienen bastante variabilidad interna. "
                
                # Recomendaciones específicas del método
                if st.session_state.metodo_enlace_seleccionado == "Ward":
                    recomendacion += "El método Ward es excelente para datos educativos como estos."
                elif st.session_state.metodo_enlace_seleccionado == "Single":
                    recomendacion += "Si ves grupos muy alargados, considera usar método Complete o Ward."
                
                st.info(recomendacion)
                
                # Botón de descarga (simulado)
                if st.button("📄 Descargar reporte educativo", key="descargar"):
                    st.info("En una implementación completa, aquí se descargaría un reporte en PDF.")
                    
        except Exception as e:
            st.error(f"Error al generar el dendrograma: {str(e)}")
            st.info("Asegúrate de que los datos sean numéricos y estén correctamente formateados.")
    
    else:
        st.info("Completa los pasos anteriores para generar el dendrograma.")

# Información adicional en la barra lateral
st.sidebar.header("ℹ️ Información")
st.sidebar.markdown("""
Esta aplicación te permite aprender sobre segmentación jerárquica 
de manera interactiva. Sigue los pasos en orden:

1. **Introducción**: Conceptos básicos
2. **Ingreso de Datos**: Carga, simula o inserta datos manualmente
3. **Visualizar Datos**: Revisa y selecciona variables
4. **Análisis**: Configura parámetros
5. **Resultado**: Interpreta el dendrograma

¡Explora diferentes configuraciones para entender mejor el análisis!
""")

# Mostrar estado actual
if st.session_state.datos is not None:
    st.sidebar.success("✅ Datos cargados")
    st.sidebar.write(f"Filas: {len(st.session_state.datos)}")
    st.sidebar.write(f"Columnas: {len(st.session_state.datos.columns)}")
else:
    st.sidebar.warning("⚠️ No hay datos cargados")

# Indicador de progreso
st.sidebar.header("📊 Progreso")
progreso = 0
if st.session_state.etapa >= 1:
    progreso += 20
if st.session_state.datos is not None:
    progreso += 20
if hasattr(st.session_state, 'variables_analisis'):
    progreso += 20
if hasattr(st.session_state, 'metrica_seleccionada'):
    progreso += 20
if st.session_state.etapa >= 4:
    progreso += 20

st.sidebar.progress(progreso / 100)
st.sidebar.write(f"Completado: {progreso}%")