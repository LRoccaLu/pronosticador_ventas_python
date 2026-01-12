"""
Proyecto de análisis y pronóstico de ventas.
Desarrollado como herramienta de apoyo a la toma de decisiones comerciales.
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import pandas as pd
from fpdf import FPDF

# Obtener datos del usuario
def obtener_datos(cantidad_meses):
    datos = []
    for i in range(1, cantidad_meses + 1):
        valor = float(input(f"Ingrese el valor del mes {i}: "))
        datos.append(valor)
    return np.array(datos)

# Métricas de evaluación de modelos
def calcular_metricas(real, pronosticado):
    mae = mean_absolute_error(real, pronosticado)
    mse = mean_squared_error(real, pronosticado)
    rmse = np.sqrt(mse)
    return mae, mse, rmse

# Pronóstico usando regresión lineal
def pronostico_regresion_lineal(datos, mes_pronostico):
    X = np.arange(1, len(datos) + 1).reshape(-1, 1)
    y = datos
    modelo = LinearRegression()
    modelo.fit(X, y)
    pronosticos_intermedios = modelo.predict(np.arange(len(datos) + 1, mes_pronostico + 1).reshape(-1, 1))
    meses_totales = np.arange(1, mes_pronostico + 1).reshape(-1, 1)
    tendencia = modelo.predict(meses_totales)
    return pronosticos_intermedios, tendencia, modelo.coef_[0], modelo.intercept_

# Pronóstico usando regresión polinómica
def pronostico_regresion_polinomica(datos, mes_pronostico, grado=2):
    X = np.arange(1, len(datos) + 1).reshape(-1, 1)
    y = datos
    poly = PolynomialFeatures(degree=grado)
    X_poly = poly.fit_transform(X)
    modelo = LinearRegression()
    modelo.fit(X_poly, y)
    meses_intermedios = np.arange(len(datos) + 1, mes_pronostico + 1).reshape(-1, 1)
    meses_intermedios_poly = poly.transform(meses_intermedios)
    pronosticos_intermedios = modelo.predict(meses_intermedios_poly)
    meses_totales = np.arange(1, mes_pronostico + 1).reshape(-1, 1)
    meses_totales_poly = poly.transform(meses_totales)
    tendencia = modelo.predict(meses_totales_poly)
    return pronosticos_intermedios, tendencia

# Pronóstico usando suavización exponencial simple
def pronostico_suavizacion_exponencial(datos, mes_pronostico):
    modelo = SimpleExpSmoothing(datos).fit(smoothing_level=0.8, optimized=False)
    pronostico = modelo.forecast(mes_pronostico - len(datos))
    tendencia = np.concatenate((modelo.fittedvalues, pronostico))
    return pronostico, tendencia

# Pronóstico usando Holt-Winters
def pronostico_holt_winters(datos, mes_pronostico):
    modelo = ExponentialSmoothing(datos, trend='add', seasonal=None).fit()
    pronostico = modelo.forecast(mes_pronostico - len(datos))
    tendencia = np.concatenate((modelo.fittedvalues, pronostico))
    return pronostico, tendencia

# Generar gráfico comparativo de los modelos
def generar_grafico_comparativo(datos, mes_pronostico, resultados):
    meses = np.arange(1, len(datos) + 1)
    plt.bar(meses, datos, color='skyblue', label='Datos históricos')
    # Añadir los pronósticos de cada modelo
    for modelo, resultado in resultados.items():
        pronosticos_intermedios, tendencia = resultado
        meses_totales = np.arange(1, mes_pronostico + 1)
        plt.plot(meses_totales, tendencia, label=modelo)
    plt.xlabel('Meses')
    plt.ylabel('Ventas')
    plt.title('Comparación de Modelos de Pronóstico')
    plt.legend()
    imagen_path = os.path.join(os.getcwd(), 'grafico_comparativo_modelos.png')
    plt.savefig(imagen_path)
    plt.show()
    return imagen_path

# Agregar las explicaciones de los modelos
def agregar_explicacion_modelos(pdf):
    pdf.set_font('Arial', '', 12)
    pdf.ln(10)
    pdf.cell(200, 10, 'Explicación de los modelos:', ln=True)

    pdf.cell(200, 10, '1. Regresión Lineal:', ln=True)
    pdf.multi_cell(190, 10, 'Este modelo proyecta las ventas futuras mediante una línea recta basada en los datos históricos. Es ideal cuando las ventas siguen una tendencia constante, lo que permite prever con mayor certeza los niveles de demanda futura. Es útil para planificar el inventario en productos con comportamiento estable en ventas.')
    
    pdf.cell(200, 10, '2. Regresión Polinómica:', ln=True)
    pdf.multi_cell(190, 10, 'A diferencia de la regresión lineal, este modelo ajusta una curva para capturar mejor las fluctuaciones en las ventas. Se recomienda para productos con comportamientos más complejos, como aquellos que presentan picos y caídas de demanda, lo que facilita una planificación más precisa ante variaciones imprevistas.')
    
    pdf.cell(200, 10, '3. Suavización Exponencial Simple:', ln=True)
    pdf.multi_cell(190, 10, 'Este método otorga mayor peso a los datos más recientes, siendo útil cuando las ventas tienen fluctuaciones sin una tendencia clara a largo plazo. Es valioso para productos nuevos o en mercados volátiles, donde las ventas recientes son más representativas de las futuras.')
    
    pdf.cell(200, 10, '4. Holt-Winters:', ln=True)
    pdf.multi_cell(190, 10, 'Este modelo captura tanto las tendencias como los patrones estacionales. Es ideal para productos que muestran ciclos de ventas repetitivos a lo largo del año, como aquellos que experimentan mayor demanda en eventos estacionales (por ejemplo, ventas aumentadas durante campañas escolares). Permite a la gerencia ajustar las estrategias de marketing y aprovisionamiento en función de estos ciclos.')

# Función para verificar si el archivo existe y generar uno nuevo con un número incremental si es necesario
def generar_nombre_pdf(base_path, base_name):
    contador = 0
    file_name = f"{base_name}.pdf"
    file_path = os.path.join(base_path, file_name)
    
    # Verificar si el archivo ya existe
    while os.path.exists(file_path):
        contador += 1
        file_name = f"{base_name} {contador}.pdf"
        file_path = os.path.join(base_path, file_name)
    
    return file_path

# Generar informe PDF
def generar_informe_comparativo(datos, mes_pronostico, resultados, imagen_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(200, 10, 'Informe Comparativo de Modelos de Pronóstico', ln=True, align='C')

    # Datos históricos
    pdf.set_font('Arial', '', 12)
    pdf.ln(10)
    pdf.cell(200, 10, 'Datos históricos:', ln=True)
    for i, valor in enumerate(datos, start=1):
        pdf.cell(200, 10, f"Mes {i}: {valor}", ln=True)
    
    pdf.ln(10)
    pdf.cell(200, 10, 'Resultados de los modelos:', ln=True)
    for modelo, resultado in resultados.items():
        pronosticos_intermedios, tendencia = resultado
        pdf.cell(200, 10, f"Pronósticos intermedios con {modelo}: {pronosticos_intermedios}", ln=True)

    # Agregar explicaciones de los modelos
    agregar_explicacion_modelos(pdf)

    # Advertencia
    pdf.ln(10)
    pdf.cell(200, 10, 'Advertencia: Los resultados pueden variar debido a factores externos.', ln=True)

    # Agregar gráfico al informe
    pdf.image(imagen_path, x=10, y=pdf.get_y(), w=190)

    # Obtener la ruta del archivo PDF con nombre dinámico
    informe_path = generar_nombre_pdf(os.path.dirname(imagen_path), 'informe_comparativo_pronostico')
    pdf.output(informe_path)

    print(f"Informe PDF generado en: {informe_path}")

def main():
    # Pedir la cantidad de meses históricos
    while True:
        cantidad_meses = int(input("Ingrese la cantidad de meses de datos que tiene (mínimo 3): "))
        if cantidad_meses >= 3:
            break
        else:
            print("Error: Debe ingresar al menos 3 meses de datos.")

    # Solicitar el mes para el pronóstico
    mes_pronostico = int(input(f"Ingrese el número del mes para el cual desea el pronóstico (debe ser mayor que {cantidad_meses}): "))

    # Obtener los datos de cada mes
    datos = obtener_datos(cantidad_meses)

    # Aplicar diferentes métodos de pronóstico
    resultados = {}
    pronostico_rl, tendencia_rl, _, _ = pronostico_regresion_lineal(datos, mes_pronostico)
    resultados["Regresión Lineal"] = (pronostico_rl, tendencia_rl)

    pronostico_rp, tendencia_rp = pronostico_regresion_polinomica(datos, mes_pronostico)
    resultados["Regresión Polinómica"] = (pronostico_rp, tendencia_rp)

    pronostico_se, tendencia_se = pronostico_suavizacion_exponencial(datos, mes_pronostico)
    resultados["Suavización Exponencial"] = (pronostico_se, tendencia_se)

    pronostico_hw, tendencia_hw = pronostico_holt_winters(datos, mes_pronostico)
    resultados["Holt-Winters"] = (pronostico_hw, tendencia_hw)

    # Generar gráfico comparativo
    imagen_path = generar_grafico_comparativo(datos, mes_pronostico, resultados)

    # Generar informe en PDF
    generar_informe_comparativo(datos, mes_pronostico, resultados, imagen_path)

if __name__ == "__main__":
    main()

