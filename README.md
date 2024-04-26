# PPGAFb_FES
Extraccción y selección de características para algoritmos de detección de Fibrilación Auricular usando la señal de fotopletismografía

La mayoría de las características extraídas están basadas en parámetros estadísticos a partir de características temporales y morfológicas. Los parámetros estadísticos empleados fueron: valor máximo (Vmax), valor mínimo (Vmin), rango intercuartil (IQR), desviación estándar (Std), mediana (Vmd), media (Vav), coeficiente de asimetría (ca) y curtosis (kurt). En la figura se muestran las 46 características extraídas para cada ventana de análisis.


Donde:
PP- Intervalo pico a pico
Ac- Área total de cada onda
m1- Pendiente de subida de las ondas sistólicas de la señal en la ventana de análisis
m2- Pendiente de adelanto, formada por el inicio de una onda y el pico sistólico de la siguiente
DP- Densidad espectral de potencia
