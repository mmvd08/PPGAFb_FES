# PPGAFb_FES
Extraccción y selección de características para algoritmos de detección de Fibrilación Auricular usando la señal de fotopletismografía

La mayoría de las características extraídas están basadas en parámetros estadísticos a partir de características temporales y morfológicas. Los parámetros estadísticos empleados fueron: valor máximo (Vmax), valor mínimo (Vmin), rango intercuartil (IQR), desviación estándar (Std), mediana (Vmd), media (Vav), coeficiente de asimetría (ca) y curtosis (kurt). En la figura se muestran las 46 características extraídas para cada ventana de análisis.


Donde:
- PP- Intervalo pico a pico
- Ac- Área total de cada onda
- m1- Pendiente de subida de las ondas sistólicas de la señal en la ventana de análisis
- m2- Pendiente de adelanto, formada por el inicio de una onda y el pico sistólico de la siguiente
- DP- Densidad espectral de potencia
- Vpp- Variación de PP: Se divide el vector de los PP de la ventana de análisis entre cada elemento del propio vector. El resultado se almacena en una matriz de la que se resta la matriz identidad y el resultado se eleva al cuadrado. Luego se halla la suma de cada fila, se buscan máximo y el mínimo y se determina su diferencia.
- Vp- Variación de la amplitud de los picos: Emplea el procedimiento descrito anteriormente con la diferencia que se emplea el vector con la amplitud de los picos de la ventana de análisis en lugar del vector de los PP.
- E- La entropía condicional corregida (CCE) (Chauhan, 2021)
- FmDP- Frecuencia de la media de la densidad espectral
- C- Correlación entre los primeros y últimos 3 s de la ventana de análisis
- FC1- La ventana de análisis se divide en tres intervalos de 3, 4 y 3 s respectivamente. Para cada intervalo calcula la frecuencia cardíaca instantánea (FCi), representadas en lo adelante como: FCi_1, FCi_2 y FCi_3. Luego FC1= FCi_2 - FCi_1
- FC2= FCi_3 - FCi_2
- FC3= FCi_3 - FCi_1
- Fmxp- Frecuencia del valor máximo de potencia

Dada la matriz de características se realiza la selección de aquellas que mejoren el rendimiento y velocidad de los clasificadores. Para ello se utiliza una estrategia de selección por importancia basada en tres clasificadores: K Vecinos más cercanos, MLP y Bosques Aleatorios. Inicialmente se seleccionaron las 23 características más relevantes para cada clasificador. Posteriormente se buscaron coincidencias de las características que más información aportan. De este grupo de características se evalúa la correlación para identificar las que son redundantes. El grupo de características que más se repite en los tres algoritmos y que no poseen una correlación superior al 70% entre ellas, se toma como punto de partida para construir el vector de características que se emplea en la validación cruzada.

