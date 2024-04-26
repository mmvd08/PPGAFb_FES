#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 15:21:46 2022

@author: aasl
"""
import EntropyHub as EH
import aux_entropia as aux
import numpy as np
from scipy import fft, arange,stats
from scipy.io import loadmat    #Para leer las senales de matlab
import scipy.signal as signal
from matplotlib.pyplot import plot, stem, axis, legend, show,  grid, figure, savefig, semilogy
from matplotlib.pyplot import xlabel, ylabel, subplot,subplots, axes, xticks, yticks  #,hold
from sklearn.cluster import DBSCAN
from scipy.fftpack import next_fast_len, rfft, irfft, ifft
from scipy.signal import welch
import pandas as pnds

####################Detección de picos:primera derivada####################################

def picos(ppg_w, fs,mediana):
    m=np.diff(ppg_w)        #calculando la primera derivada
    picos=list()
    for i in range(0, len(m)-2):    #buscando los picos en la primera derivada(cambios de signo) 
      pos1=m[i]
      pos2=m[i+1]
      if pos2<0 and pos1>0:
        picos.append(i+1)
       
    picos=np.array(picos,dtype=int)
    pico_prueba=list()
    picos_x = picos[ppg_w[picos] > mediana]    
    picobinario=np.zeros((len(m),1),dtype=int)   
    
    for i in range(0,len(picos_x)):
        if i>=len(picos_x)-10:
            amp_max=np.max(ppg_w[picos_x[len(picos_x)-10:]]) 
            umbral=amp_max*0.30 
        else:    
            window=ppg_w[picos_x[i:i+10]]
            amp_max=np.max(window)
            umbral=amp_max*0.30              #estableciendo como umbral el 30% del maximo de la señal(para eliminar artefactos de movimiento de pequeñas aplitudes)       
        if ppg_w[picos_x[i]]>umbral:
                picobinario[picos_x[i]]=1
                pico_prueba.append (picos_x[i])
             
    refract=np.ceil(fs*0.25)                 # periodo refractario=0.25s(Guyton)
    picobinario=np.array(picobinario)
    
    if np.sum(picobinario!=0)>1:
        picos=filter_picos_dbscan(picobinario, refract,ppg_w)
    else:
        picos=np.array([])
    
    return picos    

################### Depuración de picos:Algoritmo de DBSCAN ###############################
def filter_picos_dbscan(picobinario, refract,ppg):
    hw_size=refract
#    prediction=picobinario
    
    detect = np.array(picobinario, dtype='int').squeeze()
    detect2 = np.array(np.nonzero(detect)).squeeze()
    detect3 = detect2.reshape(-1, 1)
    db = DBSCAN(eps=hw_size, min_samples=1, metric='cityblock')  #Agrupar los 1 que estan a una distancia de una muestra hasta 1/4 del periodo refractario del corazón (Gython)
    clusters = db.fit_predict(detect3)
    
    top = np.max(clusters) + 1
    
    picosdbscan=list()
    for i in range(top):
        idx1 = clusters == i
        idxint = np.array(idx1, dtype='int')
        ppg1=idxint*ppg[detect2]
        val = np.argmax(ppg1)
        
        picosdbscan.append(detect2[val])

    
    picosdbscanx = np.array(picosdbscan)

    return picosdbscanx


################### Cálculo de la fecuencia cardiaca instantanea (lat/seg)######################################
def FC2 (picos,fs):  
    pp=np.diff(picos)
    FC=np.mean(fs/pp)
#    FC= (len(picos)*60)/((picos[-1]-picos[0])/fs)
    return FC


################### Cálculo de la distancia Pico-Pico #####################################
def PP(picos,fs):
    if len(picos)>0:
        pp=np.diff(picos)
        return pp/fs
    else: 
        return np.array([])


################### Desviación estandar de PP #############################################   
def stdPP(pp):
    if len(pp)>0:
        x=np.std(pp)
        return x
    else: 
        return 0

################### Máximo PP #############################################################   
def maxPP(pp):
    if len(pp)>0:
        mx=np.max(pp)
        return mx
    else:
        return 0

################### Mínimo de PP ##########################################################   
def minPP(pp):
    if len(pp)>0:
        mn=np.min(pp)
        return mn
    else:
        return 0

################### Promedio de los PP ####################################################   
def mediaPP(pp):
    if len(pp)>0:
        md=np.mean(pp)
        return md
    else:
        return 0

################### Mediana de los PP #####################################################   
def medianaPP(pp):
    if len(pp)>0:
        mdn=np.median(pp)
        return mdn
    else:
        return 0

################### Rango intercuartil de los PP ##########################################   
def ricPP(pp):
    if len(pp)>0:
        ricl=stats.iqr(pp)
        return ricl
    else: 
        return 0

################### Detección del punto de inicio:Area mínima #############################
def inicio(ppg,picos): 
    if len(picos)>0:
        area=list()
        area_position=list()
        pie=list()
        areas_min=list()
        suma=0
        minimo=np.abs(np.min(ppg))
        for i in range(0,len(picos)-1):              #Moverse por toda la señal
            for w in range(picos[i],picos[i+1]-20):  #desplazando la ventana del area entre pico y pico de la señal
                for m in range(0,20):                #calculando el área en una ventana de 20 muestras
                    suma=suma+minimo+ppg[(w+m)]
                area.append(suma)
                area_position.append((w+11))
                suma=0
            
            Area_array=area.copy()
            area.clear()
            areas_min.append(np.min(Area_array))
            areamin=np.argmin(Area_array)
            pie.append(area_position[areamin])
            area_position.clear()
            
        
        pie=np.array(pie)    
        areas_min=np.array(areas_min)          
                
        return pie    
    else:
        return np.array([])

################### Cálculo del área bajo la curva ########################################
def calc_area(ppg,pie):
      if len(pie)>0:
          areas=list()
          for i in range(1,len(pie)):     
             x1=pie[i-1]
             x2=pie[i]
             y1=ppg[x1]
             y2=ppg[x2]
             m=(y2-y1)/(x2-x1)   #construyendo una recta entre los dos pies para crear una base para el cálculo del area
             n=y1-m*x1
             suma=0
             for x in range(x1,x2):
                  suma=suma + (ppg[x]-(m*x+n))
             areas.append(suma)    
                 
          A=np.array(areas)    
          return A 
      else:
          return np.array([])
################### Desviacion estandar de las areas en la ventana #########################   
def stdA(A):
    if len(A)>0:
        a1=np.std(A)
        return a1
    else:
        return 0

################### Maximo de las areas ####################################################   
def maxA(A):
    if len(A)>0:
        amx=np.max(A)
        return amx
    else:
        return 0

################### Minimo de las areas ####################################################  
def minA(A):
    if len(A)>0:
        amn=np.min(A)
        return amn
    else:
        return 0 

################### Promedio de las areas ##################################################   
def mediaA(A):
    if len(A)>0:
        amd=np.mean(A)
        return amd
    else:
        return 0

################### Mediana de las areas ###################################################   
def medianaA(A):
    if len(A)>0:
        amdn=np.median(A)
        return amdn
    else:
        return 0

################### Rango intercuartil de las areas ###########################################   
def ricA(A):
    if len(A)>0:
        aricl=stats.iqr(A)
        return aricl
    else:
        return 0


################### calculando pendientes de inicio a pico ################################# 
def m1(picos,pie,ppg):
    if len(picos):
        n1=len(picos)
        n2=len(pie)
        m1=list()
        if n1>n2:          #determinando si hay mas picos que pies para
           for i in range(0,n2):
               x2=picos[i+1]
               x1=pie[i]
               y2=ppg[x2]
               y1=ppg[x1]
               m=(y2-y1)/(x2-x1)   
               m1.append(m)
        else:
           for i in range(0,n2):
               x2=picos[i+1]
               x1=pie[i]
               y2=ppg[x2]
               y1=ppg[x1]
               m=(y2-y1)/(x2-x1)   
               m1.append(m)
        m1=np.array(m1)
        return m1 
    else:
        return np.array([])
################### Desviacion estandar de las pendientes de subida ########################  
def stdm1(m1):
    if len(m1)>0:
        ps=np.std(m1)
        return ps
    else:
        return 0

################### Pendiente de subida maxima ############################################# 
def maxm1(m1):
    if len(m1)>0:
        psmx=np.max(m1)
        return psmx
    else:
        return 0

################### Pendiente de subida minima ############################################   
def minm1(m1):
    if len(m1)>0:
        psmn=np.min(m1)
        return psmn
    else:
        return 0

################### Promedio de las pendientes de subida ##################################   
def mediam1(m1):
    if len(m1)>0:
        psmd=np.mean(m1)
        return psmd
    else:
        return 0

################### Mediana de las pendientes de subida ###################################   
def medianam1(m1):
    if len(m1)>0:
        psmdn=np.median(m1)
        return psmdn
    else:
        return 0

################### Rango intercuartil de las pendientes de subida ########################   
def ricm1(m1):
    if len(m1)>0:
        psricl=stats.iqr(m1)
        return psricl
    else:
        return 0


################### calculando pendientes de inicio1 a pico2 ################################# 
def m1_2(picos,pie,ppg):
    if len(picos):
        n1=len(picos)
        n2=len(pie)
        m1=list()
        if n1>n2:          #determinando si hay mas picos que pies para
           for i in range(0,n2-1):
               x2=picos[i+2]
               x1=pie[i]
               y2=ppg[x2]
               y1=ppg[x1]
               m=(y2-y1)/(x2-x1)   
               m1.append(m)
        else:
           for i in range(0,n2):
               x2=picos[i+2]
               x1=pie[i]
               y2=ppg[x2]
               y1=ppg[x1]
               m=(y2-y1)/(x2-x1)   
               m1.append(m)
        m1=np.array(m1)
        return m1 
    else:
        return np.array([])
################### Desviacion estandar de las pendientes de subida ########################  
def stdm1_2(m1_2):
    if len(m1_2)>0:
        ps=np.std(m1_2)
        return ps
    else:
        return 0

################### Pendiente de subida maxima ############################################# 
def maxm1_2(m1_2):
    if len(m1_2)>0:
        psmx=np.max(m1_2)
        return psmx
    else:
        return 0

################### Pendiente de subida minima ############################################   
def minm1_2(m1_2):
    if len(m1_2)>0:
        psmn=np.min(m1_2)
        return psmn
    else:
        return 0

################### Promedio de las pendientes de subida ##################################   
def mediam1_2(m1_2):
    if len(m1_2)>0:
        psmd=np.mean(m1_2)
        return psmd
    else:
        return 0

################### Mediana de las pendientes de subida ###################################   
def medianam1_2(m1_2):
    if len(m1_2)>0:
        psmdn=np.median(m1_2)
        return psmdn
    else:
        return 0

################### Rango intercuartil de las pendientes de subida ########################   
def ricm1_2(m1_2):
    if len(m1_2)>0:
        psricl=stats.iqr(m1_2)
        return psricl
    else:
        return 0

  
################### calculando pendientes de pico a inicio ################################
def m2(picos,pie,ppg):
    if len(picos)>0:
        n1=len(picos)
        n2=len(pie)
        m1=list()
        if n1>n2:          #determinando si hay mas picos que pies para
           for i in range(0,n2):
               x1=picos[i]
               x2=pie[i]
               y1=ppg[x1]
               y2=ppg[x2]
               m=(y2-y1)/(x2-x1)   
               m1.append(m)
        else:
           for i in range(0,n2):
               x1=picos[i]
               x2=pie[i+1]
               y1=ppg[x1]
               y2=ppg[x2]
               m=(y2-y1)/(x2-x1)   
               m1.append(m)
        m1=np.array(m1)
        return m1  
    else:
        return np.array([])
################### Desviacion estandar de las pendientes de bajada #######################   
def stdm2(m2):
    if len(m2)>0:
        pb=np.std(m2)
        return pb
    else:
        return 0

################### Pendiente de bajada maxima ############################################   
def maxm2(m2):
    if len(m2)>0:
        pbmx=np.max(m2)
        return pbmx
    else:
        return 0

################### Pendiente de bajada minima ############################################   
def minm2(m2):
    if len(m2)>0:
        pbmn=np.min(m2)
        return pbmn
    else:
        return 0

################### Promedio de las pendientes de bajada ##################################  
def mediam2(m2):
    if len(m2)>0:
        pbmd=np.mean(m2)
        return pbmd
    else:
        return 0

################### Mediana de las pendientes de bajada ################################### 
def medianam2(m2):
    if len(m2)>0:
        pbmdn=np.median(m2)
        return pbmdn
    else:
        return 0

################### Rango intercuartil de las pendientes de bajada ########################   
def ricm2(m2):
    if len(m2)>0:
        pbricl=stats.iqr(m2)
        return pbricl
    else:
        return 0

################### Densidad Espectral de Potencia (Welch) ################################     
def Densidad_espectral(ppg_window,fs):      
    f, P = welch(ppg_window, fs,'hamm', nperseg=1024)  
    return f, P  
	
################### Desviacion estandar de la Densidad espectral #######################   
def stdDE(P):
    stdE=np.std(P)
    return stdE
           
################### Mediana de la Densidad espectral ################################### 
def medianaDE(P):
    Emdn=np.median(P)
    return Emdn

################### Promedio de la Densidad espectral ##################################  
def mediaDE(P):
    Emd=np.mean(P)
    return Emd

################### Skewness de la Densidad espectral ##################################  
def skwDE(P):
    Eskw=stats.skew(P)
    return Eskw

################### kurtosis de la Densidad espectral ##################################  
def KurtDE(P):
    Ekurt=stats.kurtosis(P)
    return Ekurt

################### Calculo de la Potencia total ########################################
def areaDE(P):
    suma=np.sum(P)
    return suma     

################### Calculo de la entrpia en cada venana ########################################
def entropia(ppg_window):
    tau=46
    m=2
    Cond, SEw, SEz = EH.CondEn(ppg_window, m, tau, c=6, Logx=2.718281828459045, Norm=False)
    return np.min(Cond)

################### Desviacion estandar de amplP #############################################   
def stdAP(window,picos,pie):
    if len(picos)>0:
        a=window[picos[1:]]
        p=window[pie]
        x=np.std(a-p)
        return x
    else: 
        return 0

################### Maximo de las amplitudes de P ############################################  
def maxAP(window,picos,pie):
    if len(picos)>0:
        a=window[picos[1:]]
        p=window[pie]
        mx=np.max(a-p)
        return mx
    else:
        return 0

################### Minimo de las amplitudes de P ############################################
def minAP(window,picos,pie):
    if len(picos)>0:
        a=window[picos[1:]]
        p=window[pie]
        mn=np.min(a-p)
        return mn
    else:
        return 0

################### Promedio de las amplitudes de P ###########################################   
def mediaAP(window,picos,pie):
    if len(picos)>0:
        a=window[picos[1:]]
        p=window[pie]
        md=np.mean(a-p)
        return md
    else:
        return 0

################### Mediana de las amplitudes de P ############################################   
def medianaAP(window,picos,pie):
    if len(picos)>0:
        a=window[picos[1:]]
        p=window[pie]
        mdn=np.median(a-p)
        return mdn
    else:
        return 0

################### Rango intercuartil de las amplitudes de P ################################# 
def ricAP(window,picos,pie):
    if len(picos)>0:
        a=window[picos[1:]]
        p=window[pie]
        ricl=stats.iqr(a-p)
        return ricl
    else: 
        return 0

################### Metrica 1 de los  PP ################################# 
"""
se dividen todos los intervalos pp entre cada uno y la matriz resultante se 
resta de la matriz identidad y el resultado se eleva al cuadrado. Luego se halla 
la suma de cada fila y se busca el max y el min. Se restan estos dos valores y se
devuelve el resultado. Todo se hace en unidad de tiempo y no en muestras
"""
def deltaPP2(pp):
    l=len(pp)
    mx_A=np.zeros((l,l))
    mx_ID=np.eye(l)
    for i in range(l):
        mx_A[i,:]=pp/pp[i]
        MXR_A=np.power((mx_A - mx_ID),2)
        MXR_A2=np.sum(MXR_A,axis=1)

    resta=np.max(MXR_A2)-np.min(MXR_A2)
    return resta
    
################### Metrica  de los  P ################################# 
"""
se dividen todos las amplitudes de los picos entre cada uno y la matriz resultante se 
resta de la matriz identidad y el resultado se eleva al cuadrado. Luego se halla 
la suma de cada fila y se busca el max y el min. Se restan estos dos valores y se
devuelve el resultado. Todo se hace en unidad de tiempo y no en muestras
"""
def deltaP2(picos,pies,window):
    l=len(pies)
    mx_A=np.zeros((l,l))
    mx_ID=np.eye(l)
    a=window[picos[1:]]
    p=window[pies]
    altura=a-p
    for i in range(l):
        mx_A[i,:]=altura/altura[i]
        MXR_A=np.power((mx_A - mx_ID),2)
        MXR_A2=np.sum(MXR_A,axis=1)

    resta=np.max(MXR_A2)-np.min(MXR_A2)
    return resta
 
################### Frecuencia pico de la Densidad espectral #######################   
def frec_p(P,f):
    fp=int(np.argmax(P))
    frec=f[fp]
    return frec


################### FrecMediana de la Densidad espectral ################################### 
def frecmedianaDE(Pmediana,P,f):
    pf=np.argmin(np.abs(P-Pmediana))
    frec=f[pf]
    return frec

################### FrecMedia de la Densidad espectral ################################### 
def frecmediaDE(Pmedia,P,f):
    pfm=np.argmin(np.abs(P-Pmedia))
    frec=f[pfm]
    return frec

#################Correlacion entre antes y despues de la ventana centras ################
def corr_WIN(window):
    w1=window[0:750]
    w2=window[1750:2500]    
    cor=np.corrcoef(w1,w2)
    return cor[0,1]

##########Variacion de frecuencia cardiaca instantanea dentro de la ventana ##############
def vFCI (picos,fs):   #evaluar póximamente dividiendo la ventana en 833-834-833
    p1=picos<750
    p11=picos[p1]
    
    p2=np.logical_and(picos>750,picos<1750)  
    p22=picos[p2]
    
    p3=picos>1750
    p33=picos[p3]
    
    if len(p11)<2:
        FC1=0
    else:
        fc=np.diff(p11)
        FC1=np.mean(fs/fc)
   
    if len(p22)<2:
        FC2=0
    else:
        fc=np.diff(p22)
        FC2=np.mean(fs/fc)
    
    if len(p33)<2:
        FC3=0
    else:
        fc=np.diff(p33)
        FC3=np.mean(fs/fc)
        
    FCv1=FC2-FC1
    FCv2=FC3-FC2
    FCv3=FC3-FC1
        
    return FCv1,FCv2,FCv3





