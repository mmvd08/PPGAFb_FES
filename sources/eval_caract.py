#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 10:46:38 2022

@author: mmvd08  email: mmvd08@nauta.cu
"""

import caracteristicas as caract
 
import pandas as pnds
from scipy.io import loadmat, savemat
import math
import numpy as np
from keras.utils import np_utils
from keras.utils import Sequence
from matplotlib.pyplot import plot, figure, legend, title, xlabel, ylabel, semilogy,show
import path as camino
import scipy.signal as signal

ENT = ['v100s','a103l','b265l','v628s','t343l','a104s','v818s','b269l','v101l','t351l',
       'v132s','a105l','v773l','b299l','v630s','t356s','v139l','a185l','b313l','v158s',
       't357l','v806s','a203l','v159l','b379l','t358s','v194s','a420s','v788s','b515l',
       'v199l','t393l','v254s','a582s','v783l','b516s','v309l','t394s','v823l','a631l',
       'v348s','b517l','t411l','v471l','a650s','v805l','b537l','v525l','t406s','b794s',
       'v541l','a667l','v828s','b560s','v564s','t416s','v571l','a705l','b561l','v573l',
       't430s','v574s','b562s','t434s','v632s','t458s','v635l','b588s','v831l','a712s',
       'v636s','t467l','v638s','b730s','v648s','a785l','t477l','v696s','b734s','v714s',
       't506s','v728s','a810s','t507l','v733l','b764s','v748s','t509l','v758s','b820s',
       'v761l','t546s','v769l'];
       
       
total=len(ENT)  
for k in range(0,total):
#desde
    sennal_actual=ENT[k]
    ######## generando vector de salidas
    salidas=pnds.read_csv(ENT[k]+".csv",delimiter=';')
    s=salidas.values
    vector_salida=np.ones(s[0,1],dtype='int')*s[0,2]
    tamano_s=np.shape(s)
    
    for i in range(0,tamano_s[0]):
        if s[i,0]==s[i,1]:
            vector_salida[s[i,0]]=s[i,2]
        else:
            vector_salida[s[i,0]:s[i,1]]=s[i,2]
    
    #leyendo la señal para extraer las caracteristicas
    senal1= loadmat(ENT[k]+".mat")
    val1 = senal1['val']
    ppg=val1[2,:]
    fs=250
    n = len(ppg) # longitud de la señal en muestras
    T = n/fs
    t = np.linspace(0, T, n, endpoint=False)
    
    #Filtrado (butter pasabanda de cuarto orden 0.8 a 10 Hz)
    FNyquist=fs/2
    Wn = np.array([0.8, 10])  #frecuencias de corte
    Wn1 = (2/fs) * Wn
    b, a = signal.butter(4, Wn1, 'bandpass') # Filtro(orden,frecuencia de corte, tipo)
    ppgf = signal.filtfilt(b, a, ppg)
    ppgfn = (ppgf - np.mean(ppgf))/np.std(ppgf) #normalizando la señal filtrada
    
    #calculando la mediana de la señal para el detector de picos
    mediana=np.median(np.abs(ppgfn))*0.1
    
    #calculando la cantidad de ventanas de la señal
    window=2500     #tamano de la ventana(10 segundos)
    despl=250       #solapamiento 1 segundo
    cant_windows=int((n-window)/despl)
     
    #cantidad de características por cada ventana
    cant_caract=47
    
    #creando matriz de caracteristicas para el entrenamiento
    matrix_caract=np.zeros((cant_windows,cant_caract))
    
    #creando matriz de salidas o clases
    Clas=np.zeros(cant_windows,dtype='int')
    
    x=0  
    for j in range(0,(n-window-1),despl): #desplazamiento por ventanas con solapamiento de 1 segundo
        #muestras de inicio y fin de la ventana actual
        inicio=j
        fin=j+window
    
        #seleccionando el intervalo dentro de la ventana en mitad de la ventana y dos muestras alante y dos detras
        #es decir, en una ventana de 10 segundos anoto para los cuatro segundos centrales, desde 3 hasta 7
        c1_m=inicio + 3*despl
        c2_m=inicio + 7*despl
        
        
        ppg_window=ppgfn[inicio:fin]
        picos= caract.picos(ppg_window,fs,mediana)        
        
        if len(picos)>2:
            #guardar la clase
            Clas[x]=np.argmax(np.bincount(vector_salida[c1_m:c2_m]))
            #extrayendo caracteristicas y gusrdandolas en la matriz
            frec_card=caract.FC2 (picos,fs)                       #característica 1: FC
            matrix_caract[x,0]=frec_card
            pp=caract.PP(picos,fs)
            stdpp=caract.stdPP(pp)                                #característica 2:STD(PP)
            matrix_caract[x,1]=stdpp
            maxpp=caract.maxPP(pp)                                #característica 3:Max(PP)
            matrix_caract[x,2]=maxpp
            minpp=caract.minPP(pp)                                #característica 4:Min(PP)
            matrix_caract[x,3]=minpp
            mediapp=caract.mediaPP(pp)                            #característica 5:Mean(PP)
            matrix_caract[x,4]=mediapp
            medianapp=caract.medianaPP(pp)                        #característica 6:Median(PP)
            matrix_caract[x,5]=medianapp
            inter_cuartilpp=caract.ricPP(pp)                      #característica 7:Rango_intercuartil(PP)
            matrix_caract[x,6]=inter_cuartilpp
            
            pies=caract.inicio(ppg_window,picos)
            areas=caract.calc_area(ppg_window,pies)
            stdA=caract.stdA(areas)                               #característica 8:STD(Areas)
            matrix_caract[x,7]=stdA
            maxArea=caract.maxA(areas)                            #característica 9:Max(Areas)
            matrix_caract[x,8]=maxArea
            minArea=caract.minA(areas)                            #característica 10:Min(Areas)
            matrix_caract[x,9]=minArea
            mediaArea=caract.mediaA(areas)                        #característica 11:Mean(Area)
            matrix_caract[x,10]=mediaArea
            medianaArea=caract.medianaA(areas)                    #característica 12:Median(Area)
            matrix_caract[x,11]=medianaArea
            inter_cuartilA=caract.ricA(areas)                     #característica 13:Rango_intercuartil(Areas)
            matrix_caract[x,12]=inter_cuartilA
            
            m1=caract.m1(picos,pies,ppg_window)             
            stdM1=caract.stdm1(m1)                                #característica 14:STD(pendientes de subida)
            matrix_caract[x,13]=stdM1
            maxM1=caract.maxm1(m1)                                #característica 15:Max(pendientes de subida)
            matrix_caract[x,14]=maxM1
            minM1=caract.minm1(m1)                                #característica 16:Min(pendientes de subida)
            matrix_caract[x,15]=minM1
            mediaM1=caract.mediam1(m1)                            #característica 17:Mean(pendientes de subida)
            matrix_caract[x,16]=mediaM1
            medianaM1=caract.medianam1(m1)                        #característica 18:Median(pendientes de subida)
            matrix_caract[x,17]=medianaM1
            inter_cuartilM1=caract.ricm1(m1)                      #característica 19:Rango_intercuartil(pendientes de subida)
            matrix_caract[x,18]=inter_cuartilM1
            
            m2=caract.m2(picos,pies,ppg_window)             
            stdM2=caract.stdm2(m2)                                #característica 20:STD(pendientes de bajada)
            matrix_caract[x,19]=stdM2
            maxM2=caract.maxm2(m2)                                #característica 21:Max(pendientes de bajada)
            matrix_caract[x,20]=maxM2
            minM2=caract.minm2(m2)                                #característica 22:Min(pendientes de bajada)
            matrix_caract[x,21]=minM2
            mediaM2=caract.mediam2(m2)                            #característica 23:Mean(pendientes de bajada)
            matrix_caract[x,22]=mediaM2
            medianaM2=caract.medianam2(m2)                        #característica 24:Median(pendientes de bajada)
            matrix_caract[x,23]=medianaM2
            inter_cuartilM2=caract.ricm2(m2)                      #característica 25:Rango_intercuartil(pendientes de bajada)
            matrix_caract[x,24]=inter_cuartilM2
            
            f, P=caract.Densidad_espectral(ppg_window,fs)
            stdP=caract.stdDE(P)                                  #característica 26:Desviacion estandar de la Densidad espectral
            matrix_caract[x,25]=stdP
            mdnP=caract.medianaDE(P)                              #característica 27:Mediana de la Densidad espectral
            matrix_caract[x,26]=mdnP
            mdP=caract.mediaDE(P)                                 #característica 28:Promedio de la Densidad espectral
            matrix_caract[x,27]=mdP
            skwP=caract.skwDE(P)                                  #característica 29:Skewness de la Densidad espectral
            matrix_caract[x,28]=skwP
            kurtP=caract.KurtDE(P)                                #característica 30: kurtosis de la Densidad espectral
            matrix_caract[x,29]=kurtP
            PT=caract.areaDE(P)                                   #característica 31:Potencia total
            matrix_caract[x,30]=PT
            
            entropia=caract.entropia(ppg_window)                  #característica 32: Entropia de la ventana
            matrix_caract[x,31]=entropia                          
            
            stdap=caract.stdAP(ppg_window,picos,pies)                  #característica 33:STD(amplit picos)
            matrix_caract[x,32]=stdap
            maxap=caract.maxAP(ppg_window,picos,pies)                  #característica 34:Max(amplit picos)
            matrix_caract[x,33]=maxap
            minap=caract.minAP(ppg_window,picos,pies)                  #característica 35:Min(amplit picos)
            matrix_caract[x,34]=minap
            mediaap=caract.mediaAP(ppg_window,picos,pies)              #característica 36:Mean(amplit picos)
            matrix_caract[x,35]=mediaap
            medianap=caract.medianaAP(ppg_window,picos,pies)           #característica 37:Median(amplit picos)
            matrix_caract[x,36]=medianap
            inter_cuartilap=caract.ricAP(ppg_window,picos,pies)        #característica 38:Rango_intercuartil(amplit picos)
            matrix_caract[x,37]=inter_cuartilap
            
            deltapp2=caract.deltaPP2(pp)                               #característica 39:variacion de PP
            matrix_caract[x,38]=deltapp2
            
            deltap2=caract.deltaP2(picos,pies,ppg_window)              #característica 40:variacion de la amplitud de los picos 
            matrix_caract[x,39]=deltap2
            
            frecDE=caract.frec_p(P,f)                                  #característica 41: Frecuencia de la maxima potencia
            matrix_caract[x,40]=frecDE
            
            fmnDE=caract.frecmedianaDE(mdnP,P,f)                       #característica 42: Frecuencia de la mediana de la densidad espectral                  
            matrix_caract[x,41]=fmnDE
            
            fmDE=caract.frecmediaDE(mdP,P,f)                           #característica 43: Frecuencia de la media de la densidad espectral
            matrix_caract[x,42]=fmDE
            
            corr_Win=caract.corr_WIN(ppg_window)
            matrix_caract[x,43]=corr_Win
            
            FCv1,FCv2,FCv3=caract.vFCI (picos,fs)
            matrix_caract[x,44]=FCv1
            matrix_caract[x,45]=FCv2
            matrix_caract[x,46]=FCv3
            
            x=x+1

    savemat(ENT[k]+"_c"+".mat",{"matrix":matrix_caract})
    savemat(ENT[k]+"_class"+".mat",{"Salidas":Clas})