#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 09:27:48 2022

Realizando el proceso de selección de características utilizando el algoritmo 
DecisionTreeClassifier. Los algoritmos de bosques aleatorios como los árboles 
de clasificación y regresión (CART) ofrecen puntuaciones de importancia basadas 
en la reducción del criterio utilizado para seleccionar puntos de división.

Sus resultados pueden variar dada la naturaleza estocástica del algoritmo o procedimiento
de evaluación, o las diferencias en la precisión numérica. Considere ejecutar el código
varias veces y utilice para comparar el resultado promedio.

@author: aasl
"""

# decision tree for feature importance on a classification problem
#from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot
from scipy.io import loadmat
import numpy as np
from pylab import plot, show, title, xlabel, ylabel, subplot


# definir dataset
Caract_Norm=loadmat("Caracteristicas.mat")
X = Caract_Norm['caract']
Clases=loadmat("Clases2.mat")
Y = Clases['Clases2']
# definir the model
#model0 = DecisionTreeClassifier()
model1 = RandomForestClassifier()

result=np.zeros((10,47)) 
for i in range(0,10):
    
    # fit the model
    model1.fit(X, Y)
    # get importance
    importance = model1.feature_importances_
#    # summarize feature importance
#    for i,v in enumerate(importance):
#    	print('Feature: %0d, Score: %.5f' % (i,v))
#    # plot feature importance
#    pyplot.bar([x for x in range(len(importance))], importance)
#    pyplot.show()
    
    result[i,:]=importance.reshape(-1,1).squeeze()
media=np.mean(result,axis=0)
srt_idx = np.argsort(-1*media.squeeze()) + 1
srt_val = np.sort(-1*media)
ax = pyplot.figure()
pyplot.bar([x for x in range(len(srt_val))], np.abs(srt_val))
ax.axes[0].set_xticks(range(len(srt_val)))
ax.axes[0].set_xticklabels([str(ind) for ind in srt_idx])
xlabel('Características')
ylabel('Nivel de importancia')
# title ()
