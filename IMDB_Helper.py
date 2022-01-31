# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 17:03:06 2021

@author: Sana Ullah,Leandro Martin Corona,Zoltán György Varga,Paulo Gonzalez Isaurralde
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import imdb
import warnings
from scipy import stats 
from scipy.stats import zscore,shapiro
import seaborn as sn
import statsmodels.api as sm
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler; from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression # 75|25 splits below
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score


d = {'Section': ['Section 1','Section 2','Section 3','Section 4'],'Name' : ['Sana Ulah Khan','Leandro Martin Corona','Zoltán György Varga','Paulo Gonzalez Isaurralde']}  
#d2 = {'Names' : ['Sana Ulah Khan','Leandro Martin Corona','Zoltán György Varga','Paulo Gonzalez Isaurralde']} 
contribution = pd.DataFrame(d, index=['Introduction + Data analysis + Data Extraction + Cleaning & visualization', 'Prediction Challenge', 'Exploratory Component', 'Conclusions'])

