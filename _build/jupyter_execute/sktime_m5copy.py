#!/usr/bin/env python
# coding: utf-8

# # Timeseries forecasting using Sktime

# Innerhalb dieses Notebooks wird mithilfe der Python Bibliothek Sktime eine Prognose für die zukünftigen Verkäufe von (einem) Produkt(en) durchgeführt.

# # Import Data and Packages
# 
# Für die Durchführung der Zeitreihenanalyse wird die Python-Bibliothek Sktime verwendet. Zusätzlich werden Visualisierungen mit der Bibliothek Matplotlib durchgeführt.

# In[1]:


from warnings import simplefilter

import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsRegressor
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import (
    TransformedTargetForecaster,
    make_reduction,
)
from sktime.forecasting.model_selection import (
    temporal_train_test_split,
)
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.performance_metrics.forecasting import (
    MeanAbsolutePercentageError,
    mean_absolute_percentage_error,
)
from sktime.transformations.series.detrend import Deseasonalizer, Detrender
from sktime.utils.plotting import plot_series

simplefilter("ignore", FutureWarning)
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt


# Für die Anwendung der Zeitreihenanalyse verwenden wir die Datensätze **Sales_train_evaluation.csv** und **Calender.csv**. Diese werden mithilfe von Pandas eingelesen.

# In[2]:


df_stv = pd.read_csv("/Users/lukas/Documents/MLOPS/PL/sales_train_evaluation.csv")
df_cal = pd.read_csv("/Users/lukas/Documents/MLOPS/PL/calendar.csv")


# # Overview
# 
# Zuerst betrachten wir die relevanten Dataframes df_stv und df_cal.

# In[3]:


df_stv.head()


# In[4]:


df_cal.head()


# # Data format
# 
# Für die Anwendung der Bibliothek Sktime müssen die Daten angepasst und formatiert werden. Hierfür wird zuerst ein Dataframe so angepasst, dass er zwei Spalten aufweist das Datum und die Verkaufszahlen pro Datum (Tag) eines Produktes. <br><br>
# *Anzumerken ist dabei, dass die Zeitreihenprognose in diesem Beispiel vereinfacht umgesetzt wird*

# Zuerst fügen wir die Dataframes "df_stv" und "df_cal" basierend auf der Tagesnummerierung zusammen.

# In[5]:


#Erstellung einer Liste mit den nummerierten Tagen "d"
list_d_ = [i for i in df_stv.columns if 'd_' in i] 
list_d_
list_none = []
for i in range(6):
    list_none.append("None")
list_d_ = list_none + list_d_


# In[6]:


# Zusammenführung von date und Tagesnummerierung
date_d = dict(zip(df_cal.date, df_cal.d))


# In[7]:


# Transpose df_stv
df_stv = df_stv.T


# In[8]:


# Erstellung der Spalte "d" mit Daten aus der Liste "list_d_"
df_stv["d"] = list_d_


# In[9]:


# Zusammenführung der Dataframes basierend auf der Spalte "d"
df_stv = pd.merge(df_stv, df_cal, on="d")


# In[10]:


df_stv.head(10)


# Nun reduzieren wir den Dataframe auf die nötigsten Spalten.

# In[11]:


df_example = df_stv[["date",0]]


# In[12]:


df_example.head()


# In[13]:


df_example.info()


# Anschließend passen wir den Datentyp des Produkts an.

# In[14]:


df_example[0] = df_example[0].astype(float)


# Um besser mit der Bibliothek Matplotlib arbeiten zu können wandeln wir den Dataframe in eine Pandasseries um. Zuvor setzen wir jedoch die Spalte "date" als Index.

# In[15]:


df_series = df_example.set_index('date') # Index "date" wird gesetzt


# In[16]:


df_series.index = pd.to_datetime(df_series.index, errors='coerce') # Index zu Datetime konvertieren
df_series.asfreq('D') # Hinzufügen der Frequenz


# In[17]:


df_series.info()


# In[18]:


m5_series = df_series.squeeze() # Umwandlung mittels Squeeze


# In[19]:


type(m5_series)


# In[20]:


m5_series


# Nun wurden die Daten soweit vorbereitet, dass die Prognose und benötigte Plots zur Visualisierung effektiv dargestellt werden können.
# 
# ## Sales Overview
# 
# Mithilfe von Matplotlib visualisieren wir die Produktverkäufe.

# In[21]:


plot_series(m5_series);


# index represents the timepoints

# In[22]:


m5_series.index


# In[23]:


m5_series = m5_series.asfreq("D")


# # Data Split
# 
# Nun führen wir einen einfachen Datensplit durch.

# In[24]:


y_train, y_test = temporal_train_test_split(m5_series, test_size=36)
plot_series(y_train, y_test, labels=["y_train", "y_test"])
print(y_train.shape[0], y_test.shape[0])


# # Forecasting horizon

# Nun definieren wir den Vorraussagezeitraum (fh).

# In[25]:


fh = np.arange(len(y_test)) + 1
fh


# In[26]:


fh = np.array([2, 5])  # 2nd and 5th step ahead


# In[27]:


fh = ForecastingHorizon(y_test.index, is_relative=False)
fh


# # Model - Timeseries forecasts

# In[28]:


#Forecast basierend auf dem letzten Wert
forecaster = NaiveForecaster(strategy="last")

#Model fit
forecaster.fit(y_train)

#predict values
y_pred = forecaster.predict(fh)

plot_series(y_test, y_pred, labels=["y_test", "y_pred"])
mean_absolute_percentage_error(y_pred, y_test)


# In[29]:


#Forecast basierend auf den Werten der Vorwoche (letzten Woche der Trainingsdaten)
forecaster = NaiveForecaster(strategy="last", sp=7)

forecaster.fit(y_train)
y_pred = forecaster.predict(fh)

plot_series(y_test, y_pred, labels=["y_test", "y_pred"])

mean_absolute_percentage_error(y_pred, y_test)


# # Sales in California
# ## Data Format
# 
# Nun sollen nicht nur die Verkäufe eines Produkts sondern die Verkäufe von allen Stores in Kalifornien vorhergesagt werden. Hierfür werden die Zeilen der Produkte zusammenaddiert und wie in den oberen Schritten angepasst.

# In[30]:


# Import data
df_stv_ca = pd.read_csv("/Users/lukas/Documents/MLOPS/PL/sales_train_evaluation.csv")
df_cal_ca = pd.read_csv("/Users/lukas/Documents/MLOPS/PL/calendar.csv")
# Hinzufügen von Zeile, welche Verkäufe von Kalifornien zusammenfasst
df_stv_ca = df_stv_ca[df_stv_ca["state_id"] == "CA"]
total = df_stv_ca.sum()
df_stv_ca = df_stv_ca.append(total,ignore_index=True)
# Zusammenführen der Dataframes
list_d_ = [i for i in df_stv_ca.columns if 'd_' in i] 
list_d_
list_none = []
for i in range(6):
    list_none.append("None")
list_d_ = list_none + list_d_
date_d = dict(zip(df_cal_ca.date, df_cal_ca.d))
df_stv_ca = df_stv_ca.T
df_stv_ca["d"] = list_d_
df_stv_ca = pd.merge(df_stv_ca, df_cal_ca, on="d")
# Selektion der relevanten Spalten
df_ca = df_stv_ca[["date",12196]]
df_ca[12196] = df_ca[12196].astype(float)
df_ca = df_ca.set_index('date')
df_ca.index = pd.to_datetime(df_ca.index, errors='coerce')
df_ca.asfreq('D')
m5_series_total_ca = df_ca.squeeze()


# In[31]:


m5_series_total_ca = m5_series_total_ca.asfreq("D") # Frequenz setzen


# In[32]:


m5_series_total_ca


# Nun erstellen wir ein einfaches Plot, um einen Überblick über die Verkäufe in Kalifornien zu erhalten.

# In[33]:


plot_series(m5_series_total_ca);


# ## Data Split
# 
# Anschließend folgt der Datensplit in Trainings- und Testdaten.

# In[34]:


y_train_ca, y_test_ca = temporal_train_test_split(m5_series_total_ca, test_size=36)
plot_series(y_train_ca, y_test_ca, labels=["y_train_ca", "y_test_ca"])
print(y_train_ca.shape[0], y_test_ca.shape[0])


# ## Model

# Nun definieren wir den Vorraussagehorizont (fh_ca).

# In[35]:


fh_ca = ForecastingHorizon(y_test_ca.index, is_relative=False)
fh_ca


# Im nächsten Schritt wird ein einfaches Modell erstellt, um basierend auf der Vorwoche die Anzahl der Verkäufe vorherzusagen. <br> Das Modell dient dabei primär als Vergleichswert.

# In[36]:


#Forecast basierend auf den Werten der Vorwoche (letzten Woche der Trainingsdaten)
forecaster = NaiveForecaster(strategy="last", sp=7)

forecaster.fit(y_train_ca)
y_pred_ca = forecaster.predict(fh_ca)

plot_series(y_test_ca, y_pred_ca, labels=["y_test_ca", "y_pred_ca"])

mape_baseline = mean_absolute_percentage_error(y_pred_ca, y_test_ca)


# Nun wird ein Modell mit k-nearest neighbors zur Prognose der Verkaufszahlen erstellt.

# In[37]:


regressor = KNeighborsRegressor(n_neighbors=2)

forecaster = TransformedTargetForecaster(
    [
        ("deseasonalize", Deseasonalizer(model="multiplicative", sp=7)),
        ("detrend", Detrender(forecaster=PolynomialTrendForecaster(degree=4))),
        (
            "forecast",
            make_reduction(
                regressor,
                scitype="tabular-regressor",
                window_length=15,
                strategy="recursive",
            ),
        ),
    ]
)
forecaster.fit(y_train_ca) # Modell fit
y_pred_ca = forecaster.predict(fh_ca)
plot_series(y_test_ca, y_pred_ca, labels=["y_test_ca", "y_pred_ca"])
mape_kn = mean_absolute_percentage_error(y_pred_ca, y_test_ca)


# Nun vergleichen wir die Modelle basierend auf dem Mean absolute percentage error.

# In[38]:


print("Mean absolute percentage error:",mape_baseline,"Baseline")
print("Mean absolute percentage error:",mape_kn,"KNeighborsRegressor")

