#!/usr/bin/env python
# coding: utf-8

# # Dashboard
# 
# Das gesamte Dashboard wurde mit der Python-Bibliothek Streamlit erstellt. Das Dashboard teilt sich dabei in vier Sichten auf: Home, EDA, Beispiel_Prognose und Individuelle Preisprognose für Gebrauchtwagen. <br><br>
# Die Home bzw. Startseite gibt Nutzern einen kurzen Überblick über die Möglichkeiten und Optionen des Dashboards. Innerhalb der Seite EDA werden hingegen einige Visualisierungen aus der explorativen Datenanalyse dargestellt.
# In der Sicht Beispiel_Prognose können Preisprognosen des Modells dem "echten" Verkaufspreis gegenübergestellt werden dabei werden zufällige Gebrauchtwagen aus dem Datensatz herangezogen. Innerhalb der Seite zu individuellen Preisprognose können individuell Eingaben getätigt werden, um für eigene Autokonfigurationen Verkaufspreise vorherzusagen.

# ## Home
# 
# Die Homeseite ist einfach strukturiert und stellt lediglich Text innerhalb des Dashboards dar.

# In[1]:


#-------------------
# Import Python Bibliotheken
import streamlit as st 

#-------------------
# Text der Startseite
st.write("# Data Science and MLOps")

st.sidebar.success("Wähle eine Seite aus")

st.markdown(
    """
    Innerhalb des Streamlit Dashboards werden die Ergebnisse des Projekts "Data Science and MLOps" für den Datensatz "Used auction car prices" dargestellt.
    
    ### Wähle eine der folgenden Seiten über das Seitenmenü aus:
    - Beispiel Prognose
    - EDA
    - Individuelle Preisprognose für Gebrauchtwagen
    """
)


# ## EDA
# 
# Plots innerhalb der EDA-Seite wurden mit Plotly dargestellt. Die Visaulisierungen sind dabei ein Auszug aus dem Punkt Data Understanding & Correction.

# In[ ]:


#-------------------
# Import Python Bibliotheken

import streamlit as st

import pandas as pd

import plotly.graph_objects as go
import plotly.express as px
#-------------------
# Titel und Beschreibung

st.title("Dashboard MLOPS-Projekt")

st.write("Visualisierungen ausgewählter Variablen des Datensates Used auction car prices.")

#-------------------
# Visualisierungen

# Import des Datensatzes
df = pd.read_csv("/Users/lukas/Desktop/car_price_final.csv")

#Erstellung des Sample-Dataframes
df_sample = df.sample(n=1000)

# Histogramm für Variable "year"
fig = go.Figure(data=[go.Histogram(x=df_sample["year"])])
fig.update_layout(
    title="Histogramm: Jahr",
    xaxis_title="Jahre",
    yaxis_title="Anzahl der Autos",
    plot_bgcolor="white")
# Plot!
st.plotly_chart(fig)

# Histogramm für die Variable "price"
fig = go.Figure(data=[go.Histogram(x=df_sample["price"])])
fig.update_layout(
    title="Histogramm: Preis",
    xaxis_title="Preis",
    yaxis_title="Anzahl der Autos",
    plot_bgcolor="white")
st.plotly_chart(fig)

# Boxplot für die Variable "price"
fig = go.Figure(data=[go.Box(y=df_sample["price"])])
fig.update_layout(
    title="Boxplot: Preis",
    xaxis_title="Preis",
    yaxis_title="Preis in USD",
    plot_bgcolor="white")
st.plotly_chart(fig)

# Visualisierung Jahr / Preis
fig = px.strip(df_sample, x="year", y="price", labels={"price":"Preis","year":"Jahr"}, title="Preis / Jahr")
fig.update_layout({
'plot_bgcolor': 'rgba(0, 0, 0, 0)',
'paper_bgcolor': 'rgba(0, 0, 0, 0)',
})
st.plotly_chart(fig)

# Visualisierung Jahr / Preis inkl. Schaltung
fig = px.strip(df_sample, x="year", y="price", color="transmission", labels={"price":"Preis","year":"Jahr"},title="Preis / Jahr inkl. Schaltung")
fig.update_layout({
'plot_bgcolor': 'rgba(0, 0, 0, 0)',
'paper_bgcolor': 'rgba(0, 0, 0, 0)',
})
st.plotly_chart(fig)


# ## Beispiel Prognose

# In[ ]:


#-------------------
# Import Python Bibliotheken

import streamlit as st
import numpy as np
import pandas as pd

import tensorflow as tf 
from tensorflow.keras import layers

#-------------------
# Titel und Beschreibung

st.title("Dashboard MLOPS-Projekt")

st.write("Beispielprognose für Gebrauchtwagen in den USA!")

#-------------------
# Import des Datensatzes
df = pd.read_csv("/Users/lukas/Desktop/car_price_final.csv")

#Erstellung des Sample-Dataframes
df_sample = df.sample(n=100)

st.write("Zufälliger Auszug aus dem Datensatz:")
st.dataframe(data=df_sample)

st.write("Vorhersage für ein zufälliges Auto aus dem Datensatz:")
if st.button("Wähle ein zufälliges Auto aus"):
    df_sample_input = df.sample()
    year = df_sample_input.iloc[0]["year"]
    brand = df_sample_input.iloc[0]["brand"]
    model = df_sample_input.iloc[0]["model"]
    trim = df_sample_input.iloc[0]["trim"]
    type = df_sample_input.iloc[0]["type"]
    transmission = df_sample_input.iloc[0]["transmission"]
    state = df_sample_input.iloc[0]["state"]
    condition = df_sample_input.iloc[0]["condition"]
    odometer = df_sample_input.iloc[0]["odometer"]
    color = df_sample_input.iloc[0]["color"]
    interior = df_sample_input.iloc[0]["interior"]
    seller = df_sample_input.iloc[0]["seller"]
    price = df_sample_input.iloc[0]["price"]

    # Ausgabe der Inputs
    st.write('Ausgewählt wurde:', 
              "Baujahr: ",year,
              "Marke: ",brand,
             "Modell: " , model,
             "Trim: " , trim,
             "Type: " , type,
             "Transmission: " , transmission,
             "State: " , state,
              "Condition: " , condition,
             "Odometer: " , odometer,
             "Color: " , color,
             "Interior: " , interior,
             "Seller: " , seller)
    st.write("Korrekter Preis in USD:", price)
    
    # Modell

    input_data = {
        "year":year,
        "brand":brand,
        "model":model,
        "trim": trim,
        "type": type,
        "transmission": transmission,
        "state": state,
        "condition": condition,
        "odometer": odometer,
        "color": color,
        "interior": interior,
        "seller": seller,
    }

    reloaded_model = tf.keras.models.load_model('keras_cars')

    input_dict = {name: tf.convert_to_tensor([value]) for name, value in input_data.items()}

    predictions = reloaded_model.predict(input_dict)

    st.write("------Prognose-------")
    st.write("Prognose in USD:", predictions[0][0])


# ## Individuelle Preisprognose
# 
# Innerhalb der Seite wird exemplarisch dargestellt, wie Abhängigkeiten zwischen Inputfeldern mithilfe einfacher If-Abfragen umgesetzt werden können. Eine verbesserte Lösung der Abhängigkeiten könnte mit einer Anbindung an eine Datenbank oder API bzw. eine JSON-Datei mit allen Informationen zu möglichen Konfigurationen umgesetzt werden.

# In[ ]:


#-------------------
# Import Python Bibliotheken

import streamlit as st
import numpy as np
import pandas as pd

import tensorflow as tf 
from tensorflow.keras import layers

#-------------------
# Titel und Beschreibung

st.title("Dashboard MLOPS-Projekt")

st.write("Prognose für Gebrauchtwagen in den USA")

#-------------------
# Input Felder

year = st.slider(
     'Wähle das Baujahr aus',
     2000, 2015, 2010)

brand = st.selectbox(
     'Wählen Sie die Automarke aus:',
     ('bmw', 'kia', 'mercedes'))

if brand == "bmw":
     model = st.selectbox(
     'Wählen Sie das Modell aus:',
     ('1 series', '2 series', '3 series','4 series','5 series','6 series','7 series','8 series'))

     if model == "1 series":
          trim = st.selectbox(
          'Wählen Sie die Trim aus:',
          ('128i','135i','135is'))

          type = st.selectbox(
               'Wählen Sie den Type aus:',
               ('convertible','coupe'))
     
     elif model == "2 series":
          trim = st.selectbox(
          'Wählen Sie die Trim aus:',
          ('228i','M235i'))

          type = st.text_input('Geben Sie den Type ein:', 'coupe')
     
     elif model == "3 series":
          trim = st.selectbox(
          'Wählen Sie die Trim aus:',
          ('318i','320i','323i','325i','328i','330i'))

          type = st.selectbox(
               'Wählen Sie den Type aus:',
               ('convertible','coupe','sedan'))
     
     elif model == "4 series":
          trim = st.selectbox(
          'Wählen Sie die Trim aus:',
          ('428i','435i'))

          type = st.selectbox(
               'Wählen Sie den Type aus:',
               ('convertible','coupe'))
     
     elif model == "5 series":
          trim = st.selectbox(
          'Wählen Sie die Trim aus:',
          ('525i','528i','530i','535i','540i'))

          type = st.selectbox(
               'Wählen Sie den Type aus:',
               ('sedan','wagon'))

     elif model == "6 series":
          trim = st.selectbox(
          'Wählen Sie die Trim aus:',
          ('640i','650i'))

          type = st.selectbox(
               'Wählen Sie den Type aus:',
               ('convertible','coupe'))

     elif model == "7 series":
          trim = st.selectbox(
          'Wählen Sie die Trim aus:',
          ('740i','745i','750i'))

          type = st.text_input('Geben Sie den Type ein:', 'sedan')
     
     else:
          trim = st.text_input('Geben Sie die Trim ein:', '850i')

          type = st.text_input('Geben Sie den Type ein:', 'coupe')

else:
     model = st.text_input('Geben Sie das Modell an', '')

     trim = st.text_input('Geben Sie die Trim ein:', '')

     type = st.text_input('Geben Sie den Type ein:', '')
     

transmission = st.selectbox(
     'Wählen Sie die Transmission aus:',
     ('manual', 'automatic'))

state = st.selectbox(
     'Wählen Sie den US-Staat aus:',
     ('ca','tx','az','wi','fl','ny'))

condition = st.slider(
     'Wähle das Baujahr aus',
     1.0, 5.0, 3.0, step=0.1 )

odometer = st.number_input('Gebe den Kilometerstand an', 0)

color = st.selectbox(
     'Wählen Sie eine Farbe aus:',
     ('white','gray','black','silver','brown','beige','blue','red'))

interior = st.selectbox(
     'Wählen Sie eine Farbe aus:',
     ('white','gray','black','beige','blue','red'))

if brand == "bmw":
     seller = st.text_input('Gebe den Verkäufer an', 'the hertz corporation')

else:
     seller = st.text_input('Gebe den Verkäufer an', '')

#-------------------
# Ausgabe der Inputs

st.write('Ausgewählt wurde:', 
          "Baujahr: ",year,
           "Marke: ",brand,
           "Modell: " , model,
           "Trim: " , trim,
           "Type: " , type,
           "Transmission: " , transmission,
           "State: " , state,
           "Condition: " , condition,
           "Odometer: " , odometer,
           "Color: " , color,
           "Interior: " , interior,
           "Seller: " , seller)

#-------------------#
# Modell

input_data = {
    "year":year,
    "brand":brand,
    "model":model,
    "trim": trim,
    "type": type,
    "transmission": transmission,
    "state": state,
    "condition": condition,
    "odometer": odometer,
    "color": color,
    "interior": interior,
    "seller": seller,
}

reloaded_model = tf.keras.models.load_model('my_hd_classifier')

input_dict = {name: tf.convert_to_tensor([value]) for name, value in input_data.items()}

predictions = reloaded_model.predict(input_dict)

st.write("Prognose in USD:")
st.write(predictions[0][0])


# 
