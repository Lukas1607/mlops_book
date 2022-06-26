#!/usr/bin/env python
# coding: utf-8

# # 1. Data Understanding & Data Correction
# 
# Im ersten Teil "Data Understanding & Data Correction" wird versucht, einen ersten Überblick über den Datensatz "Used Car Auction Prices" zu ermöglichen und Probleme innerhalb des Datensatzes in Bezug auf dessen Qualität zu identifizieren. Darüber hinaus wird der Datensatz innerhalb des Notebooks bereinigt, um die Vorbereitung für den folgenden Themenpunkt "Modeling" vorzubereiten.
# 
# 
# # 1.1 Import Data & Packages
# 
# Im ersten Schritt werden die benötigten Python-Bibliotheken importiert, welche im Verlauf des Punktes "Data Understanding & Data Correction" benutzt werden. Besonders relevant ist dabei die Bibliothek Pandas, welche sowohl für den Import des Datensatzes, als auch für dessen Bereinigung verwendet wird. Für sämtliche Visualisierungen wird die Bibliothek Plotly (Graph Objects).

# In[1]:


# Import der Python-Bibliotheken

import pandas as pd

import plotly.express as px
import plotly.graph_objects as go


# Anschließend importieren wir mittels der Pandas-Bibliothek den Datensatz "Used Car Auction Prices", welcher als CSV-Datei vorliegt. Aufgrund der teilweise schlechten Qualität des vorliegenden Datensatzes nutzen wir innerhalb der (Pandas) Funktion "read_csv" den Parameter "error_bad_lines=False", um Zeilen zu überspringen, welche Fehler aufgrund von einer fehlerhaften Dateninserierung auslösen. Eine Erklärung, wie es zu den Fehlern kommt erläutere ich in einem späteren Unterpunkt. Zum anderen setzen wir den Parameter "warn_bad_lines=True", um eine Ausgabe der Fehlerdetails einsehen zu können.

# In[2]:


# Import des Datensatzes

df = pd.read_csv("/Users/lukas/Documents/MLOPS/PL_2/car_prices.csv", error_bad_lines=False,warn_bad_lines=True)


# # 1.2 Data Overview
# 
# Innerhalb des Unterpunkts "Data Overview" erhalten wir einen ersten Überblick zum Datensatz.

# Für den Überblick nutzen wir die Methode "head", um die ersten fünf Zeilen des Dataframes einzusehen.

# In[3]:


df.head()


# ## Hintergrund zum Datensatz
# 
# Das Datenset "Used Car Auction Prices" basiert auf Daten, welche mittels Webscraping aus dem Internet im Jahr 2015 extrahiert wurden. Das Datenset wird seit diesem Zeitpunkt nicht mehr aktualisiert. Im Kontext des Webscrapings könnte die Durchführung der Datenextraktion ein Grund für die Importprobleme in Punkt 1.1 darstellen. Denn gegebenenfalls wurden spezielle zusätzliche Informationen für verschiedene Autos nicht korrekterweise gefiltert, welche letztendlich in eine zusätzliche Spalte inseriert wurden und somit den Fehler auslösen.
# 
# ## Erklärung der Variablen
# 
# Zum besseren Verständnis ist hier eine kurze Übersicht der Variablen aufgeführt mit den jeweiligen Bedeutungen.
# 
# - Date : Produktionsjahr der Autos
# - Make : Marke des Autos
# - Model : Das Modell des Autos
# - Trim : Die Ausführung des Autos
# - Body : Die Karosseriebauform des Autos
# - Transmission : Getriebearten des Autos
# - VIN : Fahrzeugidentifikationsnummer
# - State : Bundesstaat in welchem das Auto verkauft wird
# - Condition : Zustand des Autos
# - Odometer : Kilometerstand des Autos
# - Color : Die (Außen-)Farbe des Autos
# - Interior : Die Farbe der Inneneinrichtung
# - Seller : Der Verkäufer des Autos
# - mmr : Manhiem market record, ein Indikator für den (Markt-)Wert des Autos
# - sellingprice : Der finale Verkaufspreis
# - saledate : Das Verkaufsdatum

# Aufgrund der Bedeutung der Variablen werden einige Variablen zur vermeintlich besseren Verwendung umbenannt.

# In[4]:


# Spalten umnennen
df = df.rename(columns={"make":"brand","body":"type","sellingprice":"price","saledate":"date"})


# Für weitere Detailinformationen nutzen wir die Methoden "info" und "describe", um zum einen, ein Überblick über die Datentypen und die "Non-null"-Werte zu schaffen und zum anderen erhalten wir mit der Methode "describe" einen Überblick über die nummerischen Werte des Datensatzes und deren Verteilung.

# In[5]:


df.info()


# In[6]:


df.describe()


# # 1.3 Data Correction
# 
# Für die spätere Verwendung des Datensatzes werden die Variablen "vin" und "mmr" nicht benötigt. Aus diesem Grund werden beide Variablen entfernt.

# In[7]:


# Entfernen irrelevanter Spalten
df = df.drop(['vin','mmr'],axis=1)


# Nun visualisieren wir die fehlenden Werte innerhalb eines Plots. Hierfür müssen zuerst die fehlenden Werte berechnet und in Prozent umgerechnet werden, um anschließend die kalkulierten Werte in die Visualisierung über zu geben.

# In[8]:


# Missing Value prüfen

# Berechnungen zur Vorbereitung des folgenden Plots
mv_of_df = df.isna().sum()
len_df = len(df)
mv_p_df = 100*(mv_of_df / len_df)
mv_p_r_df = round(mv_p_df)
cn_of_df = df.columns.values.tolist()


# In[9]:


# Visualisierungen für Missing Values

fig = go.Figure([go.Bar(x=cn_of_df, y=mv_p_r_df)])
fig.update_layout(
    xaxis_title="Variablen",
    yaxis_title="Fehlende Werte in %",
    plot_bgcolor="white")

fig.show()


# In[10]:


df.count()


# Aufgrund der großen Anzahl an Einträgen können die fehlenden Werte entfernt werden.

# In[11]:


# Löschen der fehlenden Werte
df = df.dropna()  
df.count()


# In[12]:


print(df.isna().sum()) # Prüfung ob Werte erfolgreich entfernt wurden


# Anschließend prüfen wir den Dataframe noch auf Duplikate mit der Methode "duplicated".

# In[13]:


duplicate_rows_df = df[df.duplicated()]
print("Anzahl doppelter Einträge: ", duplicate_rows_df.shape)


# Nun werden die Ausprägungen der einzelnen Variablen ausgegeben, um mögliche Komplikationen mit Benennung von Marken o. ä. zu identifizieren.

# In[14]:


for col in df:
    print("---------------")
    print(df[col].unique())


# Insbesondere die Variablen "brand", "model" und "type" sind von der Problematik der unterschiedlichen Schreibweise betroffen. Deshalb wird im Folgenden alle Ausprägungen in Kleinschrift geändert. Dabei werden die Ausprägungen vor und nach Anpassung verglichen.

# In[15]:


bu = df['brand'].unique()
mu = df['model'].unique()
tu = df['type'].unique()
print("Ausprägungen der Variable Brand:",bu.__len__())
print("Ausprägungen der Variable Model:",mu.__len__())
print("Ausprägungen der Variable Type:",tu.__len__())


# In[16]:


# Ändere die Ausprägungen der Variablen, um Duplikate mit unterschiedlicher Rechtschreibung zu reduzieren. Dabei wird alles kleingeschrieben.

bu = df['brand'].unique()
mu = df['model'].unique()
tu = df['type'].unique()

df.brand = df.brand.str.lower()
df.model = df.model.str.lower()
df.type = df.type.str.lower()
df


# In[17]:


bua = df['brand'].unique()
mua = df['model'].unique()
tua = df['type'].unique()

print("Ausprägungen der Variable Brand zuvor:",bu.__len__(),"danach:", bua.__len__())
print("Ausprägungen der Variable Model zuvor:",mu.__len__(),"danach:", mua.__len__())
print("Ausprägungen der Variable Type zuvor:",tu.__len__(),"danach:",tua.__len__())


# Nun exportieren wir den angepassten Datensatz, um diesen in einem späteren Kapitel wiederverwenden zu können.

# In[18]:


df.to_csv(r'/Users/lukas/Desktop/car_price_final.csv', index = False)


# # 1.4 Data Exploration
# 
# Im Punkt "Data Exploration" werden die vorliegenden Daten genauer untersucht, da bisher nur ein geringes Wissen über deren Zusammenhänge und Verteilungen vorliegen.
# Alle Visualisierungen werden mithilfe der Python-Bibliothek Plotly (Graph Objects & Express) dargestellt.

# Aufgrund des hohen Ressourcenverbrauchs einzelner Visualisierungen wird im Folgenden ein Sample-Datenset erstellt, welches deutlich weniger Beobachtungswerte umfasst. Hierdurch kann die Datenvisualisierung effizienter umgesetzt werden.

# In[19]:


#Erstellung des Sample-Dataframes
df_sample = df.sample(n=1000)


# Nun erstellen wir für die Variable "year" ein Histogramm, um einen genauren Einblick über die Verteilungen innerhalb der Variable zu erhalten.

# In[20]:


# Histogramm für Variable "year"
fig = go.Figure(data=[go.Histogram(x=df_sample["year"])])
fig.update_layout(
    xaxis_title="Jahre",
    yaxis_title="Anzahl der Autos",
    plot_bgcolor="white")
fig.show()


# Die Variable "price" wird im Folgenden primär betrachtet, da u.a. der Preis im späteren Modell prognostiziert werden soll. Aus diesem Grund wird auch für die Variable "price" ein Histogramm erstellt.

# In[21]:


# Histogramm für die Variable "price"
fig = go.Figure(data=[go.Histogram(x=df_sample["price"])])
fig.update_layout(
    xaxis_title="Preis",
    yaxis_title="Anzahl der Autos",
    plot_bgcolor="white")
fig.show()


# Damit die Verhältnisse besser eingesehen werden können wird zusätzlich noch ein Boxplot erstellt.

# In[22]:


# Boxplot für die Variable "price"
fig = go.Figure(data=[go.Box(y=df_sample["price"])])
fig.update_layout(
    xaxis_title="Preis",
    yaxis_title="Preis in USD",
    plot_bgcolor="white")
fig.show()


# Nun verschaunlichen wir die Variablen "price" und "year" innerhalb eines Scatterplots, um mögliche Zusammenhänge der beiden Variablen erkennen zu können. Hierfür wird die Funktion "strip" verwendet, um die einzelnen Datenpunkte als "gejitterte" Markierungen innerhalb der einzelnen Jahren darzustellen. Eine alternative Visualisierung wäre beispielsweise ein Boxplot, welches u.a. bei der betrachtung weiterer Variablen verwendet wird.

# In[23]:


# Visualisierung Jahr / Preis
fig = px.strip(df_sample, x="year", y="price", labels={"price":"Preis","year":"Jahr"})
fig.update_layout({
'plot_bgcolor': 'rgba(0, 0, 0, 0)',
'paper_bgcolor': 'rgba(0, 0, 0, 0)',
})
fig.show()


# Da die "strip"-Funktion keine Möglichkeit zum aktuellen Zeitpunkt aufweist eine Trendlinie als Parameter hinzuzufügen wird im Folgenden eine weitere Visualisierung mit "scatter" durchgeführt.

# In[24]:


# Visualisierung Jahr / Preis mit Trendlinie
fig = px.scatter(df_sample, x="year", y="price", trendline="ols", labels={"price":"Preis","year":"Jahr"})
fig.update_layout({
'plot_bgcolor': 'rgba(0, 0, 0, 0)',
'paper_bgcolor': 'rgba(0, 0, 0, 0)',
})
fig.show()


# Anschließend wird noch eine Visualisierung mit der Ergänzung der Variable "transmission" dargestellt, um mögliche Unterschiede zwischen der Schaltung festzustellen.

# In[25]:


# Visualisierung Jahr / Preis inkl. Schaltung
fig = px.strip(df_sample, x="year", y="price", color="transmission", labels={"price":"Preis","year":"Jahr"})
fig.update_layout({
'plot_bgcolor': 'rgba(0, 0, 0, 0)',
'paper_bgcolor': 'rgba(0, 0, 0, 0)',
})
fig.show()


# Nun betrachten wir die Variable "brand" innerhalb eines Histogramms. Dabei sortieren wir die Marken nach der Häufigkeit mithilfe der Einstellung "categoryorder" für die x-Achse.

# In[26]:


# Histogramm für die Variable "brand"
fig = go.Figure(data=[go.Histogram(x=df_sample["brand"])])
fig.update_layout(
    xaxis_title="Marke",
    xaxis={"categoryorder":"total descending"},
    yaxis_title="Anzahl der Autos",
    plot_bgcolor="white")

fig.show()


# Für einen besseren Überblick über die einzelnen Marken und die jeweiligen Preise wird ein Boxplot erstellt.

# In[27]:


# Boxplot für die Variable "price" und "brand"
fig = go.Figure(data=[go.Box(y=df_sample["price"], x=df_sample["brand"])])
fig.update_layout(
    xaxis_title="Marke",
    yaxis_title="Preis in USD",
    plot_bgcolor="white")
fig.show()


# Des Weiteren stellen wir die Variable "condition" in einem Histogramm und später in Kombination mit der Variable "price" dar.

# In[28]:


# Histogramm für die Variable "condition"
fig = go.Figure(data=[go.Histogram(x=df_sample["condition"])])
fig.update_layout(
    xaxis_title="Zustand",
    yaxis_title="Anzahl der Autos",
    plot_bgcolor="white")

fig.show()


# In[29]:


# Visualisierung Zustand / Preis mit Trendlinie
fig = px.scatter(df_sample, x="condition", y="price", trendline="ols", labels={"price":"Preis","condition":"Zustand"})
fig.update_layout({
'plot_bgcolor': 'rgba(0, 0, 0, 0)',
'paper_bgcolor': 'rgba(0, 0, 0, 0)',
})
fig.show()


# Zu guter Letzt wird die Variable "odometer" visualisiert. Neben einem Histogramm auch in Kombination mit den Variablen "condition" und "price".

# In[30]:


# Histogramm für die Variable "odometer"
fig = go.Figure(data=[go.Histogram(x=df_sample["odometer"])])
fig.update_layout(
    xaxis_title="Meilenzähler",
    yaxis_title="Anzahl der Autos",
    plot_bgcolor="white")

fig.show()


# In[31]:


# Visualisierung Zustand / Meilenzähler mit Trendlinie
fig = px.scatter(df_sample, x="condition", y="odometer", trendline="ols", labels={"price":"Preis","year":"Jahr"})
fig.update_layout({
'plot_bgcolor': 'rgba(0, 0, 0, 0)',
'paper_bgcolor': 'rgba(0, 0, 0, 0)',
})
fig.show()


# In[32]:


# Visualisierung Meilenzähler / Preis mit Trendlinie
fig = px.scatter(df_sample, x="odometer", y="price", trendline="ols", labels={"price":"Preis","year":"Jahr"})
fig.update_layout({
'plot_bgcolor': 'rgba(0, 0, 0, 0)',
'paper_bgcolor': 'rgba(0, 0, 0, 0)',
})
fig.show()

