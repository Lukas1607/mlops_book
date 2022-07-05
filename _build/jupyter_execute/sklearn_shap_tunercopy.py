#!/usr/bin/env python
# coding: utf-8

# Innerhalb des Notebooks werden der Kerastuner und die SHAP-Bibliothek verwendet.
# 
# ## Import Data & Packages
# 
# Import der benötigten Python-Bibliotheken und benötigten Daten.

# In[1]:


import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers

from tensorboard import notebook
from tensorboard.plugins.hparams import api as hp

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import shap
import keras_tuner as kt

import numpy as np
import datetime

get_ipython().run_line_magic('load_ext', 'tensorboard')

tf.__version__


# In[25]:


# print the JS visualization code to the notebook
shap.initjs()


# Einlesen der Daten mittels Pandas.

# In[26]:


df = pd.read_csv("/Users/lukas/Desktop/car_price_final.csv")


# ## Definition Label

# In[27]:


# Zielvariable
y = df.pop('price')


# ## Data format
# 
# Nun erstellen wir einen Dataframe mit den verwendeten Variablen.

# In[28]:


# Erstellen eines Dataframes mit ausschließlich nummerischen Variablen
list_numerical = ['year', 'condition', 'odometer']

X = df[list_numerical]


# In[29]:


X


# ## Data Split
# 
# Durchführung des Datensplits mithilfe der train_test_split-Funktion.

# In[30]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[31]:


X_test


# In[32]:


X_train


# In[33]:


y_train


# In[34]:


X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)


# ## Feature preprocessing
# 
# Innerhalb des Feature preprocessing wird der StandardScaler auf die Datensplits angewandt, um eine Standartisierung durchzuführen.

# In[35]:


scaler = StandardScaler().fit(X_train[list_numerical]) 

X_train[list_numerical] = scaler.transform(X_train[list_numerical])
X_test[list_numerical] = scaler.transform(X_test[list_numerical])


# # Kerastuner
# 
# Nun wird der Kerastuner verwendet.<br><br> Hierfür wird eine Funktion erstellt, welche ein Keras Modell erstellt und anschließend zurückgibt.

# In[97]:


def build_model(hp):

    model = tf.keras.Sequential()
    
    model.add(layers.Flatten())
    
    model.add(
        layers.Dense(
            # Define the hyperparameter.
            units = hp.Int("units", min_value=32, 
                                    max_value=512, 
                                    step=32),
            activation = hp.Choice("activation", ["relu", "selu"]),
        )
    )
    model.add(layers.Dense(10))
    
    model.compile(
        optimizer="adam", loss="mean_absolute_error", metrics=["mse"],
    
    )
    return model


# In[98]:


build_model(kt.HyperParameters()) # Überprüfung ob der Model build durchgeführt werden konnte


# In[99]:


hp = kt.HyperParameters()

print(hp.Int("units", min_value=32, max_value=512, step=32))


# ## Start Search
# 
# Nun wird der Tuner definiert dabei wird RandomSearch als Tuner-Klasse verwendet.

# In[103]:


tuner = kt.RandomSearch(
    hypermodel=build_model,
    objective="val_mse",
    max_trials=3,
    executions_per_trial=2,
    overwrite=True,
    directory="tmp",
    project_name="example_mlops",
)


# In[104]:


tuner.search_space_summary()


# In[105]:


# Tensorboard Ordner
log_dir = "tmp/tb_logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tuner.search(
    X_train,
    y_train,
    epochs=2,
    validation_data=(X_val, y_val),
    # TensorBoard Callback
    callbacks=[tf.keras.callbacks.TensorBoard(log_dir=log_dir)],
)


# In[106]:


get_ipython().run_line_magic('tensorboard', '--logdir /tmp/tb_logs')


# # SHAP
# 
# Anschließend wird die Bibliothek SHAP verwendet. Hierfür wird zuerst das Modell mit den besten Hyperparametern selektiert.

# In[67]:


best_hps=tuner.get_best_hyperparameters(num_trials=1)[0] 


# In[68]:


model = tuner.hypermodel.build(best_hps) 


# In[69]:


model.fit(X_train, y_train, 
         epochs=5, 
         batch_size=13,
         validation_data=(X_test, y_test)
         )


# Nun kann der Explainer definiert werden. In diesem Beispiel mit 50 Samples.

# In[114]:


explainer = shap.KernelExplainer(model, X_train.iloc[:50,:])


# In[119]:


shap_values = explainer.shap_values(X_train.iloc[50,:], nsamples=500) # 500 samples, um die SHAP-Werte für eine Vorhersage zu bestimmen


# In[120]:


shap.force_plot(explainer.expected_value[0], shap_values[0], X_train.iloc[50,:])


# Nun versuchen wir mithilfe von SHAP mehrere Prognosen zu erklären.

# In[125]:


shap_values = explainer.shap_values(X_train.iloc[50:200,:], nsamples=500)


# In[126]:


shap.force_plot(explainer.expected_value[0], shap_values[0], X_train.iloc[50:200,:])

