#!/usr/bin/env python
# coding: utf-8

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


# In[4]:


# print the JS visualization code to the notebook
shap.initjs()


# In[5]:


df = pd.read_csv("/Users/lukas/Desktop/car_price_final.csv")


# In[6]:


# make target variable
y = df.pop('price')


# In[7]:


list_numerical = ['year', 'condition', 'odometer']

X = df[list_numerical]


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[17]:


X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)


# In[9]:


scaler = StandardScaler().fit(X_train[list_numerical]) 

X_train[list_numerical] = scaler.transform(X_train[list_numerical])
X_test[list_numerical] = scaler.transform(X_test[list_numerical])


# In[10]:


def build_model(hp):

    model = tf.keras.Sequential()
    
    model.add(layers.Flatten())
    
    model.add(
        layers.Dense(
            # Define the hyperparameter.
            units = hp.Int("units", min_value=32, 
                                    max_value=512, 
                                    step=32),
            activation = "relu",
        )
    )
    model.add(layers.Dense(10))
    
    model.compile(
        optimizer="adam", loss="mean_squared_error", metrics=["mse"],
    
    )
    return model


# In[11]:


build_model(kt.HyperParameters())


# In[12]:


hp = kt.HyperParameters()

print(hp.Int("units", min_value=32, max_value=512, step=32))


# In[13]:


tuner = kt.RandomSearch(
    hypermodel=build_model,
    objective="val_mse",
    max_trials=3,
    executions_per_trial=2,
    overwrite=True,
    directory="tmp",
    project_name="hello_world",
)


# In[14]:


tuner.search_space_summary()


# In[19]:


# Create TensorBoard folders
log_dir = "tmp/tb_logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tuner.search(
    X_train,
    y_train,
    epochs=2,
    validation_data=(X_val, y_val),
    # Use the TensorBoard callback.
    # The logs will be write to "/tmp/tb_logs".
    callbacks=[tf.keras.callbacks.TensorBoard(log_dir=log_dir)],
)


# In[20]:


get_ipython().run_line_magic('tensorboard', '--logdir /tmp/tb_logs')


# In[8]:


model = tf.keras.Sequential([
    tf.keras.layers.Dense(10,activation="relu"),
    tf.keras.layers.Dense(10,activation="relu"),
    tf.keras.layers.Dense(1)
  ])


# In[9]:


model.compile(optimizer="adam", 
              loss ="mse", 
              metrics=["mean_squared_error"])


# In[11]:


model.fit(X_train, y_train, 
         epochs=5, 
         batch_size=13,
         validation_data=(X_test, y_test)
         )


# In[13]:


# `rankdir='LR'` is to make the graph horizontal.
tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR")


# In[14]:


loss, mse = model.evaluate(X_test, y_test)

print("MSE", mse)


# In[15]:


model.save('sklearn_regression')


# In[22]:


reloaded_model = tf.keras.models.load_model('sklearn_regression')


# In[23]:


predictions = reloaded_model.predict(X_train)


# In[24]:


print(
    "Das Auto hat einen Verkauswert von rund %.1f USD" % (100 * predictions[0][0],)
)


# In[8]:


type(X_train)


# In[26]:


explainer = shap.KernelExplainer(reloaded_model, X_train.iloc[:50,:])


# In[27]:


shap_values = explainer.shap_values(X_train.iloc[20,:], nsamples=500)


# In[28]:


shap.force_plot(explainer.expected_value, shap_values[0], X_train.iloc[20,:])


# In[29]:


shap_values50 = explainer.shap_values(X_train.iloc[50:100,:], nsamples=500)


# In[30]:


shap.force_plot(explainer.expected_value, shap_values50[0], X_train.iloc[50:100,:])

