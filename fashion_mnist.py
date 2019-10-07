#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd


# In[2]:


data=keras.datasets.fashion_mnist


# In[3]:


(train_img,train_label),(test_img,test_label)=data.load_data()


# In[4]:


class_names=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']


# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


plt.imshow(train_img[0])


# In[7]:


train_img=train_img/255.0
test_img=test_img/255.0


# In[11]:


model=keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation="relu"),
    keras.layers.Dense(10,activation="softmax")    
])

model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])

model.fit(train_img, train_label, epochs=5)


# In[12]:


test_loss, test_acc=model.evaluate(test_img,test_label)


# In[13]:


print("tested acc:",test_acc)


# In[15]:


prediction=model.predict(test_img)


# In[18]:


print(class_names[np.argmax(prediction[0])])


# In[23]:


for i in range(5):
    plt.grid(False)
    plt.imshow(test_img[i],cmap=plt.cm.binary)
    plt.xlabel("Actual:"+class_names[test_label[i]])
    plt.title("Prediction"+class_names[np.argmax(prediction[i])])


# In[1]:





# In[ ]:




