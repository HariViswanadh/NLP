#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow 

#importing dataset
data = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)


#cleaning the dataset
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
dataset=[]
for i in range(0, 1000):
    show=data["Review"][i]
    show=re.sub('[^a-zA-Z]', ' ', show)
    show=show.lower()
    show=show.split()
    show=[ps.stem(word) for word in show if not word in set(stopwords.words('english'))]
    show=' '.join(show)
    dataset.append(show)
    

#creating bag of words model 
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)


#spliting the data set(indi or dep var)
X = cv.fit_transform(dataset).toarray()
y = data.iloc[:, 1].values 


# Splitting the dataset into the Training set and Test set  
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
X_train.shape


#creating neural network for nlp data
import keras
from keras.models import Sequential
from keras.layers import Dense
design=Sequential
design.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 1500))
design.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
design.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
design.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
design.fit(X_train, y_train, batch_size = 32, nb_epoch = 25)


#.making predictions and FINDING ACCURACY 
y_pred = design.predict(X_test)
y_pred = (y_pred > 0.5)
    