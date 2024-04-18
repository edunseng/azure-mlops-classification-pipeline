#!/usr/bin/env python
# coding: utf-8

# # Business Problem
# 
# 1. **Objective**
#    - To assess if a customer's license should be issued(AAI), renewed(REV) or cancelled(AAC) depending on various parameters
#    - Learn from various features of applications rejected or given in the past to come up with a decision
# 
# 
# 2. **Machine Learning Problem**
#    - Develop a machine learning model to learn relation of the target variable with the set of features available from the training data
# 
# 
# 3. **Technology**
#    - Python, h2O, Scikit-learn, tensorflow, keras, Pandas, Numpy
#    
# 
# 4. **Decision making**
#    - Select the best model which performs the best w.r.t better accuracy
#    - Metrics: Accuracy
#    
# 
# 5. **Deployment**
#    - Deploy model in a scalable way so that business decisions can be taken in near real time in assessing a customer's loan worthiness
# 
# 
# 
# **Features**<br>
# ID<br>
# LICENSE_ID<br>
# ACCOUNT_NUMBER<br>
# SITE_NUMBER<br>
# LEGAL_NAME<br>
# DOING_BUSINESS_AS_NAME<br>
# ADDRESS<br>
# CITY<br>
# STATE<br>
# ZIP_CODE<br>
# WARD<br>
# PRECINCT<br>
# WARD_PRECINCT<br>
# POLICE_DISTRICT<br>
# LICENSE_CODE<br>
# LICENSE_DESCRIPTION<br>
# LICENSE_NUMBER<br>
# APPLICATION_TYPE<br>
# APPLICATION_CREATED_DATE<br>
# APPLICATION_REQUIREMENTS_COMPLETE<br>
# PAYMENT_DATE<br>
# CONDITIONAL_APPROVAL<br>
# LICENSE_TERM_START_DATE<br>
# LICENSE_TERM_EXPIRATION_DATE<br>
# LICENSE_APPROVED_FOR_ISSUANCE<br>
# DATE_ISSUED<br>
# LICENSE_STATUS_CHANGE_DATE<br>
# SSA<br>
# LATITUDE<br>
# LONGITUDE<br>
# LOCATION<br>
# LICENSE_STATUS
# 

# In[ ]:


get_ipython().system('pip install h2o==3.36.0.2')
get_ipython().system('pip install keras==2.7.0')
get_ipython().system('pip install numpy==1.21.5')
get_ipython().system('pip install pandas==1.3.5')
get_ipython().system('pip install tensorflow==2.7.0')
get_ipython().system('pip install matplotlib==3.4.3')
get_ipython().system('pip install seaborn==0.11.2')


# In[1]:


#importing necessary libraries
import h2o
from h2o.estimators import H2OGradientBoostingEstimator, H2ORandomForestEstimator
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[2]:


#initializing h2o library
h2o.init()


# In[3]:


#function to view all columns
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)


# In[4]:


#reading data
pdf = pd.read_csv("input/License_Data.csv")
print(pdf.shape)


# In[5]:


#value counts for license status
pdf.LICENSE_STATUS.value_counts()


# In[6]:


pdf = pdf[pdf.LICENSE_STATUS.isin(['AAI', 'AAC', 'REV'])]


# In[7]:


#checking for null values
pdf.isna().sum()


# In[8]:


pdf.info()


# In[9]:


#unique values counts
pdf.nunique()


# In[10]:


pdf.head()


# In[11]:


#coountplot for license status
sns.countplot(pdf['LICENSE_STATUS'])
plt.show()


# In[12]:


#value counts 
pdf.CONDITIONAL_APPROVAL.value_counts()


# In[13]:


#putting flag for common features
pdf['LEGAL_BUSINESS_NAME_MATCH'] = pdf.apply(lambda x: 1 if str(x['LEGAL_NAME'].upper()) in str(x['DOING_BUSINESS_AS_NAME']) .upper()
                                             or str(x['DOING_BUSINESS_AS_NAME']).upper() in str(x['LEGAL_NAME']).upper() else 0, 
                                             axis=1)


# In[14]:


pdf['LICENSE_DESCRIPTION'].value_counts()


# In[15]:


#replacing required values
pdf['LICENSE_DESCRIPTION'] = pdf['LICENSE_DESCRIPTION'].replace('Motor Vehicle Repair : Engine Only (Class II)', 'Motor Vehicle Repair')
pdf['LICENSE_DESCRIPTION'] = pdf['LICENSE_DESCRIPTION'].replace('Motor Vehicle Repair: Engine/Body(Class III)', 'Motor Vehicle Repair')
pdf['LICENSE_DESCRIPTION'] = pdf['LICENSE_DESCRIPTION'].replace('Motor Vehicle Repair; Specialty(Class I)', 'Motor Vehicle Repair')
pdf['LICENSE_DESCRIPTION'] = pdf['LICENSE_DESCRIPTION'].replace('Day Care Center Under 2 Years', 'Day Care Center')
pdf['LICENSE_DESCRIPTION'] = pdf['LICENSE_DESCRIPTION'].replace('Day Care Center 2 - 6 Years', 'Day Care Center')
pdf['LICENSE_DESCRIPTION'] = pdf['LICENSE_DESCRIPTION'].replace('Day Care Center Under 2 and 2 - 6 Years', 'Day Care Center')
pdf['LICENSE_DESCRIPTION'] = pdf['LICENSE_DESCRIPTION'].replace('Peddler, non-food', 'Peddler')
pdf['LICENSE_DESCRIPTION'] = pdf['LICENSE_DESCRIPTION'].replace('Peddler, non-food, special', 'Peddler')
pdf['LICENSE_DESCRIPTION'] = pdf['LICENSE_DESCRIPTION'].replace('Peddler, food (fruits and vegtables only)', 'Peddler')
pdf['LICENSE_DESCRIPTION'] = pdf['LICENSE_DESCRIPTION'].replace('Peddler,food - (fruits and vegetables only) - special', 'Peddler')
pdf['LICENSE_DESCRIPTION'] = pdf['LICENSE_DESCRIPTION'].replace('Tire Facilty Class I (100 - 1,000 Tires)', 'Tire Facilty')
pdf['LICENSE_DESCRIPTION'] = pdf['LICENSE_DESCRIPTION'].replace('Tire Facility Class II (1,001 - 5,000 Tires)', 'Tire Facilty')
pdf['LICENSE_DESCRIPTION'] = pdf['LICENSE_DESCRIPTION'].replace('Tire Facility Class III (5,001 - More Tires)', 'Tire Facilty')
pdf['LICENSE_DESCRIPTION'] = pdf['LICENSE_DESCRIPTION'].replace('Repossessor Class A', 'Repossessor')
pdf['LICENSE_DESCRIPTION'] = pdf['LICENSE_DESCRIPTION'].replace('Repossessor Class B', 'Repossessor')
pdf['LICENSE_DESCRIPTION'] = pdf['LICENSE_DESCRIPTION'].replace('Repossessor Class B Employee', 'Repossessor')
pdf['LICENSE_DESCRIPTION'] = pdf['LICENSE_DESCRIPTION'].replace('Expediter - Class B', 'Expediter')
pdf['LICENSE_DESCRIPTION'] = pdf['LICENSE_DESCRIPTION'].replace('Expediter - Class A', 'Expediter')
pdf['LICENSE_DESCRIPTION'] = pdf['LICENSE_DESCRIPTION'].replace('Expediter - Class B Employee', 'Expediter')
pdf['LICENSE_DESCRIPTION'] = pdf['LICENSE_DESCRIPTION'].replace('Itinerant Merchant, Class II', 'Itinerant Merchant')
pdf['LICENSE_DESCRIPTION'] = pdf['LICENSE_DESCRIPTION'].replace('Itinerant Merchant, Class I', 'Itinerant Merchant')


# In[16]:


#unique counts
pdf['LICENSE_DESCRIPTION'].nunique()


# In[17]:


pdf['LEGAL_NAME'] = pdf['LEGAL_NAME'].str.replace('.', '', regex=False)
pdf['DOING_BUSINESS_AS_NAME'] = pdf['DOING_BUSINESS_AS_NAME'].str.replace('.', '', regex=False)

pdf['BUSINESS_TYPE'] = 'PVT'

pdf['BUSINESS_TYPE'] = np.where(pdf['LEGAL_NAME'].str.contains('INC'), 'INC', pdf['BUSINESS_TYPE'])
pdf['BUSINESS_TYPE'] = np.where(pdf['LEGAL_NAME'].str.contains('INCORPORATED'), 'INC', pdf['BUSINESS_TYPE'])
pdf['BUSINESS_TYPE'] = np.where(pdf['DOING_BUSINESS_AS_NAME'].str.contains('INC'), 'INC', pdf['BUSINESS_TYPE'])
pdf['BUSINESS_TYPE'] = np.where(pdf['DOING_BUSINESS_AS_NAME'].str.contains('INCORPORATED'), 'INC', pdf['BUSINESS_TYPE'])
pdf['BUSINESS_TYPE'] = np.where(pdf['LEGAL_NAME'].str.contains('LLC'), 'LLC', pdf['BUSINESS_TYPE'])
pdf['BUSINESS_TYPE'] = np.where(pdf['DOING_BUSINESS_AS_NAME'].str.contains('LLC'), 'LLC', pdf['BUSINESS_TYPE'])
pdf['BUSINESS_TYPE'] = np.where(pdf['LEGAL_NAME'].str.contains('CO'), 'CORP', pdf['BUSINESS_TYPE'])
pdf['BUSINESS_TYPE'] = np.where(pdf['LEGAL_NAME'].str.contains('CORP'), 'CORP', pdf['BUSINESS_TYPE'])
pdf['BUSINESS_TYPE'] = np.where(pdf['LEGAL_NAME'].str.contains('CORPORATION'), 'CORP', pdf['BUSINESS_TYPE'])
pdf['BUSINESS_TYPE'] = np.where(pdf['DOING_BUSINESS_AS_NAME'].str.contains('CO'), 'CORP', pdf['BUSINESS_TYPE'])
pdf['BUSINESS_TYPE'] = np.where(pdf['DOING_BUSINESS_AS_NAME'].str.contains('CORP'), 'CORP', pdf['BUSINESS_TYPE'])
pdf['BUSINESS_TYPE'] = np.where(pdf['DOING_BUSINESS_AS_NAME'].str.contains('CORPORATION'), 'CORP', pdf['BUSINESS_TYPE'])
pdf['BUSINESS_TYPE'] = np.where(pdf['LEGAL_NAME'].str.contains('LTD'), 'LTD', pdf['BUSINESS_TYPE'])
pdf['BUSINESS_TYPE'] = np.where(pdf['LEGAL_NAME'].str.contains('LIMITED'), 'LTD', pdf['BUSINESS_TYPE'])
pdf['BUSINESS_TYPE'] = np.where(pdf['DOING_BUSINESS_AS_NAME'].str.contains('LTD'), 'LTD', pdf['BUSINESS_TYPE'])
pdf['BUSINESS_TYPE'] = np.where(pdf['DOING_BUSINESS_AS_NAME'].str.contains('LIMITED'), 'LTD', pdf['BUSINESS_TYPE'])


# In[18]:


#value counts for business type
pdf['BUSINESS_TYPE'].value_counts()


# In[19]:


#countplot for business type
sns.countplot(pdf['BUSINESS_TYPE'])
plt.show()


# In[20]:


pdf.ZIP_CODE.value_counts()


# In[21]:


#filling null values and seeting flag
pdf['ZIP_CODE'].fillna(-1, inplace=True)
pdf['ZIP_CODE_MISSING'] = pdf.apply(lambda x: 1 if x['ZIP_CODE'] == -1 else 0, axis=1)


# In[22]:


#histogram plot
pdf[['SSA']].plot.hist(bins=12, alpha=0.8)


# In[23]:


#filling null values
pdf['SSA'].fillna(-1, inplace=True)


# In[24]:


#filling null values
pdf['APPLICATION_REQUIREMENTS_COMPLETE'].fillna(-1, inplace=True)
pdf['APPLICATION_REQUIREMENTS_COMPLETE'] = pdf.apply(lambda x: 0 if x['APPLICATION_REQUIREMENTS_COMPLETE'] == -1 
                                                     else 1, axis=1)


# # Train Test Split

# In[25]:


#splitting into train and test
train, test = train_test_split(pdf, test_size=0.2, random_state=123)


# In[26]:


#putting test and train file in h2o frame
train = h2o.H2OFrame(train)
test = h2o.H2OFrame(test)


# In[27]:


#trainign model
h2o_rf = H2ORandomForestEstimator(ntrees=100, seed=123, max_depth=10)
h2o_rf.train(x=['APPLICATION_TYPE', 'CONDITIONAL_APPROVAL', 'LICENSE_CODE', 'SSA', 'LEGAL_BUSINESS_NAME_MATCH', 
                'ZIP_CODE_MISSING', 'SSA', 'APPLICATION_REQUIREMENTS_COMPLETE', 'LICENSE_DESCRIPTION', 'BUSINESS_TYPE'], 
             y='LICENSE_STATUS', training_frame=train)


# In[28]:


#predicting on test data
predictions = h2o_rf.predict(test)
predictions['actual'] = test['LICENSE_STATUS']
predictions = predictions.as_data_frame()


# In[29]:


predictions.head()


# In[30]:


accuracy = (predictions[predictions.actual == predictions.predict].shape[0])* 100.0 / predictions.shape[0]


# In[31]:


accuracy


# # Data Conversion for DNN model

# In[38]:


predictors = ['APPLICATION_TYPE', 'CONDITIONAL_APPROVAL', 'LICENSE_CODE', 'SSA', 'LEGAL_BUSINESS_NAME_MATCH', 
                'ZIP_CODE_MISSING', 'SSA', 'APPLICATION_REQUIREMENTS_COMPLETE', 'LICENSE_DESCRIPTION', 'BUSINESS_TYPE']

target = ["LICENSE_STATUS_AAC", "LICENSE_STATUS_AAI", "LICENSE_STATUS_REV"]


# In[39]:


pdf[predictors].info()


# In[40]:


final_df = pdf[predictors + ["LICENSE_STATUS"]]
#onehotencoding for categorical columns
final_df = pd.get_dummies(final_df, columns=['APPLICATION_TYPE', 'CONDITIONAL_APPROVAL', 'LICENSE_CODE', 'LICENSE_DESCRIPTION', 'BUSINESS_TYPE', 'LICENSE_STATUS'])


# In[41]:


final_df.columns


# In[42]:


#splitting into train and test
train, test = train_test_split(final_df, test_size=0.2, random_state=123)

X_train = train.drop(target, axis=1).values
y_train = train[target].values

X_test = test.drop(target, axis=1).values
y_test = test[target].values

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


# In[43]:


train.head()


# # Modeling

# In[45]:


from tensorflow import keras
from tensorflow.keras import layers


# In[67]:


#building sequential model in tensorflow
model = keras.Sequential(
    [
        layers.InputLayer(input_shape=(X_train.shape[1])),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(64, activation="tanh"),
        layers.Dropout(0.2),
        layers.Dense(64, activation="tanh"),
        layers.Dropout(0.2),
        layers.Dense(3, activation="softmax"),
    ]
)
optimizer = keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
print(model.summary())


# In[68]:


#model training
model.fit(X_train, y_train, batch_size=64, epochs=20)


# In[69]:


#model evaluation 
model.evaluate(X_test, y_test)


# In[70]:


#prediction on test data
model.predict(X_test)


# In[ ]:




