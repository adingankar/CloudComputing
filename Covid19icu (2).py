#!/usr/bin/env python
# coding: utf-8

# ***Final Project - Cloud Computing - Group 9***
# ## **COVID-19 ICU BED PREDICTION ANALYSIS**

# ### **1. Preparing Amazon Sagemaker Instance**

# In[1]:


#pip install tensorflow


# In[2]:


#pip install --upgrade sagemaker


# In[3]:


#pip install xgboost


# In[4]:


# import libraries
import boto3, re, sys, math, json, os, sagemaker, urllib.request
from sagemaker import get_execution_role
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image
from IPython.display import display
from time import gmtime, strftime
from sagemaker.predictor import csv_serializer


# In[5]:


# Define IAM role
role = get_execution_role()
prefix = 'sagemaker'
my_region = boto3.session.Session().region_name # set the region of the instance


# In[6]:


# this line automatically looks for the XGBoost image URI and builds an XGBoost container.
xgboost_container = sagemaker.image_uris.retrieve("xgboost", my_region, "latest")

print("Success - the MySageMakerInstance is in the " + my_region + " region. You will use the " + xgboost_container + " container for your SageMaker endpoint.")


# In[7]:


bucket_name = 'covid19icu' # <--- CHANGE THIS VARIABLE TO A UNIQUE NAME FOR YOUR BUCKET
s3 = boto3.resource('s3')
try:
    if  my_region == 'us-east-1':
      s3.create_bucket(Bucket=bucket_name)
    else: 
      s3.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={ 'LocationConstraint': my_region })
    print('S3 bucket created successfully')
except Exception as e:
    print('S3 error: ',e)


# ### **2. Exploratory Data Analysis**

# In[8]:


import pandas as pd
bucket='covid19icu'
file_key = 'https://covid19icu.s3.amazonaws.com/Kaggle_Sirio_Libanes_ICU_Prediction.csv'
s3uri = 's3://covid19icu/Kaggle_Sirio_Libanes_ICU_Prediction.csv'.format(bucket, file_key)
df = pd.read_csv(s3uri)
df.head()


# In[9]:


import tensorflow as tf
from tensorflow import keras


# In[10]:


# The random seed
random_seed = 42

# Set random seed in tensorflow
tf.random.set_seed(random_seed)

# Set random seed in numpy
import numpy as np
np.random.seed(random_seed)


# In[11]:


#shape of the data
df.shape


# In[12]:


#Basic information about the data
df.info()


# In[13]:


#rate of patients age 65 above and below
admit = df[df['WINDOW'] =='ABOVE_12']
colors = ['#ff9999','#99ff99']
age_65 = admit.groupby('AGE_ABOVE65')['PATIENT_VISIT_IDENTIFIER'].count().reset_index()
labels =["Under65", "Above65"]
plt.pie(age_65['PATIENT_VISIT_IDENTIFIER'],explode=(0,0.1),colors=colors,labels = labels,autopct='%1.1f%%',shadow=True, startangle=90)
plt.axis('equal')
plt.title('Patients above 65 and below')
plt.show()


# In[14]:


# rate of patients admitted or not admitted to ICU 
admit=df[df['WINDOW'] =='ABOVE_12']
colors = ['#ff9999','#99ff99']
ICUratio=admit.groupby('ICU')['PATIENT_VISIT_IDENTIFIER'].count().reset_index()
labels=["not-admitted", "admitted"]
plt.pie(ICUratio['PATIENT_VISIT_IDENTIFIER'],explode=(0,0.1),colors=colors,labels = labels,autopct='%1.1f%%',shadow=True, startangle=90)
plt.axis('equal')
plt.title('ICU admissions')
plt.show()


# In[15]:


#Patients above and below 65 admitted to ICU ratio
admitted=admit[admit['ICU']==1]
colors = ['#52ffff','#ffcc99']
ICUage=admitted.groupby('AGE_ABOVE65')['PATIENT_VISIT_IDENTIFIER'].count().reset_index()
labels=["Under 65", "Above 65"]
plt.pie(ICUage['PATIENT_VISIT_IDENTIFIER'],explode=(0,0.1),colors=colors,labels = labels,autopct='%1.1f%%',shadow=True, startangle=90)
plt.axis('equal')
plt.title('Admitted on ICU age above or below 65')
plt.show()


# In[16]:


#Patients admitted to ICU based on age percentile
fig, ax = plt.subplots(figsize =(9, 9))
agepercent=admit[admit['ICU']==1]
agepercentile=agepercent.groupby('AGE_PERCENTIL')['PATIENT_VISIT_IDENTIFIER'].count().reset_index()
ax.barh(agepercentile["AGE_PERCENTIL"],agepercentile["PATIENT_VISIT_IDENTIFIER"],color=['b', 'g', 'r', 'c', 'm', 'y', 'g'])
for i in ax.patches:
    plt.text(i.get_width()+0.2, i.get_y()+0.5,
             str(round((i.get_width()), 2)),
             fontsize = 10, fontweight ='bold',
             color ='grey')
ax.set_ylabel("Age percentiles")
ax.set_xlabel("Number of patients")
ax.set_title('Patients admitted on ICU based on age group')
plt.show()


# In[17]:


#Patients based on gender admitted to ICU ratio
admitted=admit[admit['ICU']==1]
colors = ['#52ffff','#ffcc99']
ICUgender=admitted.groupby('GENDER')['PATIENT_VISIT_IDENTIFIER'].count().reset_index()
labels=["0", "1"]
plt.pie(ICUgender['PATIENT_VISIT_IDENTIFIER'],explode=(0,0.1),colors=colors,labels = labels,autopct='%1.1f%%',shadow=True, startangle=90)
plt.axis('equal')
plt.title('Admitted on ICU based on gender')
plt.show()


# In[18]:


#Patients admitted to ICU based on different window periods
fig, ax = plt.subplots(figsize =(9, 9))
windowicu=df[df['ICU']==1]
windowevent=windowicu.groupby('WINDOW')['PATIENT_VISIT_IDENTIFIER'].count().reset_index()
ax.barh(windowevent["WINDOW"],windowevent["PATIENT_VISIT_IDENTIFIER"],color=['b', 'g', 'r', 'c', 'm'])
for i in ax.patches:
    plt.text(i.get_width()+0.2, i.get_y()+0.5,
             str(round((i.get_width()), 2)),
             fontsize = 10, fontweight ='bold',
             color ='grey')
ax.set_ylabel("Window periods")
ax.set_xlabel("Number of patients")
ax.set_title('Patients admitted on ICU at different windows')
plt.show()


# In[19]:


df.fillna(method='ffill',inplace=True)
df.fillna(method='bfill',inplace=True)


# In[20]:


#Filter mean and median columns
datamean = df.filter(regex="\w+_MEAN")
datamedian = df.filter(regex="\w+_MEDIAN")
columns_mm= list(map(lambda col: col.split('_MEAN')[0], datamean.columns))


# In[21]:


#Plot mean and median to find whether they have significant difference
fig, ax= plt.subplots(int(len(columns_mm) / 3), 3, figsize=(19,19))
for i, col in enumerate(columns_mm):
    ax[i // 3, i % 3].scatter(x=datamean[f"{col}_MEAN"], y=datamedian[f"{col}_MEDIAN"])
    ax[i // 3, i % 3].set_title(col)
fig.subplots_adjust(hspace=1)
plt.show()


# In[22]:


uniquecols = ['PATIENT_VISIT_IDENTIFIER', 'AGE_ABOVE65', 'AGE_PERCENTIL', 'GENDER',
       'DISEASE GROUPING 1', 'DISEASE GROUPING 2', 'DISEASE GROUPING 3',
       'DISEASE GROUPING 4', 'DISEASE GROUPING 5', 'DISEASE GROUPING 6', 'HTN',
       'IMMUNOCOMPROMISED','WINDOW', 'ICU']
data= df[uniquecols + list(datamean.columns)]
data.head()


# In[23]:


# Display the rows and columns in data
pd.DataFrame([[data.shape[0], data.shape[1]]], columns=['# rows', '# columns'])


# In[24]:


#create a new column to check whether patient have been eventually admitted to ICU
dataICU=(data.groupby("PATIENT_VISIT_IDENTIFIER")["ICU"].sum()>0).reset_index()*1
dataICU.columns = ["PATIENT_VISIT_IDENTIFIER", "ICU_SUM"]


# In[25]:


#join the datasets based on patient ID's
dataadmit = pd.merge(data, dataICU, on = "PATIENT_VISIT_IDENTIFIER")


# In[26]:


#Display the data
dataadmit.head()


# In[27]:


#Look at the value of target
dataadmit.ICU.value_counts()


# In[28]:


#since we have the data whether the patient has been eventualy admitted into ICU or not, let's remove ICU=1 as insisted by the author
data_noicu=dataadmit[dataadmit['ICU']==0].reset_index(drop= True)


# In[29]:


data_noicu = data_noicu[data_noicu.WINDOW == "0-2"].reset_index(drop = True)


# In[30]:


#display the data
data_noicu.head()


# In[31]:


# Display the rows and columns in data
pd.DataFrame([[data_noicu.shape[0], data_noicu.shape[1]]], columns=['# rows', '# columns'])


# In[32]:


datafinal=data_noicu.drop(["PATIENT_VISIT_IDENTIFIER", "WINDOW", "ICU"],axis = 1)


# In[33]:


#Display the data
datafinal.head()


# In[34]:


#look for categorical columns and convert them
catcols= datafinal.select_dtypes(object).columns 
#print()
datafinal = pd.get_dummies(datafinal, columns = catcols)


# In[35]:


#display the data
datafinal.head()


# In[36]:


#check target variable count
datafinal.ICU_SUM.value_counts()


# In[37]:


data_cor=datafinal.corrwith(datafinal['ICU_SUM'])
print(data_cor)


# In[38]:


data_cor.describe()


# In[39]:


# Set target
target = 'ICU_SUM'


# In[40]:


from sklearn.model_selection import train_test_split

# Divide the training data into training (80%) and testing (20%)
df_train, df_test = train_test_split(datafinal, train_size=0.8, random_state=42, stratify=datafinal[target])

# Reset the index
df_train, df_test = df_train.reset_index(drop=True), df_test.reset_index(drop=True)


# In[41]:


# Check the shape of the training set
pd.DataFrame([[df_train.shape[0], df_train.shape[1]]], columns=['# rows', '# columns'])


# In[42]:


# Check the shape of the testing set
pd.DataFrame([[df_test.shape[0], df_test.shape[1]]], columns=['# rows', '# columns'])


# In[43]:


# training and validation
df_train, df_valid = train_test_split(df_train, train_size=0.8, random_state=42, stratify=df_train[target])


# In[44]:


# Reset the index
df_train, df_valid = df_train.reset_index(drop=True), df_valid.reset_index(drop=True)


# In[45]:


# Check the shape of the training set
pd.DataFrame([[df_train.shape[0], df_train.shape[1]]], columns=['# rows', '# columns'])


# In[46]:


# Check the shape of the validation set
pd.DataFrame([[df_valid.shape[0], df_valid.shape[1]]], columns=['# rows', '# columns'])


# In[47]:


# Display the first 5 instances of training set
df_train.head()


# In[48]:


# Display the first 5 instances of testing set
df_test.head()


# In[49]:


# Display the first 5 instances of validation set
df_valid.head()


# In[50]:


features = np.setdiff1d(datafinal.columns, [target])


# In[51]:


# Get the feature matrix
X_train = df_train[features]
X_valid = df_valid[features]
X_test = df_test[features]


# In[52]:


# Get the target vector
y_train = df_train[target].apply(lambda x : int(x))
y_valid = df_valid[target].apply(lambda x : int(x))
y_test = df_test[target].apply(lambda x : int(x))
y_train = pd.DataFrame(y_train)
y_valid = pd.DataFrame(y_valid)
y_test = pd.DataFrame(y_test)


# ### **3. Train the ML model**

# In[53]:


session = sagemaker.Session()

pd.concat([y_valid , X_valid], axis=1,sort=True).to_csv('validation.csv', index=False, header=False)
boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix, 'val/validation.csv')).upload_file('validation.csv')
s3_input_validation = sagemaker.inputs.TrainingInput(s3_data='s3://{}/{}/val'.format(bucket_name, prefix), content_type='csv')
pd.concat([y_train, X_train], axis=1,sort=True).to_csv('train.csv', header=False, index=False)
boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix, 'train/train.csv')).upload_file('train.csv')
s3_input_train = sagemaker.inputs.TrainingInput(s3_data='s3://{}/{}/train'.format(bucket_name, prefix), content_type='csv')


pd.DataFrame(X_test).to_csv('test.csv', header=False, index=False)
boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix, 'test/test.csv')).upload_file('test.csv')
s3_input_test = sagemaker.inputs.TrainingInput(s3_data='s3://{}/{}/test'.format(bucket_name, prefix), content_type='csv')


# In[54]:


role = get_execution_role()


from sagemaker import image_uris

xgb = sagemaker.estimator.Estimator(xgboost_container, 
                                    role,   
                                    instance_count=1, 
                                    instance_type='ml.m4.xlarge', 
                                    output_path='s3://{}/{}/output'.format(bucket_name, prefix),
                                    sagemaker_session=session)



xgb.set_hyperparameters(max_depth=5,
                        eta=0.2,
                        gamma=4,
                        min_child_weight=6,
                        subsample=0.8,
                        objective='binary:logistic',
                        num_round=100)


# In[55]:



xgb.fit({'train': s3_input_train, 'validation': s3_input_validation})


# ### **4. Deploy the model**

# In[56]:


xgb_predictor = xgb.deploy(initial_instance_count=1,instance_type='ml.m4.xlarge')


# In[57]:


from sagemaker.serializers import CSVSerializer

test_data_array = X_test.values #load the data into an array
xgb_predictor.serializer = CSVSerializer() # set the serializer type
predictions = xgb_predictor.predict(test_data_array).decode('utf-8') # predict!
predictions_array = np.fromstring(predictions[1:], sep=',') # and turn the prediction into an array
print(predictions_array.shape)


# ### **5. Evaluate model performance**

# In[58]:


cm = pd.crosstab(index=y_test.iloc[:, 0], columns=np.round(predictions_array), rownames=['Observed'], colnames=['Predicted'])
tn = cm.iloc[0,0]; fn = cm.iloc[1,0]; tp = cm.iloc[1,1]; fp = cm.iloc[0,1]; p = (tp+tn)/(tp+tn+fp+fn)*100
print("\n{0:<20}{1:<4.1f}%\n".format("Overall Classification Rate: ", p))
print("{0:<15}{1:<15}{2:>8}".format("Predicted", "Not Admitted", "Admitted"))
print("Observed")
print("{0:<15}{1:<2.0f}% ({2:<}){3:>6.0f}% ({4:<})".format("Not Admitted", tn/(tn+fn)*100,tn, fp/(tp+fp)*100, fp))
print("{0:<16}{1:<1.0f}% ({2:<}){3:>7.0f}% ({4:<}) \n".format("Admitted", fn/(tn+fn)*100,fn, tp/(tp+fp)*100, tp))


# ### **6. Clean up**

# In[59]:


xgb_predictor.delete_endpoint(delete_endpoint_config=True)


# In[60]:


bucket_to_delete = boto3.resource('s3').Bucket(bucket_name)
bucket_to_delete.objects.all().delete()

