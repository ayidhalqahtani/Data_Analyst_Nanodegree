
# coding: utf-8

# In[73]:

#!/usr/bin/python2.7

import sys
import pickle
import random
import matplotlib.pyplot as plt
from time import time
import numpy as np
from numpy import mean
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn import cross_validation
from sklearn.metrics import *
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


sys.path.append("../tools/")




# In[74]:


from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary','deferral_payments', 'total_payments', 'loan_advances',
                 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value',
                  'exercised_stock_options', 'long_term_incentive',
                 'restricted_stock', 'director_fees', 'to_messages', 'from_poi_to_this_person',
                'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] # You will need to use more features


### we have created this list for future use to add the new features to this  list and train the classifier  and compare the result to original features.
features_list1 = ['poi','salary','deferral_payments', 'total_payments', 'loan_advances',
                 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value',
                  'exercised_stock_options', 'long_term_incentive',
                 'restricted_stock', 'director_fees', 'to_messages', 'from_poi_to_this_person',
                'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

## I came back and take off 'email_address' feature because it casuses problem when I want to extract feature and lables


# In[75]:

## we will print the number of features that we will use :
print (len(features_list))



# In[76]:


### Load the dictionary containing the dataset and put it in data dictionary as below 
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)



# In[77]:

### Store to my_dataset for easy export below.

my_dataset = data_dict



# In[78]:

Total_people=len(my_dataset)


# In[79]:

## to print how many data points we have in our dataset.
print 'the total number of people in the dataset',Total_people


number_poi= 0 
for x in my_dataset:
   if my_dataset[x]['poi']==True:
        number_poi=number_poi+1
        
print 'the number of persons who is not person of interest' ,Total_people -number_poi


# In[80]:


### Task 2: Remove outliers
### We want to explore the missing values in all feature list by list them as below :
Null_value = [0 for i in range(len(features_list))]
for i, person in my_dataset.iteritems():
   for j, feature in enumerate(features_list):
       if person[feature] == 'NaN':
           Null_value[j] += 1
for i, feature in enumerate(features_list):
   print feature, Null_value[i]
 


# In[81]:

## there are many null values as can be seen above , but our task is to remove the outliers from the features 
## like salary:
## first ,we have to see the outliers before removing them especially we want exclude the people with high salary like CEO  
# AND Before that we want to see in there null value in our feature list:


features =['salary', 'bonus']
outlier_data = featureFormat(my_dataset, features)
for point in outlier_data:
    salary=point[0]
    bonus=point[1]
    plt.scatter(salary, bonus)
plt.xlabel('salary for Eron dataset')
plt.ylabel('bonus for Eron dataset')

plt.show()



# In[82]:

## as can be seen above in the graph there is outliers and we are goin to PRINT IT and remove it  in the second :

for i, x in data_dict.items():
    if x['salary'] != 'NaN' and x['salary'] > 10000000:
        print i
 


# In[83]:

my_dataset.pop('TOTAL', 0)
print "Number of datapoint after removing outliers is :", len(my_dataset)


# In[84]:



## NO we are going to show the dataset after removing the outliers :

features =['salary', 'bonus']
x = featureFormat(my_dataset, features)
for point in x:
    salary=point[0]
    bonus=point[1]
    plt.scatter(salary, bonus)
plt.xlabel('salary for Eron dataset')
plt.ylabel('bonus for Eron dataset')

plt.show()



# In[85]:

### Task 3: Create new feature(s)
## we are going to create feature to calculate the ratio of from and to message for poi
for key, v in my_dataset.iteritems():
    v['from_poi_ratio']=0
    if v['to_messages'] and v['from_poi_to_this_person']!='NaN':
        v['from_poi_to_this_person']=float(v['from_poi_to_this_person'])
        v['to_messaages']=float(v['to_messages'])
        if v['from_poi_to_this_person'] > 0:
            v['from_poi_ratio']=v['from_poi_to_this_person']/v['to_messages']


# In[86]:

for key, v in my_dataset.iteritems():
    v['to_poi_ratio']=0
    if v['from_messages'] and v['from_this_person_to_poi']!='NaN':
        v['from_this_person_to_poi']=float(v['from_this_person_to_poi'])
        v['from_messaages']=float(v['from_messages'])
        if v['from_this_person_to_poi'] > 0:
            v['to_poi_ratio']=v['from_this_person_to_poi']/v['from_messages']


# In[87]:

#We have to add the new created features to our feature list 
#
features_list1.extend(['from_poi_ratio', 'to_poi_ratio'])


# In[88]:

print "we have create two features :'from_poi_ratio','to_poi_ratio'" 

print len(features_list1)
print features_list1



#we have create two features :'from_poi_ratio','to_poi_ratio'
# the number of features list with new features is :20
# the new features list ['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'exercised_stock_options', 'long_term_incentive', 'restricted_stock', 'director_fees', 'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi', 'from_poi_ratio', 'to_poi_ratio']


# In[89]:

### Extract features and labels from  the original subset of features  for local testing
#data = featureFormat(my_dataset, features_list, sort_keys = True)
#labels, features = targetFeatureSplit(data)

### Extract features and labels from the dataset with the newly generate features  for  testing(features_list1)
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


# In[90]:


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()





# In[91]:

selection=SelectKBest(k=8)
selection.fit_transform(features, labels)
score = selection.scores_


# In[92]:

# we have used this sorting to help us to see the features scores from importance perspective , so we can select the best features 
# https://docs.python.org/3/howto/sorting.html
#https://stackoverflow.com/questions/44451467/seaborn-barplot-ordering-by-bar-length

features_selected =list(reversed(sorted(zip(features_list[1:],score), key=lambda x:x[1])))


# In[72]:

print features_selected  


# In[93]:

import pandas as pd
# I had to replace Nan with 0 because the algorthi did not run and to get accurate results.
my_dataset = pd.DataFrame(my_dataset)
my_dataset.replace('NaN', 0, inplace=True)
my_dataset.replace(np.nan, 0, inplace=True)


# In[94]:

my_dataset.tail()


# In[95]:

# Example starting point. Try investigating other evaluation techniques!
#from sklearn.cross_validation import train_test_split
 
    
## we select the features manaully here.
#features_list = ['poi','to_poi_ratio', 'total_stock_value','from_poi_to_this_person',
#                'from_poi_ratio','to_messages','bonus']

features_list=['poi','exercised_stock_options','total_stock_value','bonus','salary'
,'to_poi_ratio','deferred_income','long_term_incentive']
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)



# In[96]:


features_train, features_test, labels_train, labels_test =train_test_split(features, labels, test_size=0.3, random_state=42)
#classifier for the GaussianNB
#clf.fit(features_train, labels_train)
#prediction = clf.predict(features_test)
#accuracy = accuracy_score(prediction,labels_test)
#print "accuracy: ", accuracy
#print 'Precision = ', precision_score(labels_test,prediction)
#print 'Recall = ', recall_score(labels_test,prediction)



# In[97]:

# Random Forest classifier
clf= RandomForestClassifier(class_weight='balanced', min_samples_split=12, n_estimators=20, random_state=6)


clf.fit(features_train, labels_train)
prediction = clf.predict(features_test)
accuracy = accuracy_score(prediction,labels_test)
print "accuracy: ", accuracy
print 'Precision = ', precision_score(labels_test,prediction)
print 'Recall = ', recall_score(labels_test,prediction)



# In[ ]:




# In[98]:

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



