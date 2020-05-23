#!/usr/bin/env python
# coding: utf-8

# In[433]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import sklearn.metrics as metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.linear_model import RandomizedLogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
import pydotplus 
from IPython.display import Image
import os


# os.environ["PATH"] += os.pathsep + 'D:/Anaconda/Graphviz/bin/'
df=pd.read_csv("D:/Class Files/5243/mushrooms.csv",header=0)
# df.columns=['age','sex','chest_pain_type','resting_blood_pressure','serum_cholestoral','fasting_blood_sugar','resting_electrocardiographic_results','maximum_heart_rate_achieved','exercise_induced_angina','ST_depression','the_slope_of_the_peak_exercise_ST_segment','number_of_major_vessels','thal','target']
df.head(10)


# In[434]:



print('Number of instances = %d' %(df.shape[0]))
print('Number of attributes = %d\n' %(df.shape[1]))


print(df.dtypes)
df['class'].value_counts()


# In[435]:


df = df.replace('?',np.NaN)

print('Number of instances = %d' % (df.shape[0]))
print('Number of attributes = %d' % (df.shape[1]))
print('Number of missing values:')
for col in df.columns:
    print('\t%s: %d' % (col,df[col].isna().sum()))


# In[436]:



##### Class poisonous=1 #####


# In[437]:


df.shape
spore_print_color=df['spore-print-color'].value_counts()
print(spore_print_color)


# In[438]:


#

m_height = spore_print_color.values.tolist() #Provides numerical values
spore_print_color.axes #Provides row labels
spore_print_color_labels = spore_print_color.axes[0].tolist()

ind = np.arange(9)  # the x locations for the groups
width = 0.8        # the width of the bars
colors = ['#f8f8ff','brown','black','chocolate','red','yellow','orange','blue','purple']
fig, ax = plt.subplots(figsize=(10,5))
mushroom_bars = ax.bar(ind, m_height , width, color=colors)
ax.set_xlabel("spore print color",fontsize=20)
ax.set_ylabel('Quantity',fontsize=20)
ax.set_title('Mushrooms spore print color',fontsize=22)
ax.set_xticks(ind) #Positioning on the x axis
ax.set_xticklabels(('white','brown','black','chocolate','red','yellow','orange','blue','purple')),

for bars in mushroom_bars:
        height = bars.get_height()
        ax.text(bars.get_x() + bars.get_width()/2., 1*height,'%d' % int(height),
                ha='center', va='bottom',fontsize=10)

plt.show()


# In[439]:


poisonous_cc = [] 
edible_cc = []    
for spore_print_color in spore_print_color_labels:
    size = len(df[df['spore-print-color'] == spore_print_color].index)
    edibles = len(df[(df['spore-print-color'] == spore_print_color) & (df['class'] == 'e')].index)
    edible_cc.append(edibles)
    poisonous_cc.append(size-edibles)

    
width=0.4
fig, ax = plt.subplots(figsize=(12,7))

edible_bars = ax.bar(ind, edible_cc , width, color='g')
poison_bars = ax.bar(ind+width, poisonous_cc , width, color='r')
ax.set_xticks(ind + width / 2) #Positioning on the x axis
ax.set_xticklabels(('white','brown','black','chocolate','red','yellow','orange','blue','purple'))

ax.set_xlabel("spore print color",fontsize=20)
ax.set_ylabel('Quantity',fontsize=20)
ax.set_title('Mushrooms spore print color',fontsize=22)
ax.legend((edible_bars,poison_bars),('edible','poisonous'),fontsize=17)

for bars in edible_bars:
        height = bars.get_height()
        ax.text(bars.get_x() + bars.get_width()/2., 1*height,'%d' % int(height),
                ha='center', va='bottom',fontsize=10)
        
for bars in poison_bars:
        height = bars.get_height()
        ax.text(bars.get_x() + bars.get_width()/2., 1*height,'%d' % int(height),
                ha='center', va='bottom',fontsize=10)
plt.show()


# In[440]:


cap_shape = df['cap-shape'].value_counts()
cap_shapes_size = cap_shape.values.tolist() 
cap_shapes_types = cap_shape.axes[0].tolist() 
print(cap_shape)
# Data to plot
cap_shape_labels = ('convex','flat','knobbed','bell', 'sunken','conical')
colors = ['r','y','b','brown','g','orange']
explode = (0, 0.1, 0, 0, 0, 0)  
fig = plt.figure(figsize=(15,8))
# Plot
plt.title('Mushroom cap shape Type Percentange', fontsize=22)
patches, texts, autotexts = plt.pie(cap_shapes_size, explode=explode, labels=cap_shape_labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=160)
for text,autotext in zip(texts,autotexts):
    text.set_fontsize(10)
    autotext.set_fontsize(10)

plt.axis('equal')
plt.show()


# In[441]:


labelencoder=LabelEncoder()
df[pd.isna(df)]="NaN"

for col in df.columns:
        df[col] = labelencoder.fit_transform(df[col])
df.head(5)


# In[442]:


dups = df.duplicated()
print('Number of duplicate rows = %d' % (dups.sum()))
### No duplicated data #####


# In[443]:


# fig, axes = plt.subplots(nrows=1 ,ncols=2 ,figsize=(9, 9))
# bp1 = axes[0,0].boxplot(df['stalk-color-above-ring'],patch_artist=True)

# bp2 = axes[0,1].boxplot(df['stalk-color-below-ring'],patch_artist=True)
ax = sns.boxplot(x='class', y='odor', 
                data=df)
plt.show()
ax = sns.boxplot(x='class', y='cap-shape', 
                data=df)
plt.show()
ax = sns.boxplot(x='class', y='cap-surface', 
                data=df)
plt.show()
ax = sns.boxplot(x='class', y='cap-color', 
                data=df)
plt.show()
ax = sns.boxplot(x='class', y='bruises', 
                data=df)
plt.show()


# In[444]:


df2=df[df["class"]==1]
df2['cap-shape'].hist()
plt.title('cap shape distribution in poisonous mushrooms')
plt.grid(True)
plt.show()

df3=df[df["class"]==0]
df3['cap-shape'].hist()
plt.title('cap shape distribution in poisonous mushrooms')
plt.grid(True)
plt.show()


# In[445]:


X = df.iloc[:,1:23]  # all rows, all the features and no labels
Y = df.iloc[:, 0]  # all rows, label only


# In[446]:


X.corr()


# In[447]:


scaler = StandardScaler()
X=scaler.fit_transform(X)
X


# In[448]:


##### To estimate feature importance ####
model = ExtraTreesClassifier()
model.fit(X, Y)
importance=model.feature_importances_.tolist()
features=df.drop('class',axis=1).columns.values.tolist()

fig, ax = plt.subplots(figsize=(20,5))
ind = np.arange(22) 
importance_bars = ax.bar(ind, importance , width=0.1, color=colors)
ax.set_xlabel("Features",fontsize=20)
ax.set_ylabel('Importance',fontsize=20)
ax.set_title('Feature importance',fontsize=22)
ax.set_xticks(ind) #Positioning on the x axis
ax.set_xticklabels(features,rotation='vertical')

index_=importance.index(max(importance))

most_important_features=features[index_]
print('Feature Importance: \n',model.feature_importances_)
print('The most important feature: \n',most_important_features)


# In[449]:


pca = PCA()
pca.fit_transform(X)

covariance=pca.get_covariance()
explained_variance=pca.explained_variance_ratio_

with plt.style.context('classic'):
    plt.figure(figsize=(8, 6))
    
    plt.bar(range(22), explained_variance, alpha=0.5, align='center',
            label='individual explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()
    for a,b in zip(range(22),explained_variance):
        plt.text(a, b+0.005, '%.2f' % b, ha='center', va= 'bottom',fontsize=7)


# In[450]:



pca = PCA(n_components=15)
X=pca.fit_transform(X)

df_pca=pd.DataFrame(X)
df_pca['class']=Y
df_pca


####### Prepared to building models #######
X_train, X_test, Y_train, Y_test = train_test_split(df_pca.iloc[:,:-1],df_pca.iloc[:,-1],test_size=0.2,random_state=4)
# # # X_train=pd.DataFrame(X_train)
# # X_test=pd.DataFrame(X_test)
# # Y_train=pd.DataFrame(Y_train)
# # Y_test=pd.DataFrame(Y_test)
# # X_train.columns([''])
# columns = ['pca_%i' % i for i in range(10)]
# X = DataFrame(pca.transform(X), columns=columns, index=X.index)
# X.head()


# In[451]:


### Decision Tree #######

dt=tree.DecisionTreeClassifier(criterion='entropy',random_state=10,max_depth=20)
dt=dt.fit(X_train,Y_train)
print('Scores of the classfier:\n', dt.score(X_test, Y_test))

# dot_data = tree.export_graphviz(dt, feature_names=X_train.columns, class_names=['poisonous','edible'], filled=True, 
#                                 out_file=None) 
# graph = pydotplus.graph_from_dot_data(dot_data) 
# Image(graph.create_png())


# In[452]:


# Observing Overfitting or Underfitting ######


maxdepths = [15,20,25,30,35,40,45,50,70,100,120,150,200]

trainAcc = np.zeros(len(maxdepths))
testAcc = np.zeros(len(maxdepths))

index = 0
for depth in maxdepths:
    dt = tree.DecisionTreeClassifier(max_depth=depth)
    dt = dt.fit(X_train, Y_train)
    Y_predTrain = dt.predict(X_train)
    Y_predTest = dt.predict(X_test)
    trainAcc[index] = accuracy_score(Y_train, Y_predTrain)
    testAcc[index] = accuracy_score(Y_test, Y_predTest)
    index += 1
    
# Plot of training and test accuracies

    
plt.plot(maxdepths,trainAcc,'ro-',maxdepths,testAcc,'bv--')
plt.legend(['Training Accuracy','Test Accuracy'])
plt.xlabel('Max depth')
plt.ylabel('Accuracy')


# In[453]:


dt=tree.DecisionTreeClassifier(criterion='entropy',random_state=10,max_depth=20)
dt = dt.fit(X_train, Y_train)

y_pred = dt.predict(X_test)

cfm = confusion_matrix(Y_test, y_pred)
print(cfm)

print(classification_report(Y_test,y_pred))


# In[454]:


### Random Forest ######

rdf = RandomForestClassifier(n_estimators = 30, criterion = 'entropy', random_state = 42)
rdf.fit(X_train, Y_train)
y_pred = rdf.predict(X_test)
cfm = confusion_matrix(Y_test, y_pred)
print(cfm)

print(classification_report(Y_test,y_pred))


# In[455]:



#### SVM ######

C = [0.01, 0.1, 0.2, 0.5, 0.8, 1, 5, 10, 20, 50]
SVMtrainAcc = []
SVMtestAcc = []

for param in C:
    svm = SVC(C=param,kernel='rbf',gamma='auto')
    svm.fit(X_train, Y_train)
    Y_predTrain = svm.predict(X_train)
    Y_predTest = svm.predict(X_test)
    SVMtrainAcc.append(accuracy_score(Y_train, Y_predTrain))
    SVMtestAcc.append(accuracy_score(Y_test, Y_predTest))

plt.plot(C, SVMtrainAcc, 'ro-', C, SVMtestAcc,'bv--')
plt.legend(['Training Accuracy','Test Accuracy'])
plt.xlabel('C')
plt.xscale('log')
plt.ylabel('Accuracy')
plt.show()
### Find the optimal hyperparameter C ######
svm = SVC(C=1,kernel='rbf',gamma='auto',probability=True)
svm.fit(X_train, Y_train)
print('Scores of the classfier:\n', svm.score(X_test, Y_test))
y_pred = svm.predict(X_test)

cfm = confusion_matrix(Y_test, y_pred)
print('Confusion matrix: \n',cfm)

print(classification_report(Y_test,y_pred))


# In[456]:


class LogisticRegression_Scratch:
    #### Initiate object with learning_rate, num_iteration, here, I allow to add the intercept#####
    def __init__(self,  num_iter,learning_rate, fit_intercept=True):
        self.learning_rate = learning_rate
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
    ##### Initiate intercept as 1 ####
    def add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def sigmoid(self, z):
       #### probability function ####
        return 1 / (1 + np.exp(-z))
    ### loss function #####
    def loss_funtion(self, p, y):
        return (-y * np.log(p) - (1 - y) * np.log(1 - p)).mean()
    
    def fit(self, X, y):
        if self.fit_intercept:
            X = self.add_intercept(X)
        ### Initialize weights theta###
        self.theta = np.zeros(X.shape[1])
       ### Update weights theta num_iter times #### 
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            p = self.sigmoid(z)
            ### calculate the gradient descent of loss function with respect to theta ######
            gradient_descent = np.dot(X.T, (p - y)) / y.size
            ### Update theta#
            self.theta -= self.learning_rate * gradient_descent
        print('Intercept and Coefficient of each attributes: \n',self.theta)
    
    ####Calculate prediction probability ####
    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.add_intercept(X)
        z=np.dot(X, self.theta)
        return self.sigmoid(z)
    ### Determine class labels as either 1 or 0 by comparing with threshold #####
    def predict(self, X, threshold):
        return self.predict_prob(X) >= threshold
    
    


# In[457]:


### Using benchmark dataset which is wine quality dataset to test its performance #####
benchmark_df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv' ,sep=';',header=0)
benchmark_df.head()
benchmark_df['class'] = benchmark_df['quality'].apply(lambda x: 0 if x<=5 else 1)  
#Create a binary class
benchmark_df=benchmark_df.drop(['quality'],axis=1)
benchmark_df.head(10)
benchmark_X=benchmark_df.drop(['class'],axis=1)
benchmark_Y=benchmark_df['class']
scaler = StandardScaler()
benchmark_X=scaler.fit_transform(benchmark_X)
benchmark_X_train,benchmark_X_test,benchmark_Y_train,benchmark_Y_test=train_test_split(benchmark_X,benchmark_Y,test_size=0.2,random_state=4)
LR_scratch=LogisticRegression_Scratch(num_iter=30000,learning_rate=0.5)
LR_scratch.fit(benchmark_X_train,benchmark_Y_train)
y_pred_bm=LR_scratch.predict(benchmark_X_test,0.5)
cfm = confusion_matrix(benchmark_Y_test, y_pred_bm)
print('Confusion matrix: \n',cfm)
print(classification_report(benchmark_Y_test,y_pred_bm))


# In[477]:


LR_scratch=LogisticRegression_Scratch(num_iter=20000,learning_rate=0.05)
LR_scratch.fit(X_train,Y_train)
y_pred1=LR_scratch.predict(X_test,0.4)
cfm = confusion_matrix(Y_test, y_pred1)
print('Confusion matrix: \n',cfm)
print(classification_report(Y_test,y_pred1))


# In[474]:


LR = LogisticRegression(random_state=10, solver='sag').fit(X_train, Y_train)
print('Intercept and Coefficient of each attributes: \n',np.insert(LR.coef_[0],0,LR.intercept_))
y_pred2=LR.predict(X_test)
cfm = confusion_matrix(Y_test, y_pred2)
print('Confusion matrix: \n',cfm)
print(classification_report(Y_test,y_pred2))


# In[481]:


#### kNN ####
### Since LabelEncoder will bring unexpected 'order' to each attributes, so we should transform each attribute to some dummies variables and Standalization ###
### Have to do another preprocessing steps #####

####### Preprocessing #####
scaler=StandardScaler()
df_dummies=pd.get_dummies(df,columns=df.columns)
X_dummies=df_dummies.drop(['class_0','class_1'],axis=1)
X_dummies=scaler.fit_transform(X_dummies)
pca = PCA(n_components=15)
X_dummies=pca.fit_transform(X_dummies)

Y=df['class']
X_train_dummies, X_test_dummies, Y_train, Y_test = train_test_split(X_dummies,Y,test_size=0.2,random_state=4)
########## Finding best value of k #######
numNeighbors=[2,5,7,10,15]
trainAcc = []
testAcc = []
for k in numNeighbors:
    knn = KNeighborsClassifier(n_neighbors=k, metric='minkowski', p=2)
    knn.fit(X_train_dummies, Y_train)
    Y_predTrain = knn.predict(X_train_dummies)
    Y_predTest = knn.predict(X_test_dummies)
    trainAcc.append(accuracy_score(Y_train, Y_predTrain))
    testAcc.append(accuracy_score(Y_test, Y_predTest))

plt.plot(numNeighbors, trainAcc, 'ro-', numNeighbors, testAcc,'bv--')
plt.legend(['Training Accuracy','Test Accuracy'])
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()
#### Decided k = 5 ####
knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
knn.fit(X_train_dummies, Y_train)

y_pred = knn.predict(X_test_dummies)
cfm = confusion_matrix(Y_test, y_pred)
print(cfm)

print(classification_report(Y_test,y_pred))


# In[480]:


###### Cross Validation to select model ######
### Decision Tree validation #####
i=1
accuracy=0
kFold = KFold(n_splits=5, shuffle=True, random_state=None)
for train_index, validation_index in kFold.split(X_train):
    X_train2 = X_train.iloc[train_index]
    X_validation = X_train.iloc[validation_index]
    Y_train2 = Y_train.iloc[train_index]
    Y_validation = Y_train.iloc[validation_index]
    dt.fit(X_train2,Y_train2)
    y_pred=dt.predict(X_validation)
    print("{}'s Iteration\n".format(i))
    print('Scores: \n',dt.score(X_validation,Y_validation))
    print('\n',confusion_matrix(Y_validation,y_pred),'\n')
    print(classification_report(Y_validation,y_pred))
    ### ROC curve for each run ###
    probs = dt.predict_proba(X_validation)
    preds = probs[:,1]

    fpr, tpr, threshold = metrics.roc_curve(Y_validation, preds,pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic of Decision Tree')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    i=i+1
    score=dt.score(X_validation,Y_validation)
    accuracy=accuracy+score
    
print('Average accuracy of k-runs: \n',(accuracy/5))


# In[462]:


#### Cross Validation to evaluate Decision Tree using average scores #####
scores = cross_val_score(dt, X_train, Y_train, cv=5, scoring='accuracy')

print(scores)

print('Mean score:\n',scores.mean())


# In[463]:


####  Random Forest validation #####
i=1
accuracy=0
kFold = KFold(n_splits=5, shuffle=True, random_state=None)
for train_index, validation_index in kFold.split(X_train):
    X_train2 = X_train.iloc[train_index]
    X_validation = X_train.iloc[validation_index]
    Y_train2 = Y_train.iloc[train_index]
    Y_validation = Y_train.iloc[validation_index]
    rdf.fit(X_train2,Y_train2)
    y_pred=rdf.predict(X_validation)
    print("{}'s Iteration\n".format(i))
    print('Scores: \n',rdf.score(X_validation,Y_validation))
    print('\n',confusion_matrix(Y_validation,y_pred),'\n')
    print(classification_report(Y_validation,y_pred))
    ### ROC curve for each run ###
    probs = rdf.predict_proba(X_validation)
    preds = probs[:,1]

    fpr, tpr, threshold = metrics.roc_curve(Y_validation, preds,pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic of Random Forest')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    i=i+1
    score=rdf.score(X_validation,Y_validation)
    accuracy=accuracy+score
    
print('Average accuracy of k-runs: \n',(accuracy/5))


# In[464]:


scores = cross_val_score(rdf, X_train, Y_train, cv=5, scoring='accuracy')

print(scores)

print('Mean score:\n',scores.mean())


# In[465]:



##### SVM validation ###

i=1
accuracy=0
kFold = KFold(n_splits=5, shuffle=True, random_state=None)
for train_index, validation_index in kFold.split(X_train):
    X_train2 = X_train.iloc[train_index]
    X_validation = X_train.iloc[validation_index]
    Y_train2 = Y_train.iloc[train_index]
    Y_validation = Y_train.iloc[validation_index]
    svm.fit(X_train2,Y_train2)
    y_pred=rdf.predict(X_validation)
    print("{}'s Iteration\n".format(i))
    print('Scores: \n',svm.score(X_validation,Y_validation))
    print('\n',confusion_matrix(Y_validation,y_pred),'\n')
    print(classification_report(Y_validation,y_pred))
    ### ROC curve for each run ###
    probs = svm.predict_proba(X_validation)
    
    preds = probs[:,1]

    fpr, tpr, threshold = metrics.roc_curve(Y_validation, preds,pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic of SVM')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    i=i+1
    score=svm.score(X_validation,Y_validation)
    accuracy=accuracy+score
    
print('Average accuracy of k-runs: \n',(accuracy/5))


# In[466]:


scores = cross_val_score(svm, X_train, Y_train, cv=5, scoring='accuracy')

print(scores)

print('Mean score:\n',scores.mean())


# In[467]:


##### LogesticRegression_scratch #####

i=1
accuracy=0
kFold = KFold(n_splits=5, shuffle=True, random_state=None)
for train_index, validation_index in kFold.split(X_train):
    X_train2 = X_train.iloc[train_index]
    X_validation = X_train.iloc[validation_index]
    Y_train2 = Y_train.iloc[train_index]
    Y_validation = Y_train.iloc[validation_index]
    LR_scratch.fit(X_train2,Y_train2)
    y_pred=LR_scratch.predict(X_validation,0.5)
    print("{}'s Iteration\n".format(i))
    print('Scores: \n',accuracy_score(Y_validation,y_pred))
    print('\n',confusion_matrix(Y_validation,y_pred),'\n')
    print(classification_report(Y_validation,y_pred))
    ### ROC curve for each run ###
    probs = LR_scratch.predict_prob(X_validation)
    
#     preds = probs[:,0]

    fpr, tpr, threshold = metrics.roc_curve(Y_validation, probs,pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic of Logistic Regression from scratch')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    i=i+1
    score=accuracy_score(Y_validation,y_pred)
    accuracy=accuracy+score
    
print('Average accuracy of k-runs: \n',(accuracy/5))


# In[468]:


##### LogisticRegression #####


i=1
accuracy=0
kFold = KFold(n_splits=5, shuffle=True, random_state=None)
for train_index, validation_index in kFold.split(X_train):
    X_train2 = X_train.iloc[train_index]
    X_validation = X_train.iloc[validation_index]
    Y_train2 = Y_train.iloc[train_index]
    Y_validation = Y_train.iloc[validation_index]
    LR.fit(X_train2,Y_train2)
    y_pred=LR.predict(X_validation)
    print("{}'s Iteration\n".format(i))
    print('Scores: \n',LR.score(X_validation,Y_validation))
    print('\n',confusion_matrix(Y_validation,y_pred),'\n')
    print(classification_report(Y_validation,y_pred))
    ### ROC curve for each run ###
    probs = LR.predict_proba(X_validation)
    
    preds = probs[:,1]

    fpr, tpr, threshold = metrics.roc_curve(Y_validation, preds,pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic of Logistic Regression')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    i=i+1
    score=LR.score(X_validation,Y_validation)
    accuracy=accuracy+score
    
print('Average accuracy of k-runs: \n',(accuracy/5))


# In[469]:


scores = cross_val_score(LR, X_train, Y_train, cv=5, scoring='accuracy')

print(scores)

print('Mean score:\n',scores.mean())


# In[470]:


### kNN ######
i=1
accuracy=0
kFold = KFold(n_splits=5, shuffle=True, random_state=None)

X_train_dummies=pd.DataFrame(X_train_dummies)
for train_index, validation_index in kFold.split(X_train_dummies):
    X_train2 = X_train_dummies.iloc[train_index]
    X_validation = X_train_dummies.iloc[validation_index]
    Y_train2 = Y_train.iloc[train_index]
    Y_validation = Y_train.iloc[validation_index]
    knn.fit(X_train2,Y_train2)
    y_pred=knn.predict(X_validation)
    print("{}'s Iteration\n".format(i))
    print('Scores: \n',knn.score(X_validation,Y_validation))
    print('\n',confusion_matrix(Y_validation,y_pred),'\n')
    print(classification_report(Y_validation,y_pred))
    ### ROC curve for each run ###
    probs = knn.predict_proba(X_validation)

    preds = probs[:,1]

    fpr, tpr, threshold = metrics.roc_curve(Y_validation, preds,pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic of kNN')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    i=i+1
    score=knn.score(X_validation,Y_validation)
    accuracy=accuracy+score
    
print('Average accuracy of k-runs: \n',(accuracy/5))


# In[471]:


scores = cross_val_score(knn, X_train, Y_train, cv=5, scoring='accuracy')

print(scores)

print('Mean score:\n',scores.mean())


# In[482]:


##### knn, SVM, Random Forest highest scores,  Decision tree a little bit lower, the two Logistic Regression classifier loweset with about 0.90 ##
### knn might cause dimension sparse ####
### Choose kNN  as my model ####
knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
knn.fit(X_train_dummies, Y_train)
print('Scores of the kNN classfier:\n', knn.score(X_test_dummies, Y_test))
y_pred = knn.predict(X_test_dummies)
cfm = confusion_matrix(Y_test, y_pred)
print(cfm)

print(classification_report(Y_test,y_pred))
# svm = SVC(C=1,kernel='rbf',gamma='auto',probability=True)
# svm.fit(X_train, Y_train)
# print('Scores of the SVM classfier:\n', svm.score(X_test, Y_test))
# y_pred = svm.predict(X_test)

# cfm = confusion_matrix(Y_test, y_pred)
# print('Confusion matrix: \n',cfm)

# print(classification_report(Y_test,y_pred))

def get_confusion_matrix_values(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return(cm[0][0], cm[0][1], cm[1][0], cm[1][1])


TN, FP, FN, TP = get_confusion_matrix_values(Y_test, y_pred)
print('\nTPR:  ',TP/(TP+FN))
print('\nFPR:  ',FP/(FP+TN))


# In[ ]:




