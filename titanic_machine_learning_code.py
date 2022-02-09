#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd

df = pd.read_csv('./titanic/train.csv')
#트레이닝데이터를 가져온다. 

# In[3]:


print(df.head(5))
#트레이닝데이터의 내용을 위에서부터 5줄 확인한다. 

# In[4]:


print(df.info())
#트레이닝데이터의 정보를 읽는다. 데이터타입과 결측치가 얼마나 있는지를 확인할 수 있다. 

# In[5]:


import matplotlib.pyplot as plt
import seaborn as sb


# In[6]:


plt.figure(figsize = (12, 12))
#그래프 사이즈를 설정한다. 

# In[7]:


colormap = plt.cm.gist_heat
#히트맵을 그리기 위해 설정한다. 

# In[8]:


df.corr()
#데이터 feature들 간의 관계를 확인한다. Survived와 관련있는 데이터는 Parch와 Fare로 보인다. 

# In[9]:


sb.heatmap(df.corr(), linewidths=0.2, vmax=0.5, cmap=colormap, linecolor = 'white', annot=True)
#히트맵의 세부 조건을 설정한다. 

# In[10]:


grid = sb.FacetGrid(df, col='Survived')
grid.map(plt.hist, 'Fare', bins = 10)
plt.show()
#Survived와 Fare의 관계를 그래프로 확인한다. 

# In[11]:


grid = sb.FacetGrid(df, col='Survived')
grid.map(plt.hist, 'Parch', bins = 10)
plt.show()
#Survived와 Parch의 관계를 그래프로 확인한다. 

# In[12]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder

import numpy as np
import tensorflow as tf

np.random.seed(3)
tf.random.set_seed(3)


# In[13]:

#위에서 확인했던 내용들을 바탕으로 사용할 데이터들을 정리한다. 
df = pd.read_csv('./titanic/train.csv')
df2 = pd.read_csv('./titanic/test.csv')
#트레이닝데이터와 테스트데이터를 불러온다. 
df.fillna({'Embarked':df['Embarked'].mode()[0],'Age':int(df['Age'].mean())},inplace=True)
df2.fillna({'Embarked':df2['Embarked'].mode()[0],'Age':int(df2['Age'].mean())},inplace=True)
#결측치를 대체한다. Embarked는 데이터가 빈 곳이 많지 않기 때문에 가장 많이 발생한 값으로 채워주고 나이는 평균으로 대체한다. 
df=df.drop('Cabin', axis=1)
df2=df2.drop('Cabin', axis=1)
#결측치가 발생한 곳이 많기 때문에 예측에 도움이 되지 않는다고 판단해 제거해준다. 
df=df.drop('Pclass', axis=1)
df2=df2.drop('Pclass', axis=1)
#위에서 확인한 히트맵에서 생존율과 가장 관련 없는 것으로 나타났기 때문에 제거해준다. 
df=df.drop('Name', axis=1)
df2=df2.drop('Name', axis=1)
#이름과 생존율은 관계가 없다고 판단하였다. 
df=df.drop('PassengerId', axis=1)
df2=df2.drop('PassengerId', axis=1)
#PassengerId와 생존율은 관계가 없다고 판단하였다. 
df=df.drop('Ticket', axis=1)
df2=df2.drop('Ticket', axis=1)
#불규칙한 형태의 Ticket은 생존율과 관계 없다고 판단하였다. 

# In[14]:


import seaborn as sns

sns.pairplot(df, hue='Survived');
plt.show()
#생존 여부를 데이터들과의 관계로부터 충분히 구분해낼 수 있을지 그래프를 통해 대략적으로 확인한다. 

# In[15]:


Data_set = df.values
Data_set2 = df2.values
#트레이닝데이터, 테스트데이터를 numpy형태로 받아낸다. 
X= Data_set[:,1:7]
X2= Data_set2[:,0:6]
Y= Data_set[:,0]
#삭제한 데이터를 고려햐여 데이터의 크기를 설정한다. Y는 Survived이고 X는 나머지 데이터들이다. 
#테스트데이터에는 Survived 항목이 없기 때문에 Y2가 따로 존재하지 않는다. 

# In[16]:


df.head()
#정리된 데이터들을 앞에서부터 5줄 확인한다. 

# In[17]:


from sklearn.preprocessing import LabelEncoder
#데이터타입을 숫자로 통일해주기 위한 작업이다. 
#트레이닝데이터, 테스트 데이터 모두 Sex, Embarked 항목을 숫자로 변경해준다. 
N0 = X[:,0]
e = LabelEncoder()
e.fit(N0)
M0 = e.transform(N0)
X[:,0]=M0

N5 = X[:,5]
e = LabelEncoder()
e.fit(N5)
M5 = e.transform(N5)
X[:,5]=M5

N0_ = X2[:,0]
e = LabelEncoder()
e.fit(N0_)
M0_ = e.transform(N0_)
X2[:,0]=M0_

N5_ = X2[:,5]
e = LabelEncoder()
e.fit(N5_)
M5_ = e.transform(N5_)
X2[:,5]=M5_


T=X.astype(float)
P=Y.astype(int)

T2=X2.astype(float)


# In[25]:


model = Sequential()
model.add(Dense(30, input_dim=6, activation = 'relu'))
model.add(Dense(6, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))
#6가지 feature를 다루므로 들어가는 데이터는 6개이고 30개의 노드 은닉층을 가진다. 하나의 결과가 나타난다. 
model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])
#loss함수는 binary_crossentropy를 사용한다. adam알고리즘을 사용한다. accuracy라는 matrics를 사용한다. 
model.fit(T, P, epochs=200, batch_size=10)
#epochs를 200으로 설정하고 batch_size를 10으로 설정하여 10 간격으로 200번 시뮬레이션을 수행하는 모델이다. 

# In[21]:


prediction = model.predict(T2)
#예측 모델을 사용한다. 입력데이터는 위에서 적절하게 만들어둔 테스트데이터이다. 
submit = pd.read_csv('./titanic/gender_submission.csv')
#예측값을 제출하기 위해 gender_submission 파일을 사용한다. 
print(submit.head())
for i in range(len(prediction)):
    if prediction[i][0]>0.5 : 
        prediction[i][0] = 1
    else:
        prediction[i][0] = 0
#확률로 나타나기 때문에 적절한 값을 넣기 위해 범위를 설정해준다. 
prediction = np.array(prediction, dtype=np.int64)
submit['Survived'] = prediction
submit.to_csv('submit_first.csv', index=False)
#예측된 모든 값들을 submit_first파일을 만들어 담는다. 

# In[ ]:




