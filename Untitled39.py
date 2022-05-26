#!/usr/bin/env python
# coding: utf-8

# In[50]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


data=pd.read_excel('/Users/daskrantik01/Downloads/Project/Book5.xlsx')


# In[6]:


data


# In[7]:


df1=data.iloc[:,1:6]


# In[8]:


df1


# In[9]:


df1.corr()


# In[10]:


plt.figure(figsize=(20,10))
sns.heatmap(abs(df1.corr()), cmap='GnBu', annot=True);


# In[11]:


#first estimating GDP growth solely based on FDI inflow
x=data.iloc[:,1]
y=data.iloc[:,2]
z=data.iloc[:,3]
k=data.iloc[:,4]


# In[12]:


z


# In[13]:


mean_x=np.mean(x)
mean_y=np.mean(y)

m=len(x)

#calculating b1 and b0

numer=0
denom=0

for i in range(m):
    numer += (x[i] - mean_x)*(y[i] - mean_y)
    denom += (x[i] - mean_x)**2
    
b1=numer/denom
b0=mean_y-(b1*mean_x)

print (b1,b0)


# In[14]:


#How the independent variables interact with ecah other
sns.pairplot(data, x_vars=['Foreign direct investment, net inflows (BoP, current US$)', 'Exports of goods and services (current US$)', 'Employment to population ratio, ages 15-24, total (%) (modeled ILO estimate)'], y_vars='GDP (current US$)', height=4, aspect=1, kind='scatter')
plt.show()


# In[15]:


sns.distplot(data['GDP (current US$)']);


# In[16]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

x = data[['Foreign direct investment, net inflows (BoP, current US$)']]
y = data['GDP (current US$)']


# In[17]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 100)


# In[18]:


slr= LinearRegression()  
slr.fit(x_train, y_train)


# In[19]:


#Printing the model coefficients
print('Intercept: ', slr.intercept_)
print('Coefficient:', slr.coef_)


# In[20]:


print('GDP = 352.28837359832823 + 42.40049472 * FDI')


# In[21]:


#Line of best fit
plt.scatter(x_train, y_train)
plt.plot(x_train, 352.28837359832823 + 42.40049472*x_train, 'r')
plt.show()


# In[22]:


#Prediction of Test and Training set result  
y_pred_slr= slr.predict(x_test)  
x_pred_slr= slr.predict(x_train) 


# In[23]:


print("Prediction for test set: {}".format(y_pred_slr))


# In[24]:


slr_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred_slr})
slr_diff


# In[25]:


slr.predict([[64.36]])


# In[26]:


print('R squared value of the model: {:.2f}'.format(slr.score(x,y)*100))


# In[27]:


meanAbErr = metrics.mean_absolute_error(y_test, y_pred_slr)
meanSqErr = metrics.mean_squared_error(y_test, y_pred_slr)
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_slr))

print('Mean Absolute Error:', meanAbErr)
print('Mean Square Error:', meanSqErr)
print('Root Mean Square Error:', rootMeanSqErr)


# In[28]:


from statsmodels.stats.stattools import durbin_watson
durbinWatson = durbin_watson(x_pred_slr)

print(durbinWatson)


# In[29]:


x = data[['Foreign direct investment, net inflows (BoP, current US$)', 'Exports of goods and services (current US$)', 'Employment to population ratio, ages 15-24, total (%) (modeled ILO estimate)']]
y = data['GDP (current US$)']


# In[30]:


x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.3, random_state=100)  


# In[31]:


mlr= LinearRegression()  
mlr.fit(x_train, y_train) 


# In[32]:


#Printing the model coefficients
print(mlr.intercept_)
# pair the feature names with the coefficients
list(zip(x, mlr.coef_))


# In[33]:


y_pred_mlr= mlr.predict(x_test)  
x_pred_mlr= mlr.predict(x_train)  


# In[34]:


print("Prediction for test set: {}".format(y_pred_mlr))


# In[35]:


mlr_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred_mlr})
mlr_diff


# In[37]:


mlr.predict([[64.36, 496.49, 17.458]])


# In[38]:


print('R squared value of the model: {:.2f}'.format(mlr.score(x,y)*100))


# In[39]:


meanAbErr = metrics.mean_absolute_error(y_test, y_pred_mlr)
meanSqErr = metrics.mean_squared_error(y_test, y_pred_mlr)
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_mlr))

print('Mean Absolute Error:', meanAbErr)
print('Mean Square Error:', meanSqErr)
print('Root Mean Square Error:', rootMeanSqErr)


# In[40]:


print('GDP = 296.985565736239 + 17.655524919923796 * FDI+2.6010555064586685*Exports-2.3617188809936485*Employment')


# In[41]:


from statsmodels.stats.stattools import durbin_watson
durbinWatson = durbin_watson(y_pred_mlr)

print(durbinWatson)


# In[42]:


from statsmodels.stats.stattools import durbin_watson
durbinWatson = durbin_watson(x_pred_mlr)

print(durbinWatson)


# In[53]:


from numpy install scipy


def f_test(x, y):
    x = np.array(x)
    y = np.array(y)
    f = np.var(x, ddof=1)/np.var(y, ddof=1) #calculate F test statistic 
    dfn = x.size-1 
    dfd = y.size-1 
    p = 1-scipy.stats.f.cdf(f, dfn, dfd) #find p-value of F test statistic 
    return f, p

#perform F-test
f_test(x, y)

(4.38712, 0.019127)


# In[ ]:




