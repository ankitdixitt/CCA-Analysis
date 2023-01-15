#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[2]:


data1 = pd.read_csv("C:/Users/ankit/OneDrive/Desktop/New folder/CHD1_1.fcs.csv")

X=data1[['FSC-A','SSC-A']]
X


# In[3]:


X_mc = (X-X.mean())/(X.std())
X_mc


# In[ ]:





# In[13]:


data2=pd.read_csv("C:\\Users\\ankit\\Downloads\\PCD1_3.fcs.csv") 
Y=data2[['FSC-A','SSC-A']]
Y


# In[14]:


Y_mc = (Y-Y.mean())/(Y.std())
Y_mc


# In[15]:


from sklearn.cross_decomposition import CCA


# In[16]:


ca = CCA()
ca.fit(X_mc, Y_mc)
X_c, Y_c = ca.transform(X_mc, Y_mc)
print(X_c.shape)
print(Y_c.shape)


# In[17]:


cc_res = pd.DataFrame({"CCX_1":X_c[:, 0],
                       "CCY_1":Y_c[:, 0],
                       "CCX_2":X_c[:, 1],
                       "CCY_2":Y_c[:, 1],
                       })
cc_res


# In[18]:


import numpy as np
np.corrcoef(X_c[:, 0], Y_c[:, 0])


# In[19]:


np.corrcoef(X_c[:, 1], Y_c[:, 1])
 
    


# In[29]:


sns.set_context("talk", font_scale=1)
plt.figure(figsize=(10,8))
sns.scatterplot(x="CCX_1",
                y="CCY_1", 
                data=cc_res)
plt.title('Comp. 1, corr = %.2f' %
         np.corrcoef(X_c[:, 0], Y_c[:, 0])[0, 1])


# In[22]:


plt.figure(figsize=(10,8))
sns.scatterplot(x="CCX_1",
                y="CCY_1", data=cc_res)
plt.title('First Pair of Canonical Covariate, corr = %.2f' %
         np.corrcoef(X_c[:, 0], Y_c[:, 0])[0, 1])


# In[23]:


ccX_df = pd.DataFrame({"CCX_1":X_c[:, 0],
                       "CCX_2":X_c[:, 1],
                       })
ccX_df


# In[24]:


corr_X_df= ccX_df.corr(method='pearson') 
corr_X_df.head()


# In[25]:


plt.figure()
X_df_lt = corr_X_df.where(np.tril(np.ones(corr_X_df.shape)).astype(np.bool))


# In[26]:


sns.heatmap(X_df_lt,cmap="coolwarm",annot=True,fmt='.1g')
plt.tight_layout()
plt.savefig("Heatmap_Canonical_Correlates_from_X_and_data.jpg",
                    format='jpeg',
                    dpi=100)


# In[27]:


ccY_df = pd.DataFrame({"CCY_1":Y_c[:, 0],
                       "CCY_2":Y_c[:, 1],
                       })
 
# compute correlation with Pandas corr()
corr_Y_df= ccY_df.corr(method='pearson') 
 
# Get lower triangular correlation matrix
Y_df_lt = corr_Y_df.where(np.tril(np.ones(corr_Y_df.shape)).astype(np.bool))
 
# make a lower triangular correlation heatmap with Seaborn
plt.figure(figsize=(10,8))
sns.heatmap(Y_df_lt,cmap="coolwarm",annot=True,fmt='.1g')
plt.tight_layout()
plt.savefig("Heatmap_Canonical_Correlates_from_Y_and_data.jpg",
                    format='jpeg',
                    dpi=100)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




