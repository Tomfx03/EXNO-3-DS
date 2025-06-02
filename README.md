## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```
![image](https://github.com/user-attachments/assets/d475367f-9eef-461e-b84c-1bb77936640b)
```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/user-attachments/assets/90f44c60-3cb3-4407-8397-c0a205861b5e)
```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/user-attachments/assets/90eaa753-80db-45dd-a548-2d2501d46b37)
```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/3445eeea-e739-4462-aa4e-159c3ec995d9)
```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df=df.copy()

enc=pd.DataFrame(ohe.fit_transform(df[['nom_0']]))
enc
```
![image](https://github.com/user-attachments/assets/15abcfbc-d8db-444e-8318-db667098d695)
```
df2=pd.concat([df2,enc],axis=1)
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/user-attachments/assets/de98fb73-e72f-48cd-a340-9da189d3cd8c)
```
pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/user-attachments/assets/ea0fb468-3ea0-4fd1-b6c4-34aad59c62ea)
```
pip install --upgrade category_encoders
```
```
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
```
![image](https://github.com/user-attachments/assets/f986b30e-dbe6-4f64-ad0f-337fdd4fbcca)
```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```
![image](https://github.com/user-attachments/assets/b5126319-ea55-4059-afea-f94bef40d319)
```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
![image](https://github.com/user-attachments/assets/16980479-5726-4b05-9e3e-18111a4bd3c2)
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```
![image](https://github.com/user-attachments/assets/3c17001b-eb20-4a2a-9e9a-01989d7394bd)
```
df.skew()
```
![image](https://github.com/user-attachments/assets/e6534af3-e3de-44e2-b69c-3ea7da6ee72e)
```
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/44d18a0d-17b9-4ae3-bcbd-921df0f9373a)
```
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/user-attachments/assets/f9d7f15d-3984-429b-9b3b-6c7c30b7b0b1)
```
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/4bbcdae0-df32-4cec-87f9-d9cf7b4e0ae9)
```
np.square(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/450e46a8-c017-4380-b1c0-4c728c292728)
```
df["Highly Positve Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/e3f0cc58-84c9-4b97-a13d-fcdf396bc1bd)
```
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df.skew()
```
![image](https://github.com/user-attachments/assets/8ff2e5e8-6404-450d-a3d6-b342002bdcd5)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
![image](https://github.com/user-attachments/assets/d8ce2302-dc3d-4c3e-bc78-3c80c2ef22dc)
```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![download](https://github.com/user-attachments/assets/f1b02806-7cdb-48ce-893e-0d358b78e547)
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![download](https://github.com/user-attachments/assets/8acbab29-ce5a-4e5a-8f8a-2f3f031e3633)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![download](https://github.com/user-attachments/assets/e3c8c56a-fe51-4fa6-93e2-cf942a64bc4b)
```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
![download](https://github.com/user-attachments/assets/cda02b5d-3276-43ef-9918-75798848db98)
```
df
```
![image](https://github.com/user-attachments/assets/9c69fdb1-841a-42a9-8281-6d9ec6c40de9)
```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
![download](https://github.com/user-attachments/assets/8a258325-ed13-4261-9c11-f0462bc839df)

# RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.



       
