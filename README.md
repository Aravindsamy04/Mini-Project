### EX NO : 10 Mini Project
# Rainfall Analysis of India
# DATE : 09/11/2023
# Description :

As is widely recognized, rainfall data is necessary for the mathematical modelling of extreme hydrological events, such as droughts or floods, as well as for evaluating surface and subsurface water resources and their quality. The phase, quantity, and elevation of generic hydrometeors in the atmosphere can be estimated by ground-based radars. Satellites can provide images with visible and infrared radiation, and they can also serve as platforms for radiometers to derive the quantity and phase of hydrometeors. Radars and satellites provide spatial information on precipitation at wide scales, avoiding many problems connected to local ground measurements, including those for the areal inhomogeneity of a network. However, direct rainfall observations at point scale can be obtained only by rain gauges installed at the soil surface.

### KEY FEATURES :
Three main characteristics of rainfall are its amount, frequency and intensity, the values of which vary from place to place, day to day, month to month and also year to year. Precise knowledge of these three main characteristics is essential for planning its full utilization.

### CODE :
```
DEVELOPED BY : ARAVIND SAMY P

REGISTER NO : 212222230011
```
```
import numpy as np 
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import pickle
from sklearn.model_selection import train_test_split
#read the whole file 
df.iloc[:]
df.info()
print(f'Number of Rows: {df.shape[0]}')
print(f'Number of Columns: {df.shape[1]}')
df.columns
row3 = df.iloc[3]
print(row3)
print("\n")
row115 = df.iloc[115]
print (row115)
description = df.describe()
print(description)
#plot the actual rain 
dg = df[['jun','jul','aug','sep']]
dg.plot(figsize=(20, 5));
#plot shows the total rain 
dh = df[['total']]
dh.plot(figsize=(10, 10));
#dy represent dataframe for 10 last rows which is rainfall from year 2007 to 2016 
dy = df.tail(10)
print(dy)
#plot diagram shows only "TOTAL" rain from dy dataframe (rainfall from year 2007 to 2016)
de = dy[['total']]
de.plot(figsize=(10, 10));
#rainfall from 2007 to 2016
#plot actual rain and departure percentage 
a_d = dy[['jun', 'jul', 'aug', 'sep', 'jun_p', 'jul_p', 'aug_p', 'sep_p']]
a_d.plot(figsize=(20, 10));
#rainfall from 2007 to 2016
#only plot actual rain 
actual = dy[['jun', 'jul', 'aug', 'sep']]
actual
actual = dy[['jun', 'jul', 'aug', 'sep']]
# Calculate the rainfall ranges for each month
rainfall_ranges = actual.max() - actual.min()
# Display the rainfall ranges, minimum, and maximum values
for month, rainfall_range in rainfall_ranges.items():
    min_rainfall = actual[month].min()
    max_rainfall = actual[month].max()
    
    print(f"Rainfall Range for {month}: Range = {rainfall_range} mm,", end=' ')
    print(f"Min = {min_rainfall} mm, Max = {max_rainfall} mm")
actual.plot(figsize=(20, 10))
# Add a title and labels for the axes
plt.title('Actual Rainfall (mm) vs Year')
plt.xlabel('Year')
plt.ylabel('Actual Rainfall (mm)')
plt.show()
#rainfall from 2007 to 2016
#plot departure percentage 
de = dy[['jun_p', 'jul_p', 'aug_p', 'sep_p']]
de
de = dy[['jun_p', 'jul_p', 'aug_p', 'sep_p']]
# Calculate the rainfall ranges for each month
rainfall_ranges = de.max() - de.min()
# Display the rainfall ranges, minimum, and maximum values
for month, rainfall_range in rainfall_ranges.items():
    min_rainfall = de[month].min()
    max_rainfall = de[month].max()
    
    print(f"Rainfall Range for {month}: Range = {rainfall_range} %,", end=' ')
    print(f"Min = {min_rainfall} %, Max = {max_rainfall} %")
print("\n")
de.plot(figsize=(20, 10))
# Add a title and labels for the axes
plt.title('Departure Percentage vs Year')
plt.xlabel('Month')
plt.ylabel('Departure Percentage (%)')
plt.show()
#From 1901 to 2016
actual_df = df[['jun', 'jul', 'aug', 'sep']]
actual_df
# Calculate the average rainfall for each month
av_df = actual_df.mean()
av_df
#Stacked area chart
#Calculate the cumulative rainfall for each month or season
cumulative_rainfall = df[['jun', 'jul', 'aug', 'sep']].cumsum(axis=1)
# Plot the stacked area chart
plt.figure(figsize=(10, 6))
plt.stackplot(df.index, cumulative_rainfall.values.T, labels=['Jun', 'Jul', 'Aug', 'Sep'])
plt.title('Cumulative Rainfall Over Time')
plt.xlabel('Year')
plt.ylabel('Cumulative Rainfall (mm)')
plt.legend(loc='upper left')
plt.show()
#Scatter plot
# Create a scatter plot with trendline
plt.figure(figsize=(10, 6))
sns.regplot(x=actual_df.index, y='jun', data=actual_df, scatter=True, label='June')
sns.regplot(x=actual_df.index, y='jul', data=actual_df, scatter=True, label='July')
sns.regplot(x=actual_df.index, y='aug', data=actual_df, scatter=True, label='August')
sns.regplot(x=actual_df.index, y='sep', data=actual_df, scatter=True, label='September')
plt.title('Rainfall Variation over Time')
plt.xlabel('Year')
plt.ylabel('Rainfall (mm)')
plt.legend()
plt.show()
# Reset display options to default
pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')
actual_df
# Plotting the predicted values versus the actual values
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted")
plt.show()
```
### OUPUT:
![283140801-3951899a-1c76-47ea-b082-d5f390a4616e](https://github.com/Aravindsamy04/Mini-Project/assets/113497037/e3a64c47-8ec3-4555-b225-36c707c6754c)

![283140838-7b72f3e0-6726-4a55-aad6-7c8e605cfc17](https://github.com/Aravindsamy04/Mini-Project/assets/113497037/9a180363-279d-4fee-aec8-2ff38a29e914)


![283140878-03a9c408-6a16-4ee5-a7a6-df7a7e5a18cc](https://github.com/Aravindsamy04/Mini-Project/assets/113497037/3fc9a5b0-ec90-42f7-8287-eeb43a17b716)


![283140915-65b0bb3a-e18e-4b55-9e8c-8713789b8296](https://github.com/Aravindsamy04/Mini-Project/assets/113497037/46c6e71a-0401-4e07-af67-24555afe50bc)

![283140977-81406b91-12d4-4d0f-8336-237c5ba90bca](https://github.com/Aravindsamy04/Mini-Project/assets/113497037/206aca0f-7432-429c-9f97-89895027d364)


![283141044-8648c3a4-742f-4878-80e8-6bdc571679d2](https://github.com/Aravindsamy04/Mini-Project/assets/113497037/c5dd3176-26f7-4ea0-95af-9ef79ba3320e)


![283141078-a8d4a293-8118-4100-90e2-97a37a63b7ac](https://github.com/Aravindsamy04/Mini-Project/assets/113497037/d113d0fe-89e7-4960-b037-4e57a7ed303f)

![283141117-d40f794e-2069-44ea-a532-d11111d4e069](https://github.com/Aravindsamy04/Mini-Project/assets/113497037/2b63610b-25d6-4889-9141-c12b2b2d3719)


![283141157-993dc66a-5650-4b6a-a012-526d725f5047](https://github.com/Aravindsamy04/Mini-Project/assets/113497037/70906480-5725-4d27-b87d-6c783db33215)




![283141216-293e6529-3814-4e92-a354-19f08e56ea10](https://github.com/Aravindsamy04/Mini-Project/assets/113497037/2bd2b01d-ec04-4d4d-82c2-7eb583bb6a98)
![283141260-7016fc23-3624-412a-9b83-57b087c4604e](https://github.com/Aravindsamy04/Mini-Project/assets/113497037/b758e9a4-c520-47b9-932d-abcb9ae46cde)


![283141295-baa1763e-91d7-4c2d-aafe-7071467a9101](https://github.com/Aravindsamy04/Mini-Project/assets/113497037/98cf6648-2f09-4db9-ac28-3aae803b5619)

![283141343-6b2f5ab5-17e0-4a45-a0ec-7206dc904ba3](https://github.com/Aravindsamy04/Mini-Project/assets/113497037/71830d95-8457-4943-91c9-fd99ff83f750)


![283141381-e38b2f30-5cd7-4bd2-8fcf-a7d0481cdbb5](https://github.com/Aravindsamy04/Mini-Project/assets/113497037/b936c2b5-a419-4f5e-a175-607e6687b6fe)


![283141425-c96bdd63-9d40-42a8-8723-2cf02487172b](https://github.com/Aravindsamy04/Mini-Project/assets/113497037/9b24084a-4d64-4616-98a7-a3c3f6dc8230)


![283141468-7f9eb9e9-6c7e-4904-b3f3-5217abc1d369](https://github.com/Aravindsamy04/Mini-Project/assets/113497037/b1bb79d5-5d0d-40f9-8a11-62d2dc572c6f)

![283141506-e5da59c0-250f-4ed0-85f3-06c67177b4e9](https://github.com/Aravindsamy04/Mini-Project/assets/113497037/58108749-8c52-484d-b38c-9de9d47446c9)



![283141539-595a4ce4-d1d5-4cd9-a6dd-82d646371eab](https://github.com/Aravindsamy04/Mini-Project/assets/113497037/0ea69434-1c3c-4297-ad91-72dc5d65b2ce)

![283141568-a1ae278f-56ff-4f96-88c4-2137659cbe32](https://github.com/Aravindsamy04/Mini-Project/assets/113497037/80847f0a-9762-45a6-bf79-7b4a21db5acb)




![283141623-7c623e35-9a23-4cd2-b54d-9befe42ea958](https://github.com/Aravindsamy04/Mini-Project/assets/113497037/5e651856-cffc-4f96-8fc3-514bf5e06103)



![283141664-c41e54c5-9769-48bb-8b50-0642634f3104](https://github.com/Aravindsamy04/Mini-Project/assets/113497037/9044861a-3c18-4332-95ed-0d981c6bdc60)




![283141698-11b8f249-2db7-484d-93a8-0cc60e3d8601](https://github.com/Aravindsamy04/Mini-Project/assets/113497037/02c158ad-1587-488d-aa14-f585f36b6e0a)





### RESULT:

Thus the rainfall analysis has been executed successfully.













