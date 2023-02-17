import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = r'D:\美赛\比赛ing\Code\Problem_C_Data_Wordle.xlsx'
Ori_data = pd.read_excel(file_path)

# Change the column name
Column_name = Ori_data.iloc[0,:].values
Ori_data.columns = Column_name

# Formalise the data
Data = Ori_data.drop(axis=0,index=0)
Data = Data.iloc[:,1:]

# Standardize
mean = Data.iloc[:,3].mean()
std = Data.iloc[:,3].std()

# Get the final_data
middle_data = (Data.iloc[:,3].values).tolist()
final_data = list(reversed((middle_data - mean)*100/std))
# print(final_data)
# plot the data
plt.plot(final_data)
plt.title('Number with Date')
plt.show()