import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
data = pd.read_csv(r"C:\Users\User\OneDrive - KMITL\Documents\creditcard.csv")

dataf=((data-data.min())/(data.max()-data.min()))

# print(dataf)
dataf.to_csv(r"C:\Users\User\OneDrive - KMITL\Documents\credit_normalize.csv")