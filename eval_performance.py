import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv("exp\performance.csv")
#plt.figure()
data.plot()
plt.show()