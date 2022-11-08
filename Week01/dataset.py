import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("iris.data")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

print(x)
print(y)

