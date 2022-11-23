import pandas as pd
import numpy as np

# Read the dataset and extract inputs and output
dataset = pd.read_csv("dataset.csv")
inputs = dataset.iloc[:, :-1].values
output = dataset.iloc[:, -1].values

# Remove NaN values by applying 'mean' strategy
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
inputs = imputer.fit_transform(inputs[:, :])

# Labelling the output data in the form of 0s and 1s
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
output = label_encoder.fit_transform(output)

# Split the data into training and testing
from sklearn.model_selection import train_test_split

# We put test_size as 0.2 to divide 20% of our data for testing
inputs_train, inputs_test, \
output_train, output_test = train_test_split(inputs, output, test_size=0.2, random_state=0)

# Standardize the data
from sklearn.preprocessing import StandardScaler
standard_scaler_inputs = StandardScaler()
inputs_train = standard_scaler_inputs.fit_transform(inputs_train)
inputs_test = standard_scaler_inputs.fit_transform(inputs_test)
