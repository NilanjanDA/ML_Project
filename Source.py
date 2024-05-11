import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = sns.load_dataset("mpg")
df.head()
df.isnull().sum()
df.dropna(inplace=True)
X = df[['displacement', 'horsepower', 'weight', 'acceleration']]
Y = df.mpg
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.15, random_state = 42)
X_train
Y_train
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,Y_train)
# Model accuracy
model.score(X_test,Y_test)
# Saving the model
import pickle
filename = 'mpg_regression.sav'
pickle.dump(model, open(filename, 'wb'))