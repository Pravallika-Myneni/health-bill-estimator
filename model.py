## for data
import pandas as pd
import numpy as np
## for statistical tests
#import scipy
#import statsmodels.formula.api as smf
#import statsmodels.api as sm
## for machine learning
from sklearn import model_selection, preprocessing, feature_selection, ensemble, linear_model, metrics, decomposition

df= pd.read_csv("https://raw.githubusercontent.com/Pravallika-Myneni/health-bill-estimator/main/insurance.csv")
df.head()

df['sex'] = df['sex'].apply(lambda x: 1 if x=='female' else (0 if x=='male' else None))
df['smoker'] = df['smoker'].apply(lambda x: 1 if x=='yes' else (0 if x=='no' else None))
df['region'] = df['region'].apply(lambda x: 3 if x=='northeast' else ( 2 if x== 'northwest' else (1 if x== 'southeast' else (0 if x=='southwest' else None))))

df_1 = df.copy()
x = df_1.drop(['charges', 'region','sex','children'], axis = 1)
y = df_1['charges']

from sklearn.model_selection import train_test_split
from sklearn import metrics

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
Lin_reg = LinearRegression()
Lin_reg.fit(x, y)
print(Lin_reg.score(x_test, y_test))

user_input = {'age' : [19], 'sex' : ["female"] , 'bmi' : [29] , 'children' : [2] ,'smoker': ["no"],'region': ["northeast"] }
user_input_df = pd.DataFrame(user_input)
copy_df= pd.DataFrame(user_input)

user_input_df['sex'] = user_input_df['sex'].apply(lambda x: 1 if x=='female' else 0)
user_input_df['smoker'] = user_input_df['smoker'].apply(lambda x: 1 if x=='yes' else 0)
user_input_df['region'] = user_input_df['region'].apply(lambda x: 3 if x=='northeast' else ( 2 if x== 'northwest' else (1 if x== 'southeast' else 0)))
user_ip = user_input_df.drop(columns= ['sex','region','children'])

answer_lr = Lin_reg.predict(user_ip)
print(answer_lr)

# Save your model
import joblib
joblib.dump(Lin_reg, 'model.pkl')
print("Model dumped!")

# Load the model that you just saved
lr = joblib.load('model.pkl')

# Saving the data columns from training
model_columns = list(x.columns)
joblib.dump(model_columns, 'model_columns.pkl')
print("Models columns dumped!")