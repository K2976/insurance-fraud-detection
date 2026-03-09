import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

# load dataset
data = pd.read_csv("dataset/insurance_claims.csv")

# select useful columns
data = data[['age','months_as_customer','total_claim_amount','incident_severity','fraud_reported']]

# convert target column
data['fraud_reported'] = data['fraud_reported'].map({'Y':1,'N':0})

# encode categorical column
le = LabelEncoder()
data['incident_severity'] = le.fit_transform(data['incident_severity'])

# features
X = data.drop("fraud_reported",axis=1)

# target
y = data['fraud_reported']

# split dataset
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# model
model = RandomForestClassifier()

model.fit(X_train,y_train)

# save model
pickle.dump(model,open("model.pkl","wb"))

print("Model trained successfully")