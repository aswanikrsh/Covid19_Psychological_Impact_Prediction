import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load the csv file
df = pd.read_csv("covid_preprocesseddata.csv")
print(df.columns)

print(df.head())

# Select independent and dependent variable
X = df[['time_bp', 'time_dp', 'travel_time', 'easeof_online', 'home_env',
       'prod_inc', 'sleep_bal', 'new_skill', 'fam_connect', 'relaxed',
       'self_time', 'like_hw', 'dislike_hw', 'time_bp.1', 'age_18.0',
       'age_22.0', 'age_29.0', 'age_36.5', 'age_45.0', 'age_55.0', 'age_65.0',
       'gender_Female', 'gender_Male', 'gender_Prefer not to say',
       'occupation_Currently Out of Work', 'occupation_Entrepreneur',
       'occupation_Homemaker',
       'occupation_Medical Professional aiding efforts against COVID-19',
       'occupation_Retired/Senior Citizen', 'occupation_Student in College',
       'occupation_Student in School', 'occupation_Working Professional',
       'prefer_Complete Physical Attendance', 'prefer_Work/study from home',
       'certaindays_hw_Maybe', 'certaindays_hw_No', 'certaindays_hw_Yes']]
y = df["lifestyle_change"]

# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test= sc.transform(X_test)

# Instantiate the model
classifier = RandomForestClassifier()

# Fit the model
model=classifier.fit(X_train, y_train)
y_pred=model.predict(X_test)

accuracy_dt = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy_dt * 100)
print("\nClassification Report: \n", classification_report(y_test, y_pred))

# Make pickle file of our model
pickle.dump(classifier, open("model.pkl", "wb"))

