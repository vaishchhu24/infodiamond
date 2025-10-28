import pandas as pd
import pandasql as ps
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from joblib import dump
from imblearn.over_sampling import SMOTE

def q(s): return ps.sqldf(s, globals())

d = pd.read_csv("cleaned_good_bad_customers.csv")
d.drop_duplicates(inplace=True)

for c in d.select_dtypes(exclude=['int64','float64']).columns:
    le = LabelEncoder()
    d[c] = le.fit_transform(d[c].astype(str))

X = d.drop("bad_client_target", axis=1)
y = d["bad_client_target"]

xtr, xte, ytr, yte = train_test_split(X, y, train_size=0.8, random_state=42)

# 👇 scale BEFORE SMOTE
sc = StandardScaler()
xtr_s = sc.fit_transform(xtr)
xte_s = sc.transform(xte)

# 👇 then apply SMOTE
sm = SMOTE(random_state=42)
xtr_bal, ytr_bal = sm.fit_resample(xtr_s, ytr)

print("Before SMOTE:", ytr.value_counts().to_dict())
print("After SMOTE:", ytr_bal.value_counts().to_dict())

clf = KNeighborsClassifier(n_neighbors=7, weights="distance")  # tweaked
clf.fit(xtr_bal, ytr_bal)

yp = clf.predict(xte_s)

acc = accuracy_score(yte, yp)
f1 = f1_score(yte, yp)

print(f"Accuracy: {acc:.3f}")
print(f"F1 Score: {f1:.3f}")

dump(clf, "knn_classifier.joblib")
dump(sc, "scaler.joblib")
print("✅ model + scaler saved")
