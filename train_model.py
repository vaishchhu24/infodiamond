import pandas as pd
import pandasql as ps
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from joblib import dump

def q(s):
    return ps.sqldf(s, globals())

# Load dataset
d = pd.read_csv("cleaned_good_bad_customers.csv")

# Remove duplicates
b = d.shape[0]
d.drop_duplicates(inplace=True)
a = d.shape[0]
print(f"removed {b - a} dups, new shape {d.shape}")

# Show basic info
print("\ncols:", list(d.columns))
print("\npreview:\n", d.head())

# Identify numeric and categorical columns
num = d.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat = d.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

print("\nnums:", num)
print("\ncats:", cat)

# Encode categorical columns
for c in cat:
    le = LabelEncoder()
    d[c] = le.fit_transform(d[c].astype(str))

# Split features and target
X = d.drop("bad_client_target", axis=1)
y = d["bad_client_target"]

# Train-test split
xtr, xte, ytr, yte = train_test_split(X, y, train_size=0.8, random_state=42)

# Scale numeric data
sc = StandardScaler()
xtr = sc.fit_transform(xtr)
xte = sc.transform(xte)

# Train model
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(xtr, ytr)

# Predict
yp = clf.predict(xte)

# Evaluate
acc = accuracy_score(yte, yp)
f1 = f1_score(yte, yp)

print(f"\nacc: {acc:.3f}")
print(f"f1: {f1:.3f}")

# Save model and scaler
dump(clf, "knn_classifier.joblib")
dump(sc, "scaler.joblib")

