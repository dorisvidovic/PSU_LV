import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("occupancy_processed.csv")


X = df[["S3_Temp", "S5_CO2"]]
y = df["Room_Occupancy_Count"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


model = DecisionTreeClassifier(max_depth=3, random_state=42) # 2, 3, 5, none

model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print("Matrica zabune:\n", confusion_matrix(y_test, y_pred))
print("Točnost:", accuracy_score(y_test, y_pred))
print("Izvještaj:\n", classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Matrica zabune")
plt.show()


plt.figure(figsize=(20,10))
plot_tree(
    model,
    feature_names=["S3_Temp", "S5_CO2"],
    class_names=[str(c) for c in model.classes_],
    filled=True
)
plt.title("Stablo odlučivanja")
plt.show()