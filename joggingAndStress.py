import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. data
data = {
    "runs_per_week": [0, 1, 2, 3, 4, 5, 2, 6, 3, 0,
                      4, 5, 1, 6, 7, 2, 3, 4, 0, 5],
    "stress_events": [8, 7, 6, 4, 2, 1, 7, 3, 4, 9,
                      2, 1, 8, 3, 2, 6, 4, 3, 8, 2],
    "label": [
        "нервный", "нервный", "нервный", "спокойный", "спокойный",
        "спокойный", "нервный", "спокойный", "спокойный", "нервный",
        "спокойный", "спокойный", "нервный", "спокойный", "спокойный",
        "нервный", "спокойный", "спокойный", "нервный", "спокойный"
    ]
}

df = pd.DataFrame(data)
print("Таблица данных:")
print(df)

# 2. Scatter plot
plt.figure(figsize=(8, 6))
for label, color in zip(["спокойный", "нервный"], ["green", "red"]):
    subset = df[df["label"] == label]
    plt.scatter(subset["runs_per_week"], subset["stress_events"], label=label, c=color, alpha=0.7)

plt.xlabel("Пробежки в неделю")
plt.ylabel("Стрессовые ситуации")
plt.title("Scatter plot: бег & стресс")
plt.legend()
plt.grid(True)
plt.show()

# 3. Обучаем Decision Tree
X = df[["runs_per_week", "stress_events"]]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

tree_clf = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_clf.fit(X_train, y_train)

# 4. Визуал дерево
plt.figure(figsize=(12, 8))
plot_tree(tree_clf, feature_names=X.columns, class_names=tree_clf.classes_, filled=True, fontsize=10)
plt.title("Decision tree")
plt.show()

# 5. Feature importance
importances = pd.Series(tree_clf.feature_importances_, index=X.columns)
print("\n Feature importance:")
print(importances)

# 6. Сравнение с KNN
knn_clf = KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(X_train, y_train)

y_pred_tree = tree_clf.predict(X_test)
y_pred_knn = knn_clf.predict(X_test)

acc_tree = accuracy_score(y_test, y_pred_tree)
acc_knn = accuracy_score(y_test, y_pred_knn)

print("\n Сравнение моделей:")
print(f"Decision Tree Accuracy: {acc_tree:.2f}")
print(f"KNN Accuracy: {acc_knn:.2f}")

print("\n Classification Report (Decision Tree):")
print(classification_report(y_test, y_pred_tree, target_names=tree_clf.classes_))

print("\n Classification Report (KNN):")
print(classification_report(y_test, y_pred_knn, target_names=tree_clf.classes_))
