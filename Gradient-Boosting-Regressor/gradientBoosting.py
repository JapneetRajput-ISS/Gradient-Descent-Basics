from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import make_scorer, r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# California Housing dataset
data = fetch_california_housing()
X, y = data.data, data.target

model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

r2_scorer = make_scorer(r2_score)

r2_scores = []
for fold, (train_idx, test_idx) in enumerate(kfold.split(X)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    r2_scores.append(r2)

    print(f"Fold {fold + 1}: R² score = {r2:.4f}")

average_r2 = np.mean(r2_scores)
print("\nAverage R² score across all folds:", average_r2)

plt.figure(figsize=(8, 6))
sns.barplot(x=[f'Fold {i+1}' for i in range(5)], y=r2_scores, palette="viridis")
plt.title("R² Scores Across 5 Folds")
plt.xlabel("Fold")
plt.ylabel("R² Score")
plt.ylim(0, 1)
plt.axhline(y=average_r2, color='red', linestyle='--', label=f'Average R² = {average_r2:.4f}')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title(f"Predicted vs Actual Values (Fold 5)")
plt.show()
