import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv('apple_quality.csv')
df_clean = df.dropna(subset=['Quality'])

X = df_clean.drop(['A_id', 'Quality'], axis=1)
y = df_clean['Quality']

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accuracy_list = []

for k in range(1, 6):
    print(f"\n=== Evaluasi KNN dengan k = {k} ===")
    pipeline = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))

    # Cross-validated predictions (bukan akurasi langsung)
    y_pred = cross_val_predict(pipeline, X, y, cv=cv)

    # Akurasi
    acc = accuracy_score(y, y_pred)
    accuracy_list.append(acc)
    print(f"Akurasi: {acc:.2%}")

    # Classification report: precision, recall, F1
    print("Classification Report:")
    print(classification_report(y, y_pred))

plt.figure(figsize=(8, 5))
sns.barplot(x=[f'K={i}' for i in range(1, 6)], y=accuracy_list)
plt.title("Akurasi Model KNN untuk K = 1 hingga 5")
plt.ylabel("Akurasi")
plt.ylim(0.8, 1.0)
plt.show()
