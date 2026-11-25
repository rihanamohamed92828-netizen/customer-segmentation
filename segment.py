"""Customer segmentation with K-Means on the Mall Customers dataset."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

SEED = 42
df = pd.read_csv("data/Mall_Customers.csv").rename(columns={
    "Annual Income (k$)": "Income", "Spending Score (1-100)": "Score"})
print(df.shape, "missing:", int(df.isna().sum().sum()))
print(df.describe().round(1).to_string())

# EDA
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
for a, c in zip(ax, ["Age", "Income", "Score"]):
    sns.histplot(df[c], kde=True, ax=a, color="#4c72b0")
    a.set_title(c)
plt.tight_layout(); plt.savefig("results/distributions.png", dpi=150); plt.close()

# Choose k on Income and Spending Score
X = StandardScaler().fit_transform(df[["Income", "Score"]])
ks = range(2, 11)
inertia, sil = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
    inertia.append(km.inertia_); sil.append(silhouette_score(X, km.labels_))
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].plot(ks, inertia, "o-"); ax[0].set_title("Elbow method"); ax[0].set_xlabel("k"); ax[0].set_ylabel("Inertia")
ax[1].plot(ks, sil, "o-", color="#c44e52"); ax[1].set_title("Silhouette score"); ax[1].set_xlabel("k")
plt.tight_layout(); plt.savefig("results/choose_k.png", dpi=150); plt.close()
best_k = 5
print("silhouette by k:", {k: round(s, 3) for k, s in zip(ks, sil)})

km = KMeans(n_clusters=best_k, n_init=10, random_state=SEED).fit(X)
df["Cluster"] = km.labels_
print(f"k={best_k} silhouette={silhouette_score(X, km.labels_):.3f}")

profile = df.groupby("Cluster").agg(
    Customers=("CustomerID", "count"), Avg_Age=("Age", "mean"),
    Avg_Income=("Income", "mean"), Avg_Score=("Score", "mean")).round(1)

def name(r):
    hi_inc, hi_sc = r.Avg_Income > 60, r.Avg_Score > 50
    return {(True, True): "High income, high spend (target)",
            (True, False): "High income, low spend (careful)",
            (False, True): "Low income, high spend (impulsive)",
            (False, False): "Low income, low spend (sensible)"}.get((hi_inc, hi_sc), "Average")
profile["Segment"] = profile.apply(lambda r: name(r) if abs(r.Avg_Income - 55) > 15 or abs(r.Avg_Score - 50) > 15 else "Average (middle)", axis=1)
print(profile.to_string())
profile.to_csv("results/cluster_profiles.csv")
df.to_csv("results/customers_with_segments.csv", index=False)

plt.figure(figsize=(7.5, 5.5))
sns.scatterplot(data=df, x="Income", y="Score", hue="Cluster", palette="deep", s=60)
cent = StandardScaler().fit(df[["Income", "Score"]]).inverse_transform(km.cluster_centers_)
plt.scatter(cent[:, 0], cent[:, 1], c="black", marker="X", s=200, label="Centroids")
plt.xlabel("Annual income (k$)"); plt.ylabel("Spending score"); plt.title("Customer segments (K-Means, k=5)")
plt.legend(); plt.tight_layout(); plt.savefig("results/segments.png", dpi=150); plt.close()

# Simple dashboard
fig, ax = plt.subplots(2, 2, figsize=(12, 9))
sns.countplot(data=df, x="Cluster", hue="Cluster", palette="deep", legend=False, ax=ax[0, 0]); ax[0, 0].set_title("Customers per segment")
sns.boxplot(data=df, x="Cluster", y="Age", hue="Cluster", palette="deep", legend=False, ax=ax[0, 1]); ax[0, 1].set_title("Age by segment")
sns.boxplot(data=df, x="Cluster", y="Score", hue="Cluster", palette="deep", legend=False, ax=ax[1, 0]); ax[1, 0].set_title("Spending score by segment")
sns.countplot(data=df, x="Cluster", hue=df.columns[1], ax=ax[1, 1]); ax[1, 1].set_title("Gender mix by segment")
plt.suptitle("Customer segmentation dashboard", fontsize=14); plt.tight_layout()
plt.savefig("results/dashboard.png", dpi=150)
