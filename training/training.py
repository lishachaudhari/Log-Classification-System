import pandas as pd

df = pd.read_csv("dataset/synthetic_logs.csv")

from sentence_transformers import SentenceTransformer

from sklearn.cluster import DBSCAN

#load pre-trained Sentences Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = model.encode(df['log_message'].tolist())

dbscan = DBSCAN(eps=0.2, min_samples=1, metric='euclidean')
clusters = dbscan.fit_predict(embeddings)

df['cluster'] = clusters


cluster_counts = df['cluster'].value_counts()
large_clusters = cluster_counts[cluster_counts > 10].index

for cluster in large_clusters:
    print(f"Cluster {cluster}:")
    print(df[df['cluster'] == cluster]['log_message'].head(5).to_string(index=False))
    print()

n_cluster = df['cluster'].unique()

import re
def classify_with_regex(log_message):
    regex_patterns = {
        r"User User\d+ logged (in|out).": "User Action",
        r"Backup (started|ended) at .*": "System Notification",
        r"Backup completed successfully.": "System Notification",
        r"System updated to version .*": "System Notification",
        r"File .* uploaded successfully by user .*": "System Notification",
        r"Disk cleanup completed successfully.": "System Notification",
        r"System reboot initiated by user .*": "System Notification",
        r"Account with ID .* created by .*": "User Action"
    }
    for pattern, label in regex_patterns.items():
        if re.search(pattern, log_message, re.IGNORECASE):
            return label
    return None

df['regex_label'] = df['log_message'].apply(classify_with_regex)
regex_labeled = df[df['regex_label'] == "User Action"]
regex_labeled.shape

df_non_regex = df[df['regex_label'].isnull()].copy()
df_non_regex.shape

df_non_legacy = df_non_regex[df_non_regex.source != 'LegacyCRM']
df_non_legacy.source.unique()

filtered_embeddings = model.encode(df_non_legacy['log_message'].tolist())

X = filtered_embeddings
y = df_non_legacy['target_label'].values

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
report = classification_report(y_test, y_pred)
print(report)

import joblib

joblib.dump(clf, 'log_classifier.joblib')