# Log-Classification-System
This project implements a hybrid log classification system that categorizes system, application, and CRM logs using Regex, BERT embeddings, and LLMs. It classifies logs into categories like user actions, workflow errors, system notifications, and deprecation warnings to support monitoring, alerting, and analytics.

#Features
1. Regex-based Classification: Handles common and predictable log patterns efficiently.
2. BERT-based Classification: Uses SentenceTransformers embeddings with a LogisticRegression model to classify logs that cannot be handled by regex.
3. LLM-based Classification: Uses Groq LLM API for legacy systems or complex log messages that require semantic understanding.
4. Hybrid Approach: Combines regex, BERT, and LLM to ensure high accuracy and coverage.
5. Clustering for Pattern Discovery: Uses DBSCAN on embeddings to identify frequently occurring log patterns.
6. CSV Integration: Accepts log data in CSV format and outputs classified results.
