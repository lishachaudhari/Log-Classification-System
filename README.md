## Log-Classification-System
This project implements a hybrid log classification system that categorizes system, application, and CRM logs using Regex, BERT embeddings, and LLMs. It classifies logs into categories like user actions, workflow errors, system notifications, and deprecation warnings to support monitoring, alerting, and analytics.

# Features
1. Regex-based Classification: Handles common and predictable log patterns efficiently.
2. BERT-based Classification: Uses SentenceTransformers embeddings with a LogisticRegression model to classify logs that cannot be handled by regex.
3. LLM-based Classification: Uses Groq LLM API for legacy systems or complex log messages that require semantic understanding.
4. Hybrid Approach: Combines regex, BERT, and LLM to ensure high accuracy and coverage.
5. Clustering for Pattern Discovery: Uses DBSCAN on embeddings to identify frequently occurring log patterns.
6. CSV Integration: Accepts log data in CSV format and outputs classified results.
![Alt text](<img width="1024" height="1536" alt="fa4b30b8-7b5b-4013-8a10-1272e4df0e7a" src="https://github.com/user-attachments/assets/a002d858-3fb6-46d8-a414-3e3ca5bca571" />
)

# Project Structure
├── dataset/
│   └── synthetic_logs.csv        # Sample log dataset
├── models/
│   └── log_classifier.joblib     # Pre-trained BERT classifier
├── processor_regex.py            # Regex-based classification
├── processor_bert.py             # BERT-based classification
├── processor_llm.py              # LLM-based classification
├── classify.py                   # Main classification script
├── training/
│   ├── training.py               # Training & clustering script
│   └── dataset/
│       └── synthetic_logs.csv    # Dataset for training
└── README.md
