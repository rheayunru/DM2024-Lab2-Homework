import tensorflow as tf
import pandas as pd
import json
import re
import numpy as np
import nltk
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU devices found: {gpus}")
else:
    print("No GPU devices found.")

# Load tweets_DM.json
with open('data/tweets_DM.json', "r") as file:
    tweets_data = [json.loads(line) for line in file]

tweets_df = pd.DataFrame([tweet["_source"]["tweet"] for tweet in tweets_data])

# Load emotion.csv
emotion_df = pd.read_csv('data/emotion.csv')

# Load data_identification.csv
data_id_df = pd.read_csv('data/data_identification.csv')

# Merge tweets with data identification
merged_df = pd.merge(tweets_df, data_id_df, on="tweet_id", how="inner")

# Merge with emotion labels (only training set will have labels)
merged_df = pd.merge(merged_df, emotion_df, on="tweet_id", how="left")


# Split data into training and testing
train_df = merged_df[merged_df["identification"] == "train"]
test_df = merged_df[merged_df["identification"] == "test"]



# Step 1: Text Preprocessing
# Clean text by removing special characters and punctuation
train_df["cleaned_text"] = train_df["text"].str.lower().apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))
test_df["cleaned_text"] = test_df["text"].str.lower().apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))

# Extract features and labels from the training dataset
X = train_df["cleaned_text"]
y = train_df["emotion"]

# Step 2: Stratified Splitting
# Ensure similar class distribution in training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Step 3: Compute Class Weights
# Calculate class weights to handle imbalanced classes
class_weights = compute_class_weight(
    class_weight="balanced",  # Auto-balance weights based on class frequencies
    classes=np.unique(y_train),  # Unique class labels
    y=y_train  # Use training labels
)
class_weights_dict = dict(enumerate(class_weights))
print("Class Weights:", class_weights_dict)

# Step 4: Define the Pipeline
# Use TfidfVectorizer for text vectorization
# Define the pipeline
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(stop_words="english", max_features=5000, tokenizer=nltk.word_tokenize))
])

# Fit the pipeline on the training data
pipeline.fit(X_train)

# Transform training, validation, and testing data
X_train_tfidf = pipeline.transform(X_train)
X_val_tfidf = pipeline.transform(X_val)
X_test_tfidf = pipeline.transform(test_df["cleaned_text"])


# Step 1: 將類別標籤編碼為數字
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)  # 訓練數據編碼
y_val_encoded = label_encoder.transform(y_val)  # 驗證數據編碼

# Step 2: 定義模型，使用自動類別權重
model = RandomForestClassifier(
    random_state=42,
    class_weight="balanced_subsample",  # 自動平衡權重
    n_estimators=50,  # 減少樹數量加速運行
    max_depth=10,  # 限制深度
    n_jobs=-1  # 多核心並行運行
)

# Step 3: 模型訓練
model.fit(X_train_tfidf, y_train_encoded)

# Step 4: 驗證集預測並轉回文字標籤
y_val_pred = model.predict(X_val_tfidf)
y_val_pred_decoded = label_encoder.inverse_transform(y_val_pred)
print("Validation Performance:\n", classification_report(y_val, y_val_pred_decoded))

# Step 5: 測試集預測
y_test_pred = model.predict(X_test_tfidf)

# Step 6: 將數字標籤轉回文字標籤
y_test_pred_decoded = label_encoder.inverse_transform(y_test_pred)

# Step 7: 保存預測結果至 CSV
# 假設 test_df 包含 "tweet_id"
test_df["predicted_emotion"] = y_test_pred_decoded  # 確保保存文字標籤

# 保存為 CSV 文件，指定列名
output_file = "predicted_emotions_v2.csv"
test_df[["tweet_id", "predicted_emotion"]].rename(
    columns={"tweet_id": "id", "predicted_emotion": "emotion"}
).to_csv(output_file, index=False)

print("v2儲存")
