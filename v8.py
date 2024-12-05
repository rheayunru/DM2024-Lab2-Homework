import pandas as pd
import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# Step 1: 讀取資料
with open('data/tweets_DM.json', "r") as file:
    tweets_data = [json.loads(line) for line in file]

tweets_df = pd.DataFrame([tweet["_source"]["tweet"] for tweet in tweets_data])

emotion_df = pd.read_csv("emotion.csv")
data_identification_df = pd.read_csv("data_identification.csv")

# 合併數據
merged_df = data_identification_df.merge(tweets_df, on="tweet_id").merge(emotion_df, on="tweet_id", how="left")

# 分離訓練集和測試集
train_df = merged_df[merged_df["identification"] == "train"]
test_df = merged_df[merged_df["identification"] == "test"]

train_texts = train_df["text"].astype(str).tolist()
train_labels = train_df["emotion"].tolist()
test_texts = test_df["text"].astype(str).tolist()

# Step 2: 使用 Hugging Face 的 Twitter 專用模型
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Step 3: 預處理推文
def preprocess_tweets(texts):
    """
    預處理推文文本，包括：
    - 移除用戶名（@username）
    - 移除網址（http/https 開頭的文字）
    """
    processed_texts = []
    for text in texts:
        text = text.replace("@", "").replace("#", "").replace("http", "").replace("https", "")
        processed_texts.append(text)
    return processed_texts

train_texts = preprocess_tweets(train_texts)
test_texts = preprocess_tweets(test_texts)

# Step 4: 預測函數
def predict_sentiments(texts):
    """
    使用 Twitter RoBERTa 模型對文本進行情感預測
    """
    predictions = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        outputs = model(**inputs)
        probs = softmax(outputs.logits.detach().numpy(), axis=1)  # 轉為概率分布
        label = torch.argmax(torch.tensor(probs), axis=1).item()
        predictions.append(label)
    return predictions

# Step 5: 預測測試集情感
predicted_labels = predict_sentiments(test_texts)

# 將預測結果映射為情感標籤
labels = ["negative", "neutral", "positive"]
test_df["predicted_emotion"] = [labels[label] for label in predicted_labels]

# Step 6: 保存結果
test_df[["tweet_id", "predicted_emotion"]].to_csv("predicted_emotions_v8.csv", index=False)
print("v8儲存")
