import json
import torch
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# 檢查是否有可用的 GPU，如果沒有可用的 GPU，提示用戶並退出
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type != "cuda":
    print("No GPU found. Please ensure a GPU is available and properly configured.")
    exit()

# Step 1: 讀取資料
with open('data/tweets_DM.json', "r") as file:
    tweets_data = [json.loads(line) for line in file]

tweets_df = pd.DataFrame([tweet["_source"]["tweet"] for tweet in tweets_data])

emotion_df = pd.read_csv("data/emotion.csv")
data_identification_df = pd.read_csv("data/data_identification.csv")


# Step 2: 分離測試集
test_ids = data_identification_df[data_identification_df["identification"] == "test"]

# 測試集僅保留推文文本和 tweet_id
test_df = test_ids.merge(tweets_df, on="tweet_id", how="left")
print(f"Test data shape: {test_df.shape}")

# 定義 8 種情感標籤
emotion_labels = ["anger", "anticipation", "disgust", "fear", "sadness", "surprise", "trust", "joy"]

# Step 2: 創建 Hugging Face Dataset
test_dataset = Dataset.from_pandas(test_df[["text"]])

# Step 3: 分詞
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_bertweet")

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=128)

test_dataset = test_dataset.map(preprocess_function, batched=True)

# 設置數據格式
test_dataset = test_dataset.remove_columns(["text"])  # 確保只保留模型需要的欄位
test_dataset.set_format("torch")

# Step 4: 加載模型
model = AutoModelForSequenceClassification.from_pretrained("./fine_tuned_bertweet")

trainer = Trainer(
    model=model,
    tokenizer=tokenizer
)

# 預測測試集
predictions = trainer.predict(test_dataset)
predicted_labels = predictions.predictions.argmax(axis=1)

# 保存測試結果
test_df["predicted_emotion"] = [emotion_labels[label] for label in predicted_labels]
test_df[["tweet_id", "predicted_emotion"]].to_csv("predicted_emotions_v9.csv", index=False)
print("v9儲存")
