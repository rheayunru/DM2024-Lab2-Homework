import pandas as pd
import torch
import json
from transformers import AutoTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset


# 讀取 JSON 格式的推文數據，轉換為 DataFrame 格式
with open('data/tweets_DM.json', "r") as file:
    tweets_data = [json.loads(line) for line in file]
tweets_df = pd.DataFrame([tweet["_source"]["tweet"] for tweet in tweets_data])

# 讀取情感標籤和數據集劃分文件
emotion_df = pd.read_csv("data/emotion.csv")
data_identification_df = pd.read_csv("data/data_identification.csv")

# 分離訓練集和測試集
train_ids = data_identification_df[data_identification_df["identification"] == "train"]
test_ids = data_identification_df[data_identification_df["identification"] == "test"]

# 合併數據集，過濾空值
train_df = train_ids.merge(tweets_df, on="tweet_id").merge(emotion_df, on="tweet_id", how="left")
test_df = test_ids.merge(tweets_df, on="tweet_id", how="left")
train_df = train_df.dropna(subset=["emotion"])

# 定義情感標籤並將其轉為數字
emotion_labels = ["anger", "anticipation", "disgust", "fear", "sadness", "surprise", "trust", "joy"]
train_df["label"] = train_df["emotion"].apply(lambda x: emotion_labels.index(x))

# 創建 Hugging Face Dataset，僅保留文本和標籤
train_dataset = Dataset.from_pandas(train_df[["text", "label"]])
test_dataset = Dataset.from_pandas(test_df[["text"]])


# 加載預訓練模型和分詞器
MODEL_NAME = "cardiffnlp/twitter-roberta-large-emotion-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = RobertaForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(emotion_labels),
    ignore_mismatched_sizes=True  # 忽略維度不匹配的情況
)

# 顯式設置問題類型為單標籤分類
model.config.problem_type = "single_label_classification"
print(f"Number of labels: {model.config.num_labels}")


# 定義分詞函數，確保長度和填充
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

# 應用分詞至數據集
train_dataset = train_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

# 確保標籤為整數類型
train_dataset = train_dataset.map(lambda x: {"labels": int(x["label"])})  

# 設置數據格式供模型使用
train_dataset = train_dataset.remove_columns(["text"])
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

test_dataset = test_dataset.remove_columns(["text"])  # 僅保留必要的欄位
print(test_dataset.column_names)
test_dataset.set_format("torch")


# 訓練參數設置
training_args = TrainingArguments(
    output_dir="./results",   # 輸出目錄
    eval_strategy="no",  # 不進行評估
    learning_rate=2e-5,  # 設置學習率
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,  # 訓練週期數
    weight_decay=0.01,  # 權重衰減
    save_total_limit=1,  # 最多保存1個模型
    logging_dir="./logs",
    metric_for_best_model="accuracy",  # 評估的最佳指標
)

# 創建訓練器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

# 訓練模型
trainer.train()


# 預測測試集
predictions = trainer.predict(test_dataset)
predicted_labels = predictions.predictions.argmax(axis=1)

# 保存測試結果至 CSV 文件
test_df["predicted_emotion"] = [emotion_labels[label] for label in predicted_labels]
test_df[["tweet_id", "predicted_emotion"]].to_csv("predicted_emotions_v11.csv", index=False)
print("v11儲存✧*｡٩(ˊᗜˋ*)و✧*｡")