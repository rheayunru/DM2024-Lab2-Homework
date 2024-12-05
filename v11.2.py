import pandas as pd
import torch
import json
from transformers import AutoTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# 檢查GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type != "cuda":
    print("No GPU found. Please ensure a GPU is available and properly configured.")
    exit()

# 定義評估指標
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}

# Step 1: 讀取資料
with open('data/tweets_DM.json', "r") as file:
    tweets_data = [json.loads(line) for line in file]
tweets_df = pd.DataFrame([tweet["_source"]["tweet"] for tweet in tweets_data])
emotion_df = pd.read_csv("data/emotion.csv")
data_identification_df = pd.read_csv("data/data_identification.csv")

# Step 2: 分離訓練集和測試集
train_ids = data_identification_df[data_identification_df["identification"] == "train"]
train_df = train_ids.merge(tweets_df, on="tweet_id").merge(emotion_df, on="tweet_id", how="left")
train_df = train_df.dropna(subset=["emotion"])
emotion_labels = ["anger", "anticipation", "disgust", "fear", "sadness", "surprise", "trust", "joy"]
train_df["label"] = train_df["emotion"].apply(lambda x: emotion_labels.index(x))

# 劃分訓練和驗證集
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df["text"].tolist(),
    train_df["label"].tolist(),
    test_size=0.2,  # 劃分 20% 為驗證集
    random_state=42
)

train_dataset = Dataset.from_dict({
    "text": train_texts,
    "label": train_labels,
})
val_dataset = Dataset.from_dict({
    "text": val_texts,
    "label": val_labels,
})

# Hugging Face Dataset
# train_dataset = Dataset.from_pandas(train_df[["text", "label"]])

# 加載模型和分詞器
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_twitter_roberta")
model = RobertaForSequenceClassification.from_pretrained("./fine_tuned_twitter_roberta")

# 顯式設置 problem_type 為單標籤分類
model.config.problem_type = "single_label_classification"

# 分詞
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)

train_dataset = train_dataset.map(preprocess_function, batched=True)
val_dataset = val_dataset.map(preprocess_function, batched=True)

# 確保 labels 為整數
train_dataset = train_dataset.map(lambda x: {"labels": int(x["label"])})  
val_dataset = val_dataset.map(lambda x: {"labels": int(x["label"])})

# 設置數據格式
train_dataset = train_dataset.remove_columns(["text"])
val_dataset = val_dataset.remove_columns(["text"])

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


# 訓練參數
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch", # 每個 epoch 評估一次驗證集
    save_strategy="epoch",  # 每個 epoch 保存一次
    load_best_model_at_end=True, # 加載最佳模型
    learning_rate=1.5e-5, # 微調學習率
    per_device_train_batch_size=32, # 增加批量大小
    per_device_eval_batch_size=32, # 驗證批量大小
    num_train_epochs=5, # 繼續訓練的總 epoch 數
    weight_decay=0.01,
    save_total_limit=1,
    logging_dir="./logs",
    metric_for_best_model="accuracy",
    greater_is_better=True,
    fp16=True,                             # 如果硬件支持混合精度訓練
    # gradient_accumulation_steps=2,         # 如果內存不足，啟用梯度累積      
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset, # 更新後的訓練集
    eval_dataset=val_dataset,    # 新的驗證集
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 添加早停回調
trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=2)) # 耐心等待 2 個 epoch

# 接續訓練
trainer.train(resume_from_checkpoint=True)

# Step 8: 保存模型
model.save_pretrained("./fine_tuned_twitter_roberta_v11.2")
tokenizer.save_pretrained("./fine_tuned_twitter_roberta_v11.2")
print("訓練完成, 保存至./fine_tuned_twitter_roberta_v11.2")