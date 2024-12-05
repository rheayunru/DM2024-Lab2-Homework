import pandas as pd
import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}

# 檢查GPU
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

# Step 2: 分離訓練集和測試集
train_ids = data_identification_df[data_identification_df["identification"] == "train"]
test_ids = data_identification_df[data_identification_df["identification"] == "test"]

train_df = train_ids.merge(tweets_df, on="tweet_id").merge(emotion_df, on="tweet_id", how="left")
test_df = test_ids.merge(tweets_df, on="tweet_id", how="left")

train_df = train_df.dropna(subset=["emotion"])
emotion_labels = ["anger", "anticipation", "disgust", "fear", "sadness", "surprise", "trust", "joy"]
train_df["label"] = train_df["emotion"].apply(lambda x: emotion_labels.index(x))

# Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df[["text", "label"]])
test_dataset = Dataset.from_pandas(test_df[["text"]])

# 加載模型和分詞器
MODEL_NAME = "vinai/bertweet-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(emotion_labels))

# 分詞
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=128)

train_dataset = train_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.add_column("label", [0] * len(test_dataset))

# 設置數據格式
train_dataset = train_dataset.remove_columns(["text"])
test_dataset = test_dataset.remove_columns(["text"])
train_dataset.set_format("torch")
test_dataset.set_format("torch")

# 訓練參數
training_args = TrainingArguments(
    output_dir="./results_bertweet",
    eval_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=1,
    logging_dir="./logs",
    metric_for_best_model="accuracy",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 訓練
trainer.train()

# Step 8: 保存模型
model.save_pretrained("./fine_tuned_bertweet")
tokenizer.save_pretrained("./fine_tuned_bertweet")
print("模型訓練完成，已保存至 ./fine_tuned_bertweet")

# 測試推斷
predictions = trainer.predict(test_dataset)
predicted_labels = predictions.predictions.argmax(axis=1)

test_df["predicted_emotion"] = [emotion_labels[label] for label in predicted_labels]
test_df[["tweet_id", "predicted_emotion"]].to_csv("predicted_emotions_v9.csv", index=False)
print("v9已保存。")
