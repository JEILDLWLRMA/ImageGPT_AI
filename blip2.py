import torch
import os
from PIL import Image
import time
from torchvision import transforms
import pandas as pd
from transformers import BlipProcessor, BlipForQuestionAnswering
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split


class VQADataset(torch.utils.data.Dataset):
    def __init__(self, df, processor, img_path, is_test=False):
        self.df = df
        self.processor = processor
        self.img_path = img_path
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = os.path.join(self.img_path, row["image_id"] + ".jpg")
        image = Image.open(img_name).convert("RGB")
        question = row["question"]

        encoding = self.processor(
            images=image,
            text=question,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        if not self.is_test:
            answer = row["answer"]
            answer = self.processor(
                text=answer,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids
            encoding["labels"] = answer

        for k, v in encoding.items():
            encoding[k] = v.squeeze()

        return encoding


df = pd.read_csv("/data/pauldoun/repos/23tgthon/open/train.csv")
sample_submission = pd.read_csv("/data/pauldoun/repos/23tgthon/open/sample_submission.csv")
img_path = "/data/pauldoun/repos/23tgthon/open/image/train"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"current device is {device}")

processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

print(model)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_dataset = VQADataset(train_df, processor, img_path, is_test=False)
test_dataset = VQADataset(test_df, processor, img_path, is_test=False)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
scaler = GradScaler()  # Create a GradScaler object for mixed precision training

num_epochs = 1
best_loss = float("inf")

model.to("cuda")

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir = '/data/hyeokseung1208/dacon', # model이 저장되는 directory
    logging_dir = '/data/hyeokseung1208/dacon/logs', # log가 저장되는 directory
    num_train_epochs = num_epochs, # training epoch 수
    per_device_train_batch_size=6,  # train batch size
    per_device_eval_batch_size=6,   # eval batch size
    do_train=True,
    do_eval=True,
    gradient_accumulation_steps=64,
    logging_steps = 50, # logging step, batch단위로 학습하기 때문에 epoch수를 곱한 전체 데이터 크기를 batch크기로 나누면 총 step 갯수를 알 수 있다.
    save_steps= 50, # 50 step마다 모델을 저장한다.
    save_total_limit=2, # 2개 모델만 저장한다.
    fp16=True,
    fp16_opt_level="02",
    report_to="none"
)

from datasets import load_metric

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    m1 = load_metric('accuracy')
    m2 = load_metric('f1')

    acc = m1.compute(predictions=preds, references=labels)['accuracy']
    f1 = m2.compute(predictions=preds, references=labels)['f1']

    return {'accuracy':acc, 'f1':f1}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset, # 학습 세트
    eval_dataset=test_dataset, # 테스트 세트
    compute_metrics=compute_metrics # metric 계산 함수
)

trainer.train()

## Huggingface Hub에 모델을 업로드
# Repository 생성 & 모델 업로드
REPO_NAME = "LHS-git-vqa-fine-tuned"
AUTH_TOKEN = "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"

model.push_to_hub(
    REPO_NAME,
    use_temp_dir = True,
    use_auth_token = AUTH_TOKEN,
)

processor.push_to_hub(
    REPO_NAME,
    use_temp_dir = True,
    use_auth_token = AUTH_TOKEN,
)