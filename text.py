import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import LlamaTokenizer, LlamaForSequenceClassification, AdamW,AutoTokenizer,LlamaModel
from peft import LoraConfig, get_peft_model


class DepressionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        # 调试信息：打印 encoding 的形状
       
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
class LlamaClassifier(torch.nn.Module):
    def __init__(self, model, num_labels):
        super(LlamaClassifier, self).__init__()
        self.llama = model
        self.classifier = torch.nn.Linear(model.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.llama(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # 取cls token的输出
        logits = self.classifier(pooled_output)
        return logits
    
# 加载数据
def load_data(data_dir, label_file, tokenizer, max_len):
    texts = []
    labels = []
    label_df = pd.read_excel(label_file)
    label_map = {row['Participant_ID']: row['Gender'] for _, row in label_df.iterrows()}
    for file_name in os.listdir(data_dir):
        file_prefix = file_name.split('_')[0]
        file_path = os.path.join(data_dir, file_name)
        if int(file_prefix) in label_map:
            df = pd.read_csv(file_path)
            merged_text = ' '.join(df['value'].astype(str).str.replace('"', '').tolist())  # 将文本合并成一句话
            texts.append(merged_text)
            labels.append(label_map[int(file_prefix)])
    dataset = DepressionDataset(texts, labels, tokenizer, max_len)
    # train_size = int(0.9 * len(dataset))
    # test_size = len(dataset) - train_size
    # train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    # train_dataset=dataset
    # test_dataset=dataset
    # return train_dataset, test_dataset
    return dataset

# 加载数据集
data_dir = 'Z:\\2024program\\text_model\\textdate'
label_file = 'Z:\\2024program\\Audio_train_code_hyf_new\\data\\labels.xlsx'
tokenizer_path = "Z:\\2024program\\llama-3.2-1B-spinquant-hf"
print(os.listdir(tokenizer_path))
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
tokenizer.pad_token = tokenizer.eos_token

train_dataset= load_data(data_dir, label_file, tokenizer, max_len=512)
test_dataset =load_data('Z:\\2024program\\test', label_file, tokenizer, max_len=512)

def manual_collate_fn(batch):
    max_length = max(len(item['input_ids']) for item in batch)
    input_ids = torch.stack([torch.cat([item['input_ids'], torch.full((max_length - len(item['input_ids']),), tokenizer.pad_token_id)]) for item in batch])
    attention_mask = torch.stack([torch.cat([item['attention_mask'], torch.zeros(max_length - len(item['attention_mask']))]) for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

train_loader = DataLoader(train_dataset, batch_size=8,  shuffle=False,collate_fn=manual_collate_fn)

test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False,collate_fn=manual_collate_fn)

# 加载模型并应用 LoRA
# 加载LlamaModel并添加分类头
llama_model = LlamaModel.from_pretrained(tokenizer_path)
model = LlamaClassifier(llama_model, num_labels=2)
lora_config = LoraConfig(
    r=8,  # 低秩矩阵的秩
    lora_alpha=128,  # LoRA 缩放因子
    target_modules=["q_proj", "v_proj"],  # 应用 LoRA 的目标模块
    lora_dropout=0.1,  # LoRA 的 dropout 概率
    bias="none",  # 是否应用偏置
)
model = get_peft_model(model, lora_config)
model.to('cuda')

# 优化器和损失函数
optimizer = AdamW(model.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss()


num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0  
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to('cuda')
        attention_mask = batch['attention_mask'].to('cuda')
        labels = batch['labels'].to('cuda')
        logits = model(input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()  

    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}")


# 测试循环
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for batch in test_loader:
            input_ids = batch['input_ids'].to('cuda')
            attention_mask = batch['attention_mask'].to('cuda')
            labels = batch['labels'].to('cuda')
                
            logits = model(input_ids, attention_mask=attention_mask)  # 直接赋值给 logits
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f"Test Accuracy: {100 * correct / total}%")

# 保存模型
model.save_pretrained('./fine_tuned_llama_lora12')
tokenizer.save_pretrained('./fine_tuned_llama_lora12')