import os
os.environ['TRANSFORMERS_OFFLINE'] = "1"
from transformers import LlamaTokenizer, LlamaForSequenceClassification, AdamW,AutoTokenizer,LlamaModel,AutoModel
from peft import LoraConfig, get_peft_model, PeftModel
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel,Wav2Vec2Model, Wav2Vec2Processor
import librosa
from llama_cpp import Llama
from scipy.signal import firwin, lfilter
import numpy as np
import torch.nn.functional as F
import pandas as pd
import random


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

# 音频重采样
def resample_audio(audio_path, target_sr=16000):
    audio, sr = librosa.load(audio_path, sr=None)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio, target_sr

def load_csv_text(csv_path):
    df = pd.read_csv(csv_path, encoding='utf-8')
    text = ' '.join(df['value'].astype(str).str.replace('"', '').tolist()) # 文本在CSV的第一列
    return text
# FIR滤波器设计
def apply_fir_filter(audio, sr):
    num_taps = 101
    cutoff = 6000
    filter_coefficients = firwin(num_taps, cutoff / (sr / 2), window=('kaiser', 5.0))
    filtered_audio = lfilter(filter_coefficients, 1.0, audio)
    return filtered_audio

# 提取MFCC特征
def extract_features(audio_data, sr=16000, num_mfcc=128, num_frames=10):
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=num_mfcc)
    mfcc = mfcc.T
    if mfcc.shape[0] < num_frames:
        mfcc = np.pad(mfcc, ((0, num_frames - mfcc.shape[0]), (0, 0)), mode='constant')
    else:
        mfcc = mfcc[:num_frames, :]
    return mfcc

class AttentionBasedFusion(nn.Module):
    def __init__(self, text_dim, audio_dim, hidden_dim):
        super(AttentionBasedFusion, self).__init__()
        self.text_linear = nn.Linear(text_dim, hidden_dim)
        self.audio_linear = nn.Linear(audio_dim, hidden_dim)
        
        self.attention_weights_text = nn.Linear(hidden_dim, 1)
        self.attention_weights_audio = nn.Linear(hidden_dim, 1)

    def forward(self, text_features, audio_features):
        text_hidden = self.text_linear(text_features)  # (batch_size, hidden_dim)
        audio_hidden = self.audio_linear(audio_features)  # (batch_size, hidden_dim)
        
        attention_score_text = self.attention_weights_text(text_hidden)  # (batch_size, 1)
        attention_score_audio = self.attention_weights_audio(audio_hidden)  # (batch_size, 1)
        
        attention_scores = torch.cat((attention_score_text, attention_score_audio), dim=1)  # (batch_size, 2)
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch_size, 2)
        
        weighted_text = attention_weights[:, 0].unsqueeze(1) * text_hidden  # (batch_size, hidden_dim)
        weighted_audio = attention_weights[:, 1].unsqueeze(1) * audio_hidden  # (batch_size, hidden_dim)
        
        fused_features = torch.cat((weighted_text, weighted_audio), dim=1)  # (batch_size, 2*hidden_dim)
        
        return fused_features

class StudentModel(nn.Module):
    def __init__(self, text_model_path, audio_model_path):
        super(StudentModel, self).__init__()
        self.text_model = BertModel.from_pretrained(text_model_path)
        self.audio_model = Wav2Vec2Model.from_pretrained(audio_model_path)
        self.processor = Wav2Vec2Processor.from_pretrained(audio_model_path)
        
        self.gn_text = nn.GroupNorm(num_groups=32, num_channels=768)
        self.gn_audio = nn.GroupNorm(num_groups=32, num_channels=768)
        
        self.attention_fusion = AttentionBasedFusion(text_dim=768, audio_dim=768, hidden_dim=512)
        
        self.fc = nn.Sequential(
            nn.Linear(512 * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )
        
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, text_input, audio_input):
        text_features = self.text_model(**text_input).last_hidden_state.mean(dim=1)
        text_features = self.gn_text(text_features)
        
        audio_input_dict = {"input_values": audio_input}
        audio_features = self.audio_model(**audio_input_dict).last_hidden_state.mean(dim=1)
        audio_features = self.gn_audio(audio_features)
        
        fused_features = self.attention_fusion(text_features, audio_features)
        print(fused_features.shape)  # 搴旇鏄?(batch_size, 1024)
        output = self.fc(fused_features)
        return output

import logging

# 配置日志记录
logging.basicConfig(
    filename='training_log.log',  # 日志文件名
    level=logging.INFO,          # 日志级别为 INFO
    format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
    datefmt='%Y-%m-%d %H:%M:%S'  # 时间格式
)

criterion_kl = nn.KLDivLoss(reduction='batchmean')
criterion_ce = nn.CrossEntropyLoss()
# 知识蒸馏
def train_student_model(student_model, audio_teacher, text_data, audio_data,
                        epochs=14, save_path="Z:\\2024program", batch_size=2, alpha=0.5, true_labels=[]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    print(device)
    student_model.to(device)
    optimizer = optim.Adam(student_model.parameters(), lr=5e-5)
    
    tokenizer = BertTokenizer.from_pretrained(text_model_path)
    audio_processor = Wav2Vec2Processor.from_pretrained(audio_model_path)
    
    for epoch in range(epochs):
        student_model.train()
        total_loss = 0
        # indices = random.sample(range(len(audio_data)), 4)
        indices=len(text_data)
        batch_text_data = [text_data[i] for i in range(indices)]
        batch_audio_data = [audio_data[i] for i in  range(indices)]
        real_labels = [true_labels[i] for i in range(indices)]
        
        for i in range(0, len(batch_text_data), batch_size):
            # 获取当前批次的文本和音频数据
            current_batch_text = batch_text_data[i:i+batch_size]
            current_batch_audio = batch_audio_data[i:i+batch_size]
            current_real_labels = real_labels[i:i+batch_size]
            real_labels_tensor = torch.tensor(current_real_labels, dtype=torch.long).to(device)
            
            # 记录真实标签
            logging.info(f"real_labels: {real_labels_tensor}")
            logging.info(f"Batch text data: {current_batch_text}")
            
            # 处理文本输入
            text_inputs = [tokenizer(load_csv_text(text), padding='max_length', truncation=True, max_length=512, return_tensors="pt").to(device) for text in current_batch_text]
            torch.cuda.empty_cache()
            
            # 处理音频输入
            audio_inputs = []
            for audio in current_batch_audio:
                audio_input, _ = librosa.load(audio, sr=16000)
                # audio_input_processed = audio_processor(audio_input, return_tensors="pt", sampling_rate=16000).input_values.to(device)
                # audio_inputs.append(audio_input_processed)
                segments = np.array_split(audio_input,5)  # 分割成10个片段
                audio_inputs.extend([audio_processor(segment, return_tensors="pt", sampling_rate=16000).input_values.to(device) for segment in segments ])
            
            # 获取教师模型的预测结果
            text_teacher_outputs = [predict(load_csv_text(text), device=device) for text in current_batch_text]
            audio_teacher_outputs = [audio_teacher.predict(audio) for audio in current_batch_audio]
            
            # 将教师模型的输出转换为标签
            text_labels = torch.tensor([output[0] for output in text_teacher_outputs], dtype=torch.float32).to(device)
            audio_labels = torch.tensor([output for output in audio_teacher_outputs], dtype=torch.float32).to(device)
            
            # 组合教师标签
            # combined_labels = 0.3 * text_labels + 0.7 * audio_labels

            #     # 对 combined_labels 进行 softmax 操作
            # combined_labels = torch.nn.functional.softmax(combined_labels, dim=-1)
            combined_labels = torch.stack((text_labels, audio_labels), dim=0).mean(dim=0)
            combined_labels = torch.nn.functional.softmax(combined_labels, dim=-1)
            
            # 学生模型的前向传播
            student_outputs = [student_model(text_input, audio_input) for text_input, audio_input in zip(text_inputs, audio_inputs)]
            student_outputs = torch.stack(student_outputs)
            
            # 蒸馏损失
            kl_divergence_loss = criterion_kl(F.log_softmax(student_outputs.squeeze(1), dim=-1), combined_labels)
            
            # 交叉熵损失
   
            cross_entropy_loss = criterion_ce(student_outputs.squeeze(1), real_labels_tensor)
            
            # 结合损失
            loss = alpha * kl_divergence_loss + (1 - alpha) * cross_entropy_loss
            #   loss=cross_entropy_loss
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            logging.info(f"Epoch {epoch+1}, Batch {i}, Loss: {loss.item()}")
         
        
        # 记录每个 epoch 的总损失
        epoch_loss = total_loss / len(batch_text_data)
        logging.info(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss}")
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss}")
    
    # 保存模型
    torch.save(student_model.state_dict(), save_path+'\\NO_model.pth')
    logging.info(f"Model saved to {save_path}")
    print(f"Model saved to {save_path}")
 
    
# 初始化教师模型
tokenizer = AutoTokenizer.from_pretrained('Z:\\2024program\\fine_tuned_llama_lora')
tokenizer.pad_token = tokenizer.eos_token

llama_model = LlamaModel.from_pretrained(r'Z:\\2024program\\llama-3.2-1B-spinquant-hf')
model = LlamaClassifier(llama_model, num_labels=2)


lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
)


text_teach = PeftModel.from_pretrained(model, 'Z:\\2024program\\fine_tuned_llama_lora')
text_teach.to('cuda')

#text_teacher = TextTeacher("http://localhost:1234/v1", "xxxxxx", "model-identifier")
#audio_teacher = AudioTeacher("C:\\Users\\admin\\Desktop\\2024program\\Llama_Gan_Gao_Huang\\model.pt")
def predict(text, model=text_teach , tokenizer=tokenizer, device='cpu', max_length=512):
    inputs = tokenizer(
        text,
        return_tensors='pt',
        max_length=max_length,
        truncation=True,
        padding='max_length'
    ).to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        logits = outputs
        probabilities = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()

    return probabilities.cpu().numpy()

class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # 双向LSTM的输出维度是hidden_dim * 2

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

    def predict(self, audio_file, sr=16000, num_mfcc=128, num_frames=10):
        self.eval()
        audio, _ = resample_audio(audio_file, sr)
        filtered_audio = apply_fir_filter(audio, sr)
        features = extract_features(filtered_audio, sr=sr, num_mfcc=num_mfcc, num_frames=num_frames)
        inputs = torch.tensor([features], dtype=torch.float32)
        with torch.no_grad():
            outputs = self(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)  # 将输出转换为概率分布
        return probs.squeeze().tolist()  # 返回概率值


audio_teacher=BiLSTM(input_dim=128, hidden_dim=64, output_dim=2, num_layers=1)
audio_teacher.load_state_dict(torch.load('Z:\\2024program\\0.98quantized_bilstm_model.pth'))

# 初始化学生模型
config_path = r"uncased_L-12_H-768_A-12\\bert_config.json"
text_model_path = r"Z:\\2024program\\bert_uncased_L-12_H-768_A-12"
audio_model_path = r"Z:\\2024program\\wav2vec2-base-960h"

student_model = StudentModel(text_model_path , audio_model_path)

# 准备数据
label_file = 'Z:\\2024program\Audio_train_code_hyf_new\data\labels.xlsx'
label_df = pd.read_excel(label_file)
label_dict = dict(zip(label_df['Participant_ID'], label_df['Gender']))

# 读取音频文件
audio_dir = 'Z:\\2024program\\audio_segments'
audio_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith('.wav')]

# 读取转录文件
transcript_dir = 'Z:\\2024program\\text_model\\textdate'
transcript_files = [os.path.join(transcript_dir, f) for f in os.listdir(transcript_dir) if f.endswith('.csv')]

# 过滤掉没有标签的音频文件
def get_label(audio_file, label_dict):
    participant_id = int(os.path.basename(audio_file).split('_')[0])
    return label_dict.get(participant_id, -1)

audio_files = [f for f in audio_files if get_label(f, label_dict) != -1]

# 过滤掉没有音频文件的标签
valid_participant_ids = {int(os.path.basename(f).split('_')[0]) for f in audio_files}
label_dict = {k: v for k, v in label_dict.items() if k in valid_participant_ids}

# 匹配音频文件和转录文件
def match_audio_transcript(audio_files, transcript_files):
    audio_transcript_pairs = []
    for audio_file in audio_files:
        participant_id = int(os.path.basename(audio_file).split('_')[0])
        transcript_file = next((f for f in transcript_files if int(os.path.basename(f).split('_')[0]) == participant_id), None)
        if transcript_file:
            audio_transcript_pairs.append((audio_file, transcript_file))
    return audio_transcript_pairs

audio_transcript_pairs = match_audio_transcript(audio_files, transcript_files)

# 获取标签
true_labels = [label_dict[int(os.path.basename(audio_file).split('_')[0])] for audio_file, _ in audio_transcript_pairs]

# 数据准备
def prepare_data(audio_transcript_pairs):
    audio_data = []
    transcript_data = []
    for audio_file, transcript_file in audio_transcript_pairs:
        try:
            audio, _ = librosa.load(audio_file, sr=16000)
            with open(transcript_file, 'r', encoding='utf-8') as f:
                transcript = f.read()
            audio_data.append(audio_file)
            transcript_data.append(transcript_file)
        except Exception as e:
            print(f"Error loading audio or transcript file: {e}")
    return audio_data, transcript_data

audio_data,text_data = prepare_data(audio_transcript_pairs)
#模型评估

# 训练学生模型
train_student_model(student_model, audio_teacher, text_data, audio_data,true_labels=true_labels)