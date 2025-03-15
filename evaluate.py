import torch
from transformers import BertTokenizer, Wav2Vec2Processor, BertForSequenceClassification, Wav2Vec2ForCTC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, classification_report,roc_curve
import librosa
import numpy as np
from langchain_community.document_loaders import TextLoader
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel,Wav2Vec2Model, Wav2Vec2Processor
import librosa
from llama_cpp import Llama
from scipy.signal import firwin, lfilter
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
# 音频重采样
def resample_audio(audio_path, target_sr=16000):
    audio, sr = librosa.load(audio_path, sr=None)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio, target_sr


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
def extract_sentiment_label(output):
    content = output['content']
    if "Positive" in content:
        return 1
    elif "Negative" in content:
        return 0
    else:
        return 0  # 默认返回中性或负面的标签
class StudentModel(nn.Module):
    def __init__(self, text_model_path, audio_model_path):
        super(StudentModel, self).__init__()
        self.text_model = BertModel.from_pretrained(text_model_path)
        self.audio_model = Wav2Vec2Model.from_pretrained(audio_model_path)
        self.processor = Wav2Vec2Processor.from_pretrained(audio_model_path)
        
        # 添加组归一化层
        self.gn_text = nn.GroupNorm(num_groups=32, num_channels=768)
        self.gn_audio = nn.GroupNorm(num_groups=32, num_channels=768)
        
        # 特征维度
        self.fc = nn.Sequential(
            nn.Linear(768 + 768, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

        
        # 使用Kaiming初始化方法初始化线性层
        nn.init.kaiming_normal_(self.fc[0].weight, mode='fan_in', nonlinearity='relu')  

    def forward(self, text_input, audio_input):
        text_features = self.text_model(**text_input).last_hidden_state.mean(dim=1)
        text_features = self.gn_text(text_features)
        
        audio_input_dict = {"input_values": audio_input}
        audio_features = self.audio_model(**audio_input_dict).last_hidden_state.mean(dim=1)
        audio_features = self.gn_audio(audio_features)
        
        combined_features = torch.cat((text_features, audio_features), dim=1)
        output = self.fc(combined_features)
        return output
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
            _, predicted = torch.max(outputs, 1)
        return predicted.item() 


audio_teacher=BiLSTM(input_dim=128, hidden_dim=64, output_dim=2, num_layers=1)
audio_teacher.load_state_dict(torch.load('Z:\\2024program\\0.98quantized_bilstm_model.pth'))
    

# 加载标签文件
label_file = 'Z:\\2024program\\Audio_train_code_hyf_new/data/labels.xlsx'
label_df = pd.read_excel(label_file)
label_dict = dict(zip(label_df['Participant_ID'], label_df['Gender']))

# 读取音频文件
audio_dir = 'Z:\\2024program\\audio_test'
audio_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith('.wav')]

# 读取转录文件
transcript_dir = 'Z:\\2024program\\test'
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
def load_csv_text(csv_path):
    df = pd.read_csv(csv_path, encoding='utf-8')
    text = ' '.join(df['value'].astype(str).str.replace('"', '').tolist())  # 文本在CSV的第一列
    return text
# 数据准备
def prepare_data(audio_transcript_pairs):
    audio_data = []
    transcript_data = []
    for audio_file, transcript_file in audio_transcript_pairs:
        try:
            # audio, _ = librosa.load(audio_file, sr=16000)
            # with open(transcript_file, 'r', encoding='utf-8') as f:
            #     transcript = f.read()
            audio_data.append(audio_file)
            transcript_data.append(transcript_file)
        except Exception as e:
            print(f"Error loading audio or transcript file: {e}")
    return audio_data, transcript_data

audio_data, transcript_data = prepare_data(audio_transcript_pairs)

# 评估模型
def evaluate_student_model(student_model_path, audio_data, transcript_data, true_labels, text_model_path, audio_model_path, batch_size=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载学生模型
    student_model = StudentModel(text_model_path, audio_model_path)
    student_model.load_state_dict(torch.load(student_model_path, map_location=device))
    student_model.to(device)
    student_model.eval()
    
    tokenizer = BertTokenizer.from_pretrained(text_model_path)
    audio_processor = Wav2Vec2Processor.from_pretrained(audio_model_path)
    
    predicted_labels = []
    predicted_probabilities = []
    
    with torch.no_grad():
        for i in range(0, len(audio_data), batch_size):
            batch_audio_data = audio_data[i:i+batch_size]
            batch_transcript_data = transcript_data[i:i+batch_size]
            print(batch_transcript_data)
            # 处理文本输入
            text_inputs = [tokenizer(load_csv_text(text), padding='max_length', truncation=True, max_length=512, return_tensors="pt").to(device) for text in batch_transcript_data]
            
            # 处理音频输入
            audio_inputs = []
            for audio in batch_audio_data:
                audio_input, _ = librosa.load(audio, sr=16000)  # 加载音频数据
                # print(f"Audio Input Shape (before processing): {audio_input.shape}")  # 打印音频数据的形状

                # # 使用 audio_processor 处理音频数据
                # audio_input_processed = audio_processor(audio_input, return_tensors="pt", sampling_rate=16000).input_values.to(device)
                # print(f"Audio Input Shape (after processing): {audio_input_processed.shape}")  # 打印处理后的音频数据形状
                # audio_inputs.append(audio_input_processed)  # 将处理后的音频数据添加到列表中
                segments = np.array_split(audio_input,5)  # 分割成10个片段
                audio_inputs.extend([audio_processor(segment, return_tensors="pt", sampling_rate=16000).input_values.to(device) for segment in segments ])
            
            # 模型前向传播
            student_outputs = []
            for text_input, audio_input in zip(text_inputs, audio_inputs):
                if text_input is not None and audio_input is not None:
                    output = student_model(text_input, audio_input)
                    student_outputs.append(output)
            
            if student_outputs:
                student_outputs = torch.stack(student_outputs).squeeze(1)  # 移除不必要的中间维度
                
                # 确保 student_outputs 的形状一致
                if student_outputs.shape[1] == 2:
                    student_probabilities = torch.nn.functional.softmax(student_outputs, dim=1)
                    _, predicted = torch.max(student_probabilities, dim=-1)
                    predicted_labels.extend(predicted.cpu().numpy())

                    # 收集预测概率用于AUC计算
                    predicted_probabilities.extend(student_probabilities[:, 1].cpu().numpy())
                else:
                    # 调整形状
                    student_outputs = student_outputs.squeeze(1)
                    if student_outputs.shape[1] == 2:
                        student_probabilities = torch.nn.functional.softmax(student_outputs, dim=1)
                        _, predicted = torch.max(student_probabilities, dim=-1)
                        predicted_labels.extend(predicted.cpu().numpy())
                        print(predicted)
                        # 收集预测概率用于AUC计算
                        predicted_probabilities.extend(student_probabilities[:, 1].cpu().numpy())
                    else:
                        print(f"Skipping batch due to inconsistent output shape: {student_outputs.shape}")
            else:
                print("Skipping batch due to empty student_outputs")
    
    # 转为numpy数组
    print("true_labels:", true_labels)
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    print(predicted_labels)
    predicted_probabilities = np.array(predicted_probabilities)
    
    # 计算评估指标
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, zero_division=0)
    recall = recall_score(true_labels, predicted_labels, zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, zero_division=0)
    auc = roc_auc_score(true_labels, predicted_probabilities)
    
    # 打印指标
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'AUC: {auc:.4f}')
    
    # 绘制AUC曲线
    fpr, tpr, _ = roc_curve(true_labels, predicted_probabilities)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()
    
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1, "auc": auc}

# 调用评估函数
evaluate_student_model(
    student_model_path="Z:\\2024program\\Bert_modeL.pth",
    audio_data=audio_data,
    transcript_data=transcript_data,
    true_labels=true_labels,
    text_model_path="Z:\\2024program\\bert_uncased_L-12_H-768_A-12",
    audio_model_path="Z:\\2024program\\wav2vec2-base-960h"
)