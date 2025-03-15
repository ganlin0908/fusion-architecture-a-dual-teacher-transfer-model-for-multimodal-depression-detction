import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import librosa
from scipy.signal import firwin, lfilter
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
device = torch.device( "cpu")
print(f"Using device: {device}")

# 定义BiLSTM模型
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

# 音频重采样
def resample_audio(audio_path, target_sr=16000):
    audio, sr = librosa.load(audio_path, sr=None)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio, target_sr

# FIR滤波器设计
def apply_fir_filter(audio, sr):
    num_taps = 101
    cutoff = 7000
    filter_coefficients = firwin(num_taps, cutoff/(sr/2), window=('kaiser',5.0))
    filtered_audio = lfilter(filter_coefficients, 1.0, audio)
    return filtered_audio

# 语音活动检测（VAD）
def voice_activity_detection(audio, sr, threshold=0.05):
    energy = np.sum(np.abs(audio)**2, axis=0)
    threshold_energy = threshold * np.max(energy)
    speech_segments = np.where(energy > threshold_energy)[0]
    return speech_segments

# 读取CSV文件中的转录文本
def read_transcripts_from_csv(csv_path):
    # df = pd.read_csv(csv_path)
    # transcripts = df.iloc[:, 0].tolist()
    df = pd.read_csv(csv_path, delimiter='\t')
    # 只提取 'value' 列的数据
    transcripts = df['value'].tolist()
    return transcripts

# 数据准备
def prepare_data(audio_files, transcript_files):
    resampled_audios = [resample_audio(audio_file) for audio_file in audio_files]
    filtered_audios = [apply_fir_filter(audio, sr) for audio, sr in resampled_audios]
    transcripts = [read_transcripts_from_csv(transcript_file) for transcript_file in transcript_files]
    return filtered_audios, transcripts



def extract_features(audio_data, sr=16000, num_mfcc=128, num_frames=10):
    # 提取MFCC特征
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=num_mfcc)
    # 转置以获得形状为 (frames, num_mfcc)
    mfcc = mfcc.T
    # 如果特征帧数少于num_frames，进行填充
    if mfcc.shape[0] < num_frames:
        mfcc = np.pad(mfcc, ((0, num_frames - mfcc.shape[0]), (0, 0)), mode='constant')
    # 如果特征帧数多于num_frames，进行截断
    else:
        mfcc = mfcc[:num_frames, :]
    return mfcc


# 准备特征
def prepare_features(filtered_audios, sr=16000, num_mfcc=128, num_frames=10):
    features_list = []
    for audio in filtered_audios:
        features = extract_features(audio, sr=sr, num_mfcc=num_mfcc, num_frames=num_frames)
        features_list.append(features)
    return features_list


def get_label(audio_file, label_dict):
    # 提取Participant_ID
    participant_id = os.path.basename(audio_file).split('_')[0]
    # 根据Participant_ID返回对应的标签
    return label_dict.get(int(participant_id),-1)
    

# 训练模型
def train_model(model, dataloader, criterion, optimizer,scheduler, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到GPU
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step(epoch_loss)
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到GPU
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

# 量化（未定）
def quantize_weights(model, num_bits=8):
    for name, param in model.named_parameters():
        param.data = (param.data / torch.max(torch.abs(param.data)) * (2**(num_bits - 1) - 1)).round() / ((2**(num_bits - 1) - 1) / torch.max(torch.abs(param.data)))
    return model

if __name__ == "__main__":
    #目前是直接输入，但是需要对应标签
    # 314_P，308_P，抑郁症患者,标签是0
    # 302_P，334_P，正常人，标签是1


    # audio_files = [
    #     'C:\\Users\\admin\\Desktop\\2024program\\Audio_train_code_hyf_new\\dataset\\302_P\\302_AUDIO.wav', 
    #     'C:\\Users\\admin\\Desktop\\2024program\\Audio_train_code_hyf_new\\dataset\\308_P\\308_AUDIO.wav', 
    #     'C:\\Users\\admin\\Desktop\\2024program\\Audio_train_code_hyf_new\\dataset\\314_P\\314_AUDIO.wav', 
    #     'C:\\Users\\admin\\Desktop\\2024program\\Audio_train_code_hyf_new\\dataset\\334_P\\334_AUDIO.wav'
    # ]
    
    # transcript_files = [
    #     'C:\\Users\\admin\\Desktop\\2024program\\Audio_train_code_hyf_new\\dataset\\302_P\\302_TRANSCRIPT.csv', 
    #     'C:\\Users\\admin\\Desktop\\2024program\\Audio_train_code_hyf_new\\dataset\\308_P\\308_TRANSCRIPT.csv', 
    #     'C:\\Users\\admin\\Desktop\\2024program\\Audio_train_code_hyf_new\\dataset\\314_P\\314_TRANSCRIPT.csv', 
    #     'C:\\Users\\admin\\Desktop\\2024program\\Audio_train_code_hyf_new\\dataset\\334_P\\334_TRANSCRIPT.csv'
    # ]

    label_file = 'Z:\\2024program\\Audio_train_code_hyf_new\\data\\labels.xlsx'
    label_df = pd.read_excel(label_file)
    label_dict = dict(zip(label_df['Participant_ID'], label_df['Gender']))
 
    # 读取音频文件
    audio_dir = 'Z:\\2024program\\audio_segments'
    audio_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith('.wav')]


    # 过滤掉没有标签的音频文件
    audio_files = [f for f in audio_files if get_label(f,label_dict) != -1]
    
    # 过滤掉没有音频文件的标签
    valid_participant_ids = {int(os.path.basename(f).split('_')[0]) for f in audio_files}
    label_dict = {k: v for k, v in label_dict.items() if k in valid_participant_ids}
    
    # 数据准备
    filtered_audios, _ = prepare_data(audio_files, [])  # 不需要转录文件
    
    # 提取特征并生成输入数据
    num_mfcc = 128
    num_frames = 10
    features_list = prepare_features(filtered_audios, sr=16000, num_mfcc=num_mfcc, num_frames=num_frames)
    inputs = torch.tensor(np.array(features_list), dtype=torch.float32)

    # 获取标签
    labels_list = [get_label(audio_file, label_dict) for audio_file in audio_files]
    
    labels = torch.tensor(labels_list, dtype=torch.long)
 
    # 创建数据加载器
    dataset = TensorDataset(inputs, labels)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 定义模型、损失函数和优化器
    model = BiLSTM(input_dim=128, hidden_dim=64, output_dim=2, num_layers=1)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1, verbose=True)

    # 训练模型
    train_model(model, dataloader, criterion, optimizer, scheduler,num_epochs=50)

     # 评估模型
    evaluate_model(model, dataloader)

    # 保存模型
    torch.save(model.state_dict(), 'bilstm_model.pth')

    # 量化模型权重
    quantized_model = quantize_weights(model)

    # 保存量化后的模型
    torch.save(quantized_model.state_dict(), 'quantized_bilstm_model.pth')

    print("Training and quantization complete. Models saved.")