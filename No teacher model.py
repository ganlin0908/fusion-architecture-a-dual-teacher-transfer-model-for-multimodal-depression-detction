import os
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import pandas as pd
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 閰嶇疆鏃ュ織璁板綍
logging.basicConfig(
    filename='bert_evaluation_log.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 瀹氫箟 BERT 鍒嗙被鍣ㄦā鍨?
class BertClassifier(nn.Module):
    def __init__(self, num_labels):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('Z:\\2024program\\bert_uncased_L-12_H-768_A-12')
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # 浣跨敤 [CLS] token 鐨勮緭鍑?
        logits = self.classifier(pooled_output)
        return logits

# 鏁版嵁鍑嗗鍑芥暟
def load_csv_text(csv_path):
    df = pd.read_csv(csv_path, encoding='utf-8')
    text = ' '.join(df['value'].astype(str).str.replace('"', '').tolist())
    return text

def prepare_data(audio_transcript_pairs):
    transcript_data = []
    for _, transcript_file in audio_transcript_pairs:
        try:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                transcript = f.read()
            transcript_data.append(transcript)
        except Exception as e:
            print(f"Error loading transcript file: {e}")
    return transcript_data

# 璇勪及鍑芥暟
def evaluate_model(model, text_data, true_labels, batch_size=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('Z:\\2024program\\bert_uncased_L-12_H-768_A-12')

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for i in range(0, len(text_data), batch_size):
            current_batch_text = text_data[i:i+batch_size]
            current_real_labels = true_labels[i:i+batch_size]

            # 澶勭悊鏂囨湰杈撳叆
            text_inputs = tokenizer(current_batch_text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)

            # 鍓嶅悜浼犳挱
            outputs = model(text_inputs['input_ids'], text_inputs['attention_mask'])
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()

            all_predictions.extend(predictions)
            all_labels.extend(current_real_labels)

    # 璁＄畻璇勪及鎸囨爣
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    logging.info(f"Evaluation Results:")
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1 Score: {f1:.4f}")

    print(f"Evaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return accuracy, precision, recall, f1

# 鍔犺浇璁粌濂界殑妯″瀷
def load_trained_model(model_path, num_labels=2):
    model = BertClassifier(num_labels=num_labels)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# 鏁版嵁鍑嗗
label_file = 'Z:\\2024program\\Audio_train_code_hyf_new\\data\\labels.xlsx'
label_df = pd.read_excel(label_file)
label_dict = dict(zip(label_df['Participant_ID'], label_df['Gender']))

# 璇诲彇闊抽鏂囦欢
audio_dir = 'Z:\\2024program\\audio_segments'
audio_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith('.wav')]

# 璇诲彇杞綍鏂囦欢
transcript_dir = 'Z:\\2024program\\test'
transcript_files = [os.path.join(transcript_dir, f) for f in os.listdir(transcript_dir) if f.endswith('.csv')]


def get_label(audio_file, label_dict):
    participant_id = int(os.path.basename(audio_file).split('_')[0])
    return label_dict.get(participant_id, -1)

audio_files = [f for f in audio_files if get_label(f, label_dict) != -1]

valid_participant_ids = {int(os.path.basename(f).split('_')[0]) for f in audio_files}
label_dict = {k: v for k, v in label_dict.items() if k in valid_participant_ids}


def match_audio_transcript(audio_files, transcript_files):
    audio_transcript_pairs = []
    for audio_file in audio_files:
        participant_id = int(os.path.basename(audio_file).split('_')[0])
        transcript_file = next((f for f in transcript_files if int(os.path.basename(f).split('_')[0]) == participant_id), None)
        if transcript_file:
            audio_transcript_pairs.append((audio_file, transcript_file))
    return audio_transcript_pairs

audio_transcript_pairs = match_audio_transcript(audio_files, transcript_files)

# 鑾峰彇鏍囩
true_labels = [label_dict[int(os.path.basename(audio_file).split('_')[0])] for audio_file, _ in audio_transcript_pairs]

# 鍑嗗鏁版嵁
text_data = prepare_data(audio_transcript_pairs)

# 鍔犺浇璁粌濂界殑妯″瀷
model_path = "Z:\\2024program\\bert_model.pth"
bert_model = load_trained_model(model_path)

# 璇勪及妯″瀷
evaluate_model(bert_model, text_data, true_labels)