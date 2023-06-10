import sys
import codecs
import json
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# # 입력 및 출력 인코딩 설정
sys.stdin = codecs.getreader('utf-8')(sys.stdin.detach())
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())


data = sys.stdin.read()

# # JSON 데이터를 딕셔너리로 역직렬화
jsonData = json.loads(data)
# print('데이터',jsonData) 

##################################################################################


class CustomDataset(Dataset):

    def __init__(self, data_path):
        self.session_dataset = self.preprocess(data_path)
        # label은 다음 3종류로 분류('abuse' : 욕설 포함, 'hate' : 그 외 혐오 표현, 'neutral' : 중립)
        self.labels = ['hate', 'neutral']
        # tokenizer로 Kobert 지정
        self.tokenizer = AutoTokenizer.from_pretrained("skt/kobert-base-v1")

    def __len__(self):
        return len(self.session_dataset)

    def __getitem__(self, idx):
        return self.session_dataset[idx]

    def preprocess(self, data_path):
        """preprocessing csv files
        """

        '''
        1. Speaker에 대한 label encoding을 진행하되 Dialogue_ID 단위로 적용
        2. Dialogue별로 speaker, origin_text, types를 하나의 리스트로 묶음
        3. 하나의 Dialogue 내 발화가 단계적으로 누적되도록 데이터 구성
        '''
        datasets = []
        # Load dataset
        sep_data = data_path[['speaker', 'origin_text', 'types', 'dialogue']].copy()

        # Dialogue_ID별 화자 넘버링
        start = -1
        speaker_to_num = []
        for idx in sep_data.index:
            # initialize
            if sep_data.loc[idx, 'dialogue'] != start:
                start += 1
                speaker_num = 0
                speaker_list = []

            speaker = sep_data.loc[idx, 'speaker']
            if speaker not in speaker_list:
                speaker_to_num.append(speaker_num)
                speaker_num += 1
                speaker_list.append(speaker)
            else:
                speaker_to_num.append(speaker_list.index(speaker))
        sep_data['speaker'] = speaker_to_num

        # Dialogue기준으로 화자, 발화, 타입 데이터를 groupby
        grouped_by_dialogue = sep_data.groupby(['dialogue'], as_index=True)

        def func(x):
            """화자, 발화, 감정을 하나의 리스트에 담아 session으로 정의한다."""
            result = x.apply(lambda x: [x['speaker'], x['origin_text'], x['types']], axis=1)
            return result

        dialogue_session = grouped_by_dialogue.apply(func)  # dialogue와 session번호로 구분된 multi-index
        '''dialogue_session result (실제 결과에서는 Speaker의 이름 대신 label 번호가 적용된다.)
        '''

        session_datasets = []
        for idx in data_path['dialogue'].unique():
            session_datasets.append(dialogue_session[idx].tolist())

        # 세션 내 대화를 연이어 붙이는 작업 진행
        for sess in session_datasets:
            length = len(sess)
            for i in range(1, length + 1):
                datasets.append(sess[:i])

        return datasets

    def collate_fn(self, sessions):
        '''
        PM 만들기
            : PM 데이터는 하나의 세션에서 예측하고자 하는 마지막 utt의 타입과 동일한 화자의 이전 utt를 따로 뽑아둔다.
            아래 예시는 6번째 세션에서 최종 화자의 이전 발화를 담은 케이스를 의미한다.
            0 = cls token, 2 = sep token
                [
                    [],
                    [],
                    [],
                    [],
                    [],
                    [tensor([[    0,  1793,   328,  1793,     6,    52,   115,   213,     7,     5,
                                827,     6,   593,    84,  2349,     8,   847,   106,   160,    23,
                                5,  1300,     4,     2,   370,    17,    27,   241,    10, 16333,
                                328,     2]])],
                    [],...
                ]

        CoM 만들기
            : 각 세션에서 뽑은 모든 utt를 보관
                tensor([[    0,  1793,   328,  1793,     6,    52,   115,   213,     7,     5,
                    827,     6,   593,    84,  2349,     8,   847,   106,   160,    23,
                        5,  1300,     4,     2,   370,    17,    27,   241,    10, 16333,
                    328,     2]])
        '''

        # tokenize 시 overflow를 막기 위해 최대 길이를 크게 지정
        max_seq_len = 512
        batch_input = []
        batch_labels = []
        batch_PM_input = []

        for session in sessions:
            input_sep = self.tokenizer.cls_token  # tokenizer.cls_token:str = '<s>'
            curr_speaker, curr_ot, curr_types = session[-1]  # current는 세션 내 마지막 화자, 발화, 타입 정보를 의미한다.
            PM_input = []
            for i, line in enumerate(session):
                speaker, ot, types = line
                input_sep += " " + ot + self.tokenizer.sep_token  # tokenizer.sep_token:str = '</s>'
                if i < len(session) - 1 and speaker == curr_speaker:  # curr화자의 이전 발화가 존재할 경우를 의미

                    # PM에 대한 Kobert tokenize (직접 cls token과 sep token을 붙여주므로 add_special_tokens를 False로 설정)
                    PM_input.append(self.tokenizer.encode(ot, add_special_tokens=False, return_tensors='pt'))

            batch_PM_input.append(PM_input)
            batch_input.append(input_sep)
            batch_labels.append(self.labels.index(types))  # 각 세션의 마지막 화자의 타입만 label로 적용

        # CoM Kobert tokenize (padding True로 설정하면 각 배치 내 가장 긴 utt에 대해 패딩을 진행한다.)
        tokenized_ot = self.tokenizer(batch_input, add_special_tokens=False, return_tensors='pt',
                                      max_length=max_seq_len, truncation=True, padding=True)
        batch_input_ids = tokenized_ot['input_ids']
        batch_attention_mask = tokenized_ot['attention_mask']

        return batch_input_ids, batch_attention_mask, batch_PM_input, torch.tensor(batch_labels)


class ERCModel(nn.Module):
    def __init__(self, num_class, device):
        super().__init__()  # 미 입력 시 에러 발생

        # CoM, PM model로 Kobert 지정
        self.com_model = AutoModel.from_pretrained("skt/kobert-base-v1")
        self.PM_model = AutoModel.from_pretrained("skt/kobert-base-v1")

        '''GRU 모델 세팅 (참조: <https://pytorch.org/docs/stable/generated/torch.nn.GRU.html>)
            - GRU 모델의 input, ouput dimension은 roberta의 hidden size를 따른다.
            - input size (seq_len, bs, h)를 입력으로 받고, output size (num_layers, bs, h)를 출력한다.
            - bidirectional=True로 설정된다면 num_layers * 2를 해야하지만 논문에서 제안한 대로. 여기서는 False로 설정한다.
        '''
        self.hidden_dim = self.com_model.config.hidden_size  # roberta의 hidden_dim (768)
        self.h_0 = torch.zeros(size=(2, 1, self.hidden_dim)).to(
            device)  # (num_layers * num_directions, batch, hidden_size)
        self.speakerGRU = nn.GRU(self.hidden_dim, self.hidden_dim, 2,
                                 dropout=0.3)  # (input, hidden, num_layer) (BERT_hidden_size, BERT_hidden_size, num_layer)

        # classification
        self.W = nn.Sequential(
            nn.Linear(self.hidden_dim, num_class),
            nn.Softmax(dim=-1)
        )

    def forward(self, data):

        input_ids = data[0].to(device)
        attention_mask = data[1].to(device)
        PM_input = data[2]  # PM은 리스트 내 여러 utt 텐서가 존재하는 형태이므로 to(device)를 개별적으로 적용해야 한다.
        labels = data[3].to(device)

        # CoM_model의 cls 토큰 확보
        com_cls_output = self.com_model(input_ids, attention_mask)['last_hidden_state'][:, 0, :]

        '''PM 데이터는 각 세션별로 존재하지 않는 경우부터 다량 존재하는 경우로 나뉜다.
        PM 데이터가 없을 경우 0값의 행렬을, 있을 경우 각 utt를 Kobert 모델에 학습시킨 뒤 GRU 모델로 추가 학습시킨다.
        '''
        pm_gru_final = []
        for utts in PM_input:
            if utts:  # 현재 세션의 PM tensor가 존재한다면
                pm_cls_output = []
                for utt in utts:
                    cls_output = self.PM_model(utt.to(device))['last_hidden_state'][:, 0,
                                 :]  # input_ids만 넣을 경우 attention mask는 자동 1로 채워진다.
                    pm_cls_output.append(cls_output)
                pm_output = torch.cat(pm_cls_output, 0).unsqueeze(1)  # (speaker_num, batch=1, hidden_dim)
                pm_gru_output, _ = self.speakerGRU(pm_output, self.h_0)
                pm_gru_final.append(pm_gru_output[-1, :, :])  # (1(bs), hidden_dim) 마지막 uttr token에 대해서만 가져간다.
            else:  # 존재하지 않을 경우
                pm_gru_final.append(torch.zeros(1, self.hidden_dim).to(device))  # PM tensor가 비어있다면 0값 차원을 맞춰 보낸다.
        pm_gru_final = torch.cat(pm_gru_final, 0)

        # CoM과 PM을 element-wise sum한 뒤 label 차원으로 축소 후 softmax
        final_output = self.W(com_cls_output + pm_gru_final)

        return final_output, labels


def get_model_predict(inputdata, device):
    model = ERCModel(2, device).to(device)
    # 학습시킨 모델 불러오기
    model.load_state_dict(torch.load('model_label_2.pt', map_location=torch.device('cpu')))

    tokenizer = AutoTokenizer.from_pretrained("skt/kobert-base-v1")

    # Load dataset
    data_list = pd.DataFrame(columns=['speaker', 'origin_text', 'types', 'dialogue'])
    # 입력받은 채팅내용 및 화자(닉네임), dialogue 추가
    # [{'speaker': 'John', 'origin_text': 'Hello', 'types': 'neutral', 'dialogue': 0}]
    new_data = pd.DataFrame.from_records(inputdata)
    data_list = pd.concat([data_list, new_data], ignore_index=True)
    data_list.groupby('types', group_keys=False)
    dataset = CustomDataset(data_list)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn)
    loss_fn = nn.CrossEntropyLoss().to(device)

    # valid_loss
    valid_loss = 0
    pred_list, target_list = [], []

    label = ['hate', 'neutral']

    with torch.no_grad():
        for data in tqdm(dataloader, leave=False):
            pred, target = model(data)
            loss = loss_fn(pred, target)

            valid_loss += loss.item()  # 최종 평균을 구하기 위해 전부 더한다.

            # evaluation
            pred_label = pred.argmax(1).item()
            target_label = target.item()
            pred_list.append(pred_label)
            target_list.append(target_label)

    return str(label[pred_list[-1]])

chatList = []

device = torch.device('cpu')
chatList.append(jsonData)
result = get_model_predict(chatList, device)
sys.stdout.write(result)
sys.stdout.flush()