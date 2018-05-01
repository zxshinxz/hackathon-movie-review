# -*- coding: utf-8 -*-

"""
Copyright 2018 NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import argparse
import os

import numpy as np
import torch

from torch.autograd import Variable
import torch.autograd as autograd
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torch.nn.functional as F

import nsml
# currently throwing error
from nsml import DATASET_PATH, HAS_DATASET, GPU_NUM, IS_ON_NSML
# from nsml import DATASET_PATH, HAS_DATASET, GPU_NUM
# IS_ON_NSML = False

from dataset import MovieReviewDataset, preprocess


# DONOTCHANGE: They are reserved for nsml
# This is for nsml leaderboard
def bind_model(model, config):
    # 학습한 모델을 저장하는 함수입니다.
    def save(filename, *args):
        checkpoint = {
            'model': model.state_dict()
        }
        torch.save(checkpoint, filename)

    # 저장한 모델을 불러올 수 있는 함수입니다.
    def load(filename, *args):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model'])
        print('Model loaded')

    def infer(raw_data, **kwargs):
        """

        :param raw_data: raw input (여기서는 문자열)을 입력받습니다
        :param kwargs:
        :return:
        """
        # dataset.py에서 작성한 preprocess 함수를 호출하여, 문자열을 벡터로 변환합니다
        preprocessed_data = preprocess(raw_data, config.strmaxlen)
        model.eval()
        # 저장한 모델에 입력값을 넣고 prediction 결과를 리턴받습니다
        output_prediction = model(preprocessed_data)
        point = output_prediction.data.squeeze(dim=1).tolist()
        # DONOTCHANGE: They are reserved for nsml
        # 리턴 결과는 [(confidence interval, 포인트)] 의 형태로 보내야만 리더보드에 올릴 수 있습니다. 리더보드 결과에 confidence interval의 값은 영향을 미치지 않습니다
        return list(zip(np.zeros(len(point)), point))

    # DONOTCHANGE: They are reserved for nsml
    # nsml에서 지정한 함수에 접근할 수 있도록 하는 함수입니다.
    nsml.bind(save=save, load=load, infer=infer)


def collate_fn(data: list):
    """
    PyTorch DataLoader에서 사용하는 collate_fn 입니다.
    기본 collate_fn가 리스트를 flatten하기 때문에 벡터 입력에 대해서 사용이 불가능해, 직접 작성합니다.

    :param data: 데이터 리스트
    :return:
    """
    review = []
    label = []
    for datum in data:
        review.append(datum[0])
        label.append(datum[1])
    # 각각 데이터, 레이블을 리턴
    return review, np.array(label)


class JJuModel(nn.Module):
    """
    영화리뷰 예측을 위한 Regression 모델입니다.
    """

    def __init__(self, embedding_dim: int, max_length: int):
        """
        initializer

        :param embedding_dim: 데이터 임베딩의 크기입니다
        :param max_length: 인풋 벡터의 최대 길이입니다 (첫 번째 레이어의 노드 수에 연관)
        """
        super(JJuModel, self).__init__()
        self.character_size = 251
        self.embedding_dim = 150
        self.embedding_dim_cnn = 24
        self.output_dim = 1  # Regression
        # self.max_length = max_length
        self.max_length = 150
        self.n_layers = 6
        self.hiddenSize = 150
        self.bidirectional = True
        self.n_directions = int(self.bidirectional) + 1

        # 임베딩
        self.embeddings = nn.Embedding(self.character_size, self.embedding_dim)
        self.embeddings_cnn = nn.Embedding(self.character_size, self.embedding_dim_cnn)

        self.rnn = nn.GRU(self.max_length, self.max_length, self.n_layers,
                          bidirectional=self.bidirectional,
                          batch_first=True,
                          dropout=0.27)

        self.cnn_res = nn.Sequential(
            nn.Conv1d(self.max_length, (self.max_length + 40) + 40, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d((self.max_length + 40) + 40, (self.max_length + 40) + 40, 5, padding=2),
            nn.BatchNorm1d((self.max_length + 40) + 40),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d((self.max_length + 40) + 40, ((self.max_length + 40) + 40) + 40, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(((self.max_length + 40) + 40) + 40, ((self.max_length + 40) + 40) + 40, 5, padding=2),
            nn.BatchNorm1d(((self.max_length + 40) + 40) + 40),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(((self.max_length + 40) + 40) + 40, (((self.max_length + 40) + 40) + 40), 3, padding=1),
            nn.ReLU(),
            nn.Conv1d((((self.max_length + 40) + 40) + 40), (((self.max_length + 40) + 40) + 40), 5, padding=2),
            nn.BatchNorm1d((((self.max_length + 40) + 40) + 40)),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # 첫 번째 레이어
        self.fc1 = nn.Linear(270, 150)

        # 두 번째 (아웃풋) 레이어
        self.fc2 = nn.Linear(150, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, 1)

    def init_hidden(self, batch_size):
        # the first is the hidden h
        # the second is the cell  c
        if GPU_NUM:
            return autograd.Variable(torch.zeros(self.n_layers * self.n_directions, batch_size, self.max_length).cuda())
        else:
            return autograd.Variable(torch.zeros(self.n_layers * self.n_directions, batch_size, self.max_length))

    def forward(self, data: list):

        # 임베딩의 차원 변환을 위해 배치 사이즈를 구합니다.
        batch_size = len(data)

        self.rnn_hidden = self.init_hidden(batch_size)

        # list로 받은 데이터를 torch Variable로 변환합니다.
        data_in_torch = Variable(torch.from_numpy(np.array(data)).long())

        # 만약 gpu를 사용중이라면, 데이터를 gpu 메모리로 보냅니다.
        if GPU_NUM:
            data_in_torch = data_in_torch.cuda()

        embeds = self.embeddings(data_in_torch)

        rnn_out, self.rnn_hidden = self.rnn(embeds, self.rnn_hidden)

        cnn = self.cnn_res(self.rnn_hidden.transpose(0, 1).transpose(1, 2))

        hidden1 = self.fc1(cnn.view(batch_size, -1))

        hidden2 = self.fc2(hidden1)
        hidden3 = self.fc3(hidden2)
        hidden4 = self.fc4(hidden3)
        sigmoid = torch.sigmoid(hidden4)
        output = sigmoid * 9 + 1
        return output


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--pause', type=int, default=0)
    args.add_argument('--iteration', type=str, default='0')

    # User options
    args.add_argument('--output', type=int, default=1)
    args.add_argument('--epochs', type=int, default=50)
    args.add_argument('--batch', type=int, default=400)
    args.add_argument('--strmaxlen', type=int, default=200)
    # args.add_argument('--embedding', type=int, default=8)
    args.add_argument('--embedding', type=int, default=32)
    config = args.parse_args()

    if not HAS_DATASET and not IS_ON_NSML:  # It is not running on nsml
        DATASET_PATH = '../sample_data/movie_review/'

    model = JJuModel(config.embedding, config.strmaxlen)
    if GPU_NUM:
        model = model.cuda()

    # DONOTCHANGE: Reserved for nsml use
    bind_model(model, config)

    optimizer = optim.Adam(model.parameters(), lr=0.0004)
    criterion = nn.MSELoss()
    # criterion = nn.CrossEntropyLoss()

    # DONOTCHANGE: They are reserved for nsml
    if config.pause:
        nsml.paused(scope=locals())

    # 학습 모드일 때 사용합니다. (기본값)
    if config.mode == 'train':
        # 데이터를 로드합니다.
        dataset = MovieReviewDataset(DATASET_PATH, config.strmaxlen)
        train_loader = DataLoader(dataset=dataset,
                                  batch_size=config.batch,
                                  shuffle=True,
                                  collate_fn=collate_fn,
                                  num_workers=2)
        total_batch = len(train_loader)
        # epoch마다 학습을 수행합니다.
        for epoch in range(config.epochs):
            avg_loss = 0.0

            # example data 가 batch_size 보다 크면 loop으로 더 돌림
            for i, (data, labels) in enumerate(train_loader):
                predictions = model(data)

                label_vars = Variable(torch.from_numpy(labels))
                if GPU_NUM:
                    label_vars = label_vars.cuda()

                loss = criterion(predictions, label_vars)
                if GPU_NUM:
                    loss = loss.cuda()

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                print('Batch : ', i + 1, '/', total_batch,
                      ', MSE in this minibatch: ', loss.data[0])
                avg_loss += loss.data[0]

            print('epoch:', epoch, ' train_loss:', float(avg_loss / total_batch))
            # nsml ps, 혹은 웹 상의 텐서보드에 나타나는 값을 리포트하는 함수입니다.
            #
            nsml.report(summary=True, scope=locals(), epoch=epoch, epoch_total=config.epochs,
                        train__loss=float(avg_loss / total_batch), step=epoch)
            # DONOTCHANGE (You can decide how often you want to save the model)
            nsml.save(epoch)

    # 로컬 테스트 모드일때 사용합니다
    # 결과가 아래와 같이 나온다면, nsml submit을 통해서 제출할 수 있습니다.
    # [(0.0, 9.045), (0.0, 5.91), ... ]
    elif config.mode == 'test_local':
        with open(os.path.join(DATASET_PATH, 'train/train_data'), 'rt', encoding='utf-8') as f:
            reviews = f.readlines()
        res = nsml.infer(reviews)
        print(res)
