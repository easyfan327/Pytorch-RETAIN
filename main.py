# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import pickle as pickle
import random
from sklearn.metrics import roc_auc_score


class RetainNN(nn.Module):
    def __init__(self, params: dict):
        super(RetainNN, self).__init__()
        """
        num_embeddings(int): size of the dictionary of embeddings
        embedding_dim(int) the size of each embedding vector
        """
        #self.emb_layer = nn.Embedding(num_embeddings=params["num_embeddings"], embedding_dim=params["embedding_dim"])
        self.emb_layer = nn.Linear(in_features=params["num_embeddings"], out_features=params["embedding_dim"])
        self.dropout = nn.Dropout(params["dropout_p"])
        self.variable_level_rnn = nn.GRU(params["var_rnn_hidden_size"], params["var_rnn_output_size"])
        self.visit_level_rnn = nn.GRU(params["visit_rnn_hidden_size"], params["visit_rnn_output_size"])
        self.variable_level_attention = nn.Linear(params["var_rnn_output_size"], params["var_attn_output_size"])
        self.visit_level_attention = nn.Linear(params["visit_rnn_output_size"], params["visit_attn_output_size"])
        self.output_dropout = nn.Dropout(params["output_dropout_p"])
        self.output_layer = nn.Linear(params["embedding_output_size"], params["num_class"])

        self.var_hidden_size = params["var_rnn_hidden_size"]

        self.visit_hidden_size = params["visit_rnn_hidden_size"]

        self.n_samples = params["batch_size"]
        self.reverse_rnn_feeding = params["reverse_rnn_feeding"]


    def forward(self, input, var_rnn_hidden, visit_rnn_hidden):
        """

        :param input:
        :param var_rnn_hidden:
        :param visit_rnn_hidden:
        :return:
        """
        # emb_layer: input(*): LongTensor of arbitrary shape containing the indices to extract
        # emb_layer: output(*,H): where * is the input shape and H = embedding_dim
        # print("size of input:")
        # print(input.shape)
        v = self.emb_layer(input)
        # print("size of v:")
        # print(v.shape)
        v = self.dropout(v)

        # GRU:
        # input of shape (seq_len, batch, input_size)
        # seq_len: visit_seq_len
        # batch: batch_size
        # input_size: embedding dimension
        #
        # h_0 of shape (num_layers*num_directions, batch, hidden_size)
        # num_layers(1)*num_directions(1)
        # batch: batch_size
        # hidden_size:
        if self.reverse_rnn_feeding:
            visit_rnn_output, visit_rnn_hidden = self.visit_level_rnn(torch.flip(v, [0]), visit_rnn_hidden)
            alpha = self.visit_level_attention(torch.flip(visit_rnn_output, [0]))
        else:
            visit_rnn_output, visit_rnn_hidden = self.visit_level_rnn(v, visit_rnn_hidden)
            alpha = self.visit_level_attention(visit_rnn_output)
        visit_attn_w = F.softmax(alpha, dim=0)

        if self.reverse_rnn_feeding:
            var_rnn_output, var_rnn_hidden = self.variable_level_rnn(torch.flip(v, [0]), var_rnn_hidden)
            beta = self.variable_level_attention(torch.flip(var_rnn_output, [0]))
        else:
            var_rnn_output, var_rnn_hidden = self.variable_level_rnn(v, var_rnn_hidden)
            beta = self.variable_level_attention(var_rnn_output)
        var_attn_w = torch.tanh(beta)

        # print("beta attn:")
        # print(var_attn_w.shape)
        # '*' = hadamard product (element-wise product)
        attn_w = visit_attn_w * var_attn_w
        c = torch.sum(attn_w * v, dim=0)
        # print("context:")
        # print(c.shape)

        c = self.output_dropout(c)
        #print("context:")
        #print(c.shape)
        output = self.output_layer(c)
        #print("output:")
        #print(output.shape)
        output = F.softmax(output, dim=1)
        # print("output:")
        # print(output.shape)

        return output, var_rnn_hidden, visit_rnn_hidden

    def init_hidden(self, current_batch_size):
        return torch.zeros(current_batch_size, self.var_hidden_size).unsqueeze(0), torch.zeros(current_batch_size, self.visit_hidden_size).unsqueeze(0)


def init_params(params: dict):
    # embedding matrix
    params["num_embeddings"] = 942
    params["embedding_dim"] = 128
    # embedding dropout
    params["dropout_p"] = 0.5
    # Alpha
    params["visit_rnn_hidden_size"] = 128
    params["visit_rnn_output_size"] = 128
    params["visit_attn_output_size"] = 1
    # Beta
    params["var_rnn_hidden_size"] = 128
    params["var_rnn_output_size"] = 128
    params["var_attn_output_size"] = 128

    params["embedding_output_size"] = 128
    params["num_class"] = 2
    params["output_dropout_p"] = 0.8

    params["batch_size"] = 100
    params["n_epoches"] = 100

    params["test_ratio"] = 0.2
    params["validation_ratio"] = 0.1
    params["sequence_file"] = "retaindataset.3digitICD9.seqs"
    params["label_file"] = "retaindataset.morts"

    params["reverse_rnn_feeding"] = True

    # TODO: Customized Loss
    # TODO: REF: https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
    params["customized_loss"] = True

def padMatrixWithoutTime(seqs, options):
    lengths = np.array([len(seq) for seq in seqs]).astype('int32')
    n_samples = len(seqs)
    maxlen = np.max(lengths)

    x = np.zeros((maxlen, n_samples, options['num_embeddings']))
    for idx, seq in enumerate(seqs):
        for xvec, subseq in zip(x[:, idx, :], seq):
            xvec[subseq] = 1.

    return x, lengths


def init_data(params: dict):
    sequences = np.array(pickle.load(open(params["sequence_file"], 'rb')))
    labels = np.array(pickle.load(open(params["label_file"], 'rb')))

    data_size = len(labels)
    ind = np.random.permutation(data_size)

    test_size = int(params["test_ratio"] * data_size)
    validation_size = int(params["validation_ratio"] * data_size)

    test_indices = ind[:test_size]
    valid_indices = ind[test_size:test_size + validation_size]
    train_indices = ind[test_size + validation_size:]

    train_set_x = sequences[train_indices]
    train_set_y = labels[train_indices]
    test_set_x = sequences[test_indices]
    test_set_y = labels[test_indices]
    valid_set_x = sequences[valid_indices]
    valid_set_y = labels[valid_indices]

    # setx_tensor = torch.from_numpy(xbpad)
    # sety_tensor = torch.from_numpy(train_set_y)

    # train_ds = TensorDataset(torch.from_numpy(train_set_x.values), torch.from_numpy(train_set_y.values))
    # train_dl = DataLoader(train_ds, batch_size=params["batch_size"])


    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    train_sorted_index = len_argsort(train_set_x)
    train_set_x = [train_set_x[i] for i in train_sorted_index]
    train_set_y = [train_set_y[i] for i in train_sorted_index]

    valid_sorted_index = len_argsort(valid_set_x)
    valid_set_x = [valid_set_x[i] for i in valid_sorted_index]
    valid_set_y = [valid_set_y[i] for i in valid_sorted_index]

    test_sorted_index = len_argsort(test_set_x)
    test_set_x = [test_set_x[i] for i in test_sorted_index]
    test_set_y = [test_set_y[i] for i in test_sorted_index]

    return train_set_x, train_set_y, valid_set_x, valid_set_y, test_set_x, test_set_y


if __name__ == "__main__":

    parameters = dict()
    init_params(parameters)

    train_set_x, train_set_y, valid_set_x, valid_set_y, test_set_x, test_set_y = init_data(parameters)

    model = RetainNN(params=parameters)
    for name, parm in model.named_parameters():
        print(name, parm)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0, rho=0.9, eps=1e-6, weight_decay=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()

    n_batches = int(np.ceil(float(len(train_set_y)) / float(parameters["batch_size"])))
    for epoch in range(parameters["n_epoches"]):
        model.train()
        loss_vector = torch.zeros(n_batches, dtype=torch.float)
        for index in random.sample(range(n_batches), n_batches):
            xb = train_set_x[index*parameters["batch_size"]:(index+1)*parameters["batch_size"]]
            yb = train_set_y[index*parameters["batch_size"]:(index+1)*parameters["batch_size"]]
            xbpad, xbpad_lengths = padMatrixWithoutTime(seqs=xb, options=parameters)
            xbpadtensor = torch.from_numpy(xbpad).float()
            ybtensor = torch.from_numpy(np.array(yb)).long()
            #print(xbpadtensor.shape)
            var_rnn_hidden_init, visit_rnn_hidden_init = model.init_hidden(xbpadtensor.shape[1])

            pred, var_rnn_hidden_init, visit_rnn_hidden_init = model(xbpadtensor, var_rnn_hidden_init, visit_rnn_hidden_init)
            pred = pred.squeeze(1)
            # print("pred:")
            # print(pred.shape)
            # print(pred.data)
            # print("ybtensor:")
            # print(ybtensor.shape)

            loss = loss_fn(pred, ybtensor)
            loss.backward()
            loss_vector[index] = loss
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        x, x_length = padMatrixWithoutTime(seqs=valid_set_x, options=parameters)
        x = torch.from_numpy(x).float()
        y_true = torch.from_numpy(np.array(valid_set_y)).long()
        var_rnn_hidden_init, visit_rnn_hidden_init = model.init_hidden(x.shape[1])
        y_hat, var_rnn_hidden_init, visit_rnn_hidden_init = model(x, var_rnn_hidden_init, visit_rnn_hidden_init)
        y_true = y_true.unsqueeze(1)
        y_true_oh = torch.zeros(y_hat.shape).scatter_(1, y_true, 1)
        auc = roc_auc_score(y_true=y_true_oh.detach().numpy(), y_score=y_hat.detach().numpy())

        print("epoch:{} train-loss:{} valid-auc:{}".format(epoch, torch.mean(loss_vector), auc))

    model.eval()
    x, x_length = padMatrixWithoutTime(seqs=test_set_x, options=parameters)
    x = torch.from_numpy(x).float()
    y_true = torch.from_numpy(np.array(test_set_y)).long()
    var_rnn_hidden_init, visit_rnn_hidden_init = model.init_hidden(x.shape[1])
    y_hat, var_rnn_hidden_init, visit_rnn_hidden_init = model(x, var_rnn_hidden_init, visit_rnn_hidden_init)
    y_true = y_true.unsqueeze(1)
    y_true_oh = torch.zeros(y_hat.shape).scatter_(1, y_true, 1)
    auc = roc_auc_score(y_true=y_true_oh.detach().numpy(), y_score=y_hat.detach().numpy())

    print("test auc:{}".format(auc))


        