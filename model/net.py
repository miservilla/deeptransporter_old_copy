"""Defines the neural network, loss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lifelines.utils import concordance_index
import math
from sklearn import metrics as smetrics
import r2python
from scipy.stats.stats import spearmanr
from past.builtins import basestring
import copy
# 3x3 convolution


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def conv1d(in_channels, out_channels, kernel_size=5, stride=2):
    padding_size = int((kernel_size - 1) / 2)
    # print(padding_size)
    # print(in_channels)
    return nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                     stride=stride, padding=padding_size, bias=False)

# Residual block


class ConvolutionBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=5, stride=2, dropout_rate=0):
        super(ConvolutionBlock, self).__init__()
        self.dropout_rate = dropout_rate
        # print(in_channels)
        # print(out_channels)
        self.conv1 = conv1d(in_channels, out_channels, kernel_size, stride=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        # self.relu = nn.Residual(inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        residual = x
        out = self.conv1(residual)
        out = self.bn1(out)
        out = self.relu(out)
        out = F.dropout(out, p=self.dropout_rate, training=self.training)
        out = self.maxpool(out)
        return out


# FC block

class FullConnectedBlock(nn.Module):
    ''' Implmenent are resdual style fully connected layer"
    '''

    def __init__(self, in_channels, out_channels, dropout_rate, use_residual=True):
        super(FullConnectedBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout_rate = dropout_rate
        self.use_residual = use_residual
        #self.norm1 = nn.LayerNorm(in_channels)
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.norm2 = nn.LayerNorm(out_channels)

    def forward(self, x):
        residual = x
        out = x
        #out = self.norm1(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = F.dropout(out, p=self.dropout_rate, training=self.training)
        out = self.norm2(out)
        if self.use_residual and self.in_channels == self.out_channels:
            out += residual
        return out


class FullConnectedBlock_v0(nn.Module):

    def __init__(self, in_channels, out_channels, dropout_rate):
        super(FullConnectedBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fc = nn.Linear(in_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout_rate = dropout_rate
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.fc(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = F.dropout(out, p=self.dropout_rate, training=self.training)
        return out


# EmbeddingNet
class EmbeddingNet(nn.Module):
    '''
    encoder
    '''

    def __init__(self, params, block=ConvolutionBlock):
        super(EmbeddingNet, self).__init__()
        self.params = copy.deepcopy(params)
        self.in_channels = 1
        if len(self.params.kernel_sizes) == 1:
            self.params.kernel_sizes = [self.params.kernel_sizes[0]] * len(self.params.out_channels_list)
        if len(self.params.strides) == 1:
            self.strides = [self.params.strides[0]] * len(self.params.out_channels_list)
        self.output_size = self.params.input_size
        print("initial output size")
        print(self.output_size)
        self.layers_block1 = self.make_layers(block, self.params.out_channels_list, kernel_sizes=self.params.kernel_sizes, strides=self.params.strides, dropout_rate=self.params.dropout_rate)
        # output_size is updated
        print("final convolution layer output size")
        if(len(self.params.out_channels_list) > 0):
            self.fc_output_size = self.output_size * self.params.out_channels_list[-1]
        else:
            self.fc_output_size = self.params.input_size

        print("initial fully connected size")
        print(self.fc_output_size)
        # self.params.FC_size_list.append(self.params.embedding_size)
        self.FC_block1 = self.make_layers_FC(
            FullConnectedBlock, self.params.FC_size_list, self.params.dropout_rate)
        print("Embedding size")
        self.norm = nn.LayerNorm(self.fc_output_size)
        self.fc3 = nn.Linear(self.fc_output_size, self.params.embedding_size)

    def make_layers_FC(self, block, FC_size_list, dropout_rate):
        layers = []
        num_layers = len(FC_size_list)
        for i in range(0, num_layers):
            layers.append(block(self.fc_output_size, FC_size_list[
                          i], dropout_rate))
            self.fc_output_size = FC_size_list[i]
            print(self.fc_output_size)
        return nn.Sequential(*layers)

    def make_layers(self, block, out_channels_list, kernel_sizes, strides, dropout_rate):
        layers = []
        num_layers = len(out_channels_list)
        for i in range(0, num_layers):
            layers.append(block(self.in_channels, out_channels_list[
                          i], kernel_sizes[i], stride=strides[i], dropout_rate=dropout_rate))
            self.in_channels = out_channels_list[i]
            padding_size = (kernel_sizes[i] - 1) / 2
            convolution_stride = 1

            self.output_size = int(((self.output_size + 2 * padding_size - kernel_sizes[
                                   i]) / convolution_stride) + 1)  # convolution layer output_size
            # maxpool output_size
            self.output_size = int((
                (self.output_size - kernel_sizes[i]) / strides[i]) + 1)
            print(self.output_size)
        return nn.Sequential(*layers)

    def forward(self, x):
        # reshape numbatch * num_dim to numbatch * num_in_channel * num_dim
        out = x.view(x.size(0), 1, -1)
        out = self.layers_block1(out)
        temp = out.size()
        out = out.view(out.size(0), -1)
        out = self.FC_block1(out)
        out = self.norm(out)
        out = self.fc3(out)
        return out


class outputLayer(nn.Module):
    '''
    decoder
    '''

    def __init__(self, params):
        super(outputLayer, self).__init__()
        self.params = copy.deepcopy(params)
        # change internal params to adapt encoder for internal_layers of decoder
        self.params.input_size = params.embedding_size
        self.survival_len, self.cont_len, self.bin_len = int(len(params.survival_indices) / 2), len(params.continuous_phenotype_indices), len(params.binary_phenotype_indices)

        self.params.embedding_size = self.survival_len + self.cont_len + self.bin_len
        self.params.out_channels_list = []  # no convolution layer
        self.params.FC_size_list = params.decoder_FC_size_list
        self.internal_layers = EmbeddingNet(self.params)
        self.dense1_bn = nn.BatchNorm1d(self.survival_len)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.internal_layers(x)
        # tracer()
        surv_out = out[:, :self.survival_len]
        if self.survival_len > 0:
            surv_out = self.dense1_bn(surv_out)
        out = torch.cat((
            surv_out,
            out[:, self.survival_len:(self.survival_len + self.cont_len)],
            self.sigmoid(out[:, (self.survival_len + self.cont_len):])
        ), 1)
        return out


def feature_attention(attn_mat, embedding, dropout=None):
    "perform matrix multiplication "

    bs = attn_mat.size(0)
    embedding = embedding.view([bs, 1, embedding.shape[1]])
    output = torch.matmul(embedding, attn_mat)
    output = output.view([bs, output.shape[2]])
    return output


class AttentionEncoder(nn.Module):
    "Attention Encoder for deepImmune produce a matrix that  could be multiplied with embedding"

    def __init__(self, params):
        super(AttentionEncoder, self).__init__()
        self.params = copy.deepcopy(params)
        self.output_size = params.embedding_size
        # change internal params to adapt encoder for internal_layers of decoder
        self.params.input_size = params.attention_input_size

        self.params.embedding_size = self.output_size * self.output_size
        self.params.out_channels_list = []  # no convolution layer
        self.params.FC_size_list = params.attention_FC_size_list
        self.internal_layers = EmbeddingNet(self.params)
        # self.softmaxs = clones(nn.Softmax(dim=1), self.output_size)

    def forward(self, x):
        out = self.internal_layers(x)
        # tracer()
        bs = out.size(0)
        out = out.view(bs, self.output_size, self.output_size)
        out_list = [nn.Softmax(dim=1)(out[inx, :, :])
                    for inx in range(bs)]
        out = torch.stack(out_list)
        out = out - 0.5  # for range [-0.5 0.5]
        return out


class outputLayer_simple(nn.Module):

    def __init__(self, params):
        super(outputLayer_simple, self).__init__()
        self.linear_output_size = linear_output_size
        self.binary_output_size = binary_output_size
        self.linear1 = nn.Linear(embedding_size, self.linear_output_size + self.binary_output_size)
        self.dense1_bn = nn.BatchNorm1d(1)
        self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.LogSoftmax()

    def forward(self, x):
        linear1_out = self.linear1(x)
        survival_out = self.dense1_bn(linear1_out[:, 0:1])
        binary_output = self.sigmoid(linear1_out[:, self.linear_output_size:])
        out = torch.cat((survival_out, linear1_out[:, 1:self.linear_output_size], binary_output), 1)

        return out


class outputLayer_old(nn.Module):

    def __init__(self, embedding_size, linear_output_size=32, binary_output_size=32):
        super(outputLayer, self).__init__()
        self.linear_output_size = linear_output_size
        self.binary_output_size = binary_output_size
        if linear_output_size > 0:
            self.linear1 = nn.Linear(embedding_size, linear_output_size)
            self.dense1_bn = nn.BatchNorm1d(1)
        if binary_output_size > 0:
            self.linear2 = nn.Linear(embedding_size, binary_output_size)
        self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.LogSoftmax()

    def forward(self, x):
        if(self.linear_output_size > 0):
            linear1_out = self.linear1(x)
            # print(linear1_out.shape)
            # add a batch normalization for survival prediction
            survival_out = self.dense1_bn(linear1_out[:, 0:1])

        if(self.binary_output_size > 0):
            linear2_out = self.linear2(x)
            # binary_output = self.sigmoid(linear2_out)
            binary_output = linear2_out  # sigmoid not required with BCEWithLogitsLoss
            binary_output = self.sigmoid(binary_output)

        if(self.linear_output_size == 0):
            out = binary_output
        elif(self.binary_output_size == 0):
            out = torch.cat((survival_out, linear1_out[:, 1:]), 1)
        else:
            out = torch.cat((survival_out, linear1_out[:, 1:], binary_output), 1)
        # print(out.shape)
        return out


class outputLayer_old(nn.Module):

    def __init__(self, embedding_size, linear_output_size=32, binary_output_size=32):
        super(outputLayer, self).__init__()
        self.linear_output_size = linear_output_size
        self.binary_output_size = binary_output_size
        self.linear1 = nn.Linear(embedding_size, 1)
        self.linear2 = nn.Linear(1, self.linear_output_size + self.binary_output_size)

        self.dense1_bn = nn.BatchNorm1d(1)

        self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.LogSoftmax()

    def forward(self, x):
        linear1_out = self.linear1(x)
        linear2_out = self.linear2(linear1_out)
        survival_out = self.dense1_bn(linear2_out[:, 0:1])
        binary_output = self.sigmoid(linear2_out[:, self.linear_output_size:])
        out = torch.cat((survival_out, linear2_out[:, 1:self.linear_output_size], binary_output), 1)

        return out


class outputLayer_incomplete(nn.Module):

    def __init__(self, params):
        super(outputLayer, self).__init__()
        self.params = params
        self.survival_size = int(len(params.survival_indices) / 2)
        self.joint_size = len(params.joint_indices)
        self.linear_output_size = len(params.continuous_phenotype_indices) + self.survival_size
        self.binary_output_size = len(params.binary_phenotype_indices)
        self.total_output_size = self.linear_output_size + self.total_output_size
        self.joint_order_inx = get_joint_order()
        self.survival_order_inx = get_survival_order()
        if self.joint_size > 0:
            self.truncated_output_size = self.total_output_size - self.joint_size + 1
        else:
            self.truncated_output_size = self.total_output_size

        # if total_output_size > 0:
        self.linear1 = nn.Linear(params.embedding_size, total_output_size)
        self.dense1_bn = nn.BatchNorm1d(1)
        # if binary_output_size > 0:
        # self.linear2 = nn.Linear(params.embedding_size, binary_output_size)

        if joint_size > 0:
            self.linear_fork = nn.Linear(1, self.joint_size)
        self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.LogSoftmax()

    def get_joint_order():
        order_inx = np.array(range(total_output_size))
        if self.joint_size > 0:
            start_inx = 1
            for ii in range(total_output_size):
                if any(self.params.joint_indices == ii):
                    order_inx[ii] = 0
                else:
                    order_inx[ii] = start_inx
                    start_inx = start_inx + 1
        return order_inx

    def get_survival_order():
        order_inx = np.array(range(total_output_size))
        if self.joint_size > 0:
            start_inx = 1
            for ii in range(total_output_size):
                if any(self.params.joint_indices == ii):
                    order_inx[ii] = 0
                else:
                    order_inx[ii] = start_inx
                    start_inx = start_inx + 1
        return order_inx

    def forward(self, x):
        # the first output is joint if present
        out1 = self.linear1(x)
        if self.joint_size > 0:
            joint_output = out1[:, 0]
            joint_output = linear_fork(joint_output)
            out2 = torch.cat([joint_output, out1], 1)
            out2 = out2[:, self.joint_order_inx]
        else:
            out2 = out1

        if self.survival_size > 0:
            survival_out = self.dense1_bn(out2[:, 0:self.survival_size])
            out2 = torch.cat(survival_out, out2[:, self.survival_size:])

        if self.binary_size > 0:
            binary_out = self.sigmoid(out2[:, self.total_output_size - self.binary_size:])
            out2 = torch.cat(out2[:, :self.total_output_size - self.binary_size], binary_out)

        out = out2[:, self.order_inx]

        return out


class outputLayer_incomplete(nn.Module):

    def __init__(self, params):
        super(outputLayer, self).__init__()
        self.params = params
        self.survival_size = int(len(params.survival_indices) / 2)
        self.joint_size = len(params.joint_indices)
        self.linear_output_size = len(params.continuous_phenotype_indices) + self.survival_size
        self.binary_output_size = len(params.binary_phenotype_indices)
        self.total_output_size = self.linear_output_size + self.total_output_size
        self.joint_order_inx = get_joint_order()
        self.survival_order_inx = get_survival_order()
        if self.joint_size > 0:
            self.truncated_output_size = self.total_output_size - self.joint_size + 1
        else:
            self.truncated_output_size = self.total_output_size

        # if total_output_size > 0:
        self.linear1 = nn.Linear(params.embedding_size, total_output_size)
        self.dense1_bn = nn.BatchNorm1d(1)
        # if binary_output_size > 0:
        # self.linear2 = nn.Linear(params.embedding_size, binary_output_size)

        if joint_size > 0:
            self.linear_fork = nn.Linear(1, self.joint_size)
        self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.LogSoftmax()

    def get_joint_order():
        order_inx = np.array(range(total_output_size))
        if self.joint_size > 0:
            start_inx = 1
            for ii in range(total_output_size):
                if any(self.params.joint_indices == ii):
                    order_inx[ii] = 0
                else:
                    order_inx[ii] = start_inx
                    start_inx = start_inx + 1
        return order_inx

    def get_survival_order():
        order_inx = np.array(range(total_output_size))
        if self.joint_size > 0:
            start_inx = 1
            for ii in range(total_output_size):
                if any(self.params.joint_indices == ii):
                    order_inx[ii] = 0
                else:
                    order_inx[ii] = start_inx
                    start_inx = start_inx + 1
        return order_inx

    def forward(self, x):
        # the first output is joint if present
        out1 = self.linear1(x)
        if self.joint_size > 0:
            joint_output = out1[:, 0]
            joint_output = linear_fork(joint_output)
            out2 = torch.cat([joint_output, out1], 1)
            out2 = out2[:, self.joint_order_inx]
        else:
            out2 = out1

        if self.survival_size > 0:
            survival_out = self.dense1_bn(out2[:, 0:self.survival_size])
            out2 = torch.cat(survival_out, out2[:, self.survival_size:])

        if self.binary_size > 0:
            binary_out = self.sigmoid(out2[:, self.total_output_size - self.binary_size:])
            out2 = torch.cat(out2[:, :self.total_output_size - self.binary_size], binary_out)

        out = out2[:, self.order_inx]

        return out


# tempNet
class tempNet(nn.Module):

    def __init__(self, block, input_size, out_channels_list=[32, 32, 32],
                 embedding_size=32, blocks=1, kernel_sizes=[5], strides=[2]):
        super(tempNet, self).__init__()
        self.in_channels = 1
        self.block = block
        if len(kernel_sizes) == 1:
            self.kernel_sizes = [kernel_sizes] * len(out_channels_list)
        if len(strides) == 1:
            self.strides = [strides] * len(out_channels_list)

        self.lay1 = self.make_layer(block, out_channels_list[
                                    0], kernel_sizes=self.kernel_sizes, strides=self.strides[0])
        # self.embedding_size = embedding_size
        # self.blocks = blocks
        # self.kernel_size = kernel_size
        # self.strides = strides
       # (1, 64, 5, stride=2

    def make_layer(self, block, out_channels, kernel_sizes, strides):
        layers = []
        layers.append(block(self.in_channels, out_channels,
                            kernel_sizes[0], stride))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(
                block(out_channels, out_channels, kernel_size, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = x
        # out = self.make_layer(self.block,  self.out_channels_list[0], kernel_size = self.kernel_size, stride = self.strides[0], blocks=self.blocks)(out)
        out = self.lay1(out)
        print("here")
        print(out)
        return out


# EmbeddingNet
class EmbeddingNet_FC(nn.Module):

    def __init__(self, block, input_size, out_channels_list,
                 embedding_size=32,
                 dropout_rate=0.1):
        super(EmbeddingNet_FC, self).__init__()
        self.in_channels = input_size
        self.output_size = input_size
        self.dropout_rate = dropout_rate
        print("initial output size")
        print(self.output_size)
        self.layers_block1 = self.make_layers_FC(
            block, out_channels_list, dropout_rate)
        # output_size is updated
        print("final output size")
        print(self.output_size)
        self.fc3 = nn.Linear(self.in_channels, embedding_size)

    def make_layers_FC(self, block, out_channels_list, dropout_rate):
        layers = []
        num_layers = len(out_channels_list)
        for i in range(0, num_layers):
            layers.append(block(self.in_channels, out_channels_list[
                          i], dropout_rate))
            self.in_channels = out_channels_list[i]
        return nn.Sequential(*layers)

    def forward(self, x):
        # reshape numbatch * num_dim to numbatch * num_in_channel * num_dim
        # out = x.view(x.size(0), 1, -1)
        out = x
        out = self.layers_block1(out)
        out = self.fc3(out)
        return out


class embedding_conv(nn.Module):

    def __init__(self, input_size, num_classes=1):
        super(ConvNet1D, self).__init__()
        layer1_size = math.floor(input_size / 2)
        output_size = math.floor(layer1_size / 2)
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        # self.output_size = math_ceiling
        self.fc = nn.Linear(output_size * 32, num_classes)
        self.dense1_bn = nn.BatchNorm1d(num_classes)


class oneLayerNet(nn.Module):

    def __init__(self, input_size, num_classes):
        super(oneLayerNet, self).__init__()

    def forward(self, x):
        out = nn.Linear(self.input_size, self.num_classes)(x)
        out = nn.ReLU()(out)
        out = nn.BatchNorm1d(self.num_classes)(out)
        return out


class ImmuneEmbed(nn.Module):

    def __init__(self, input_size, hidden_layers):
        super(ImmuneEmbed, self).__init__()

    def forward(self, x):
        out = nn.Linear(self.input_size, self.hidden_layer[0])(x)
        out = nn.ReLU()(out)
        out = nn.BatchNorm1d(self.hidden_layer[0])(out)
        for layer_size in self.hidden_layers:
            out = nn.Linear(layer_size)(out)
            out = nn.ReLU()(out)
            out = nn.BatchNorm1d(layer_size)(out)
        return out


# Fully connected neural network with one hidden layer

class NeuralNet(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes=1):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        # batchnorm takes both channels or hidden size as input
        self.dense1_bn = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dense2_bn = nn.BatchNorm1d(num_classes)
        # self.dense2_bn = nn.BatchNorm1d(num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dense1_bn(out)
        out = self.fc2(out)
        # print(out.shape)
        out = self.dense2_bn(out)
        return out

# 1D Convolutional neural network (two convolutional layers)


class ConvNet1D(nn.Module):

    def __init__(self, input_size, num_classes=1):
        super(ConvNet1D, self).__init__()
        layer1_size = math.floor(input_size / 2)
        output_size = math.floor(layer1_size / 2)
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        # self.output_size = math_ceiling
        self.fc = nn.Linear(output_size * 32, num_classes)
        self.dense1_bn = nn.BatchNorm1d(num_classes)

    def forward(self, x):
        gene_len, batch_size = x.size()  # x is gene_len x batch_size
        # print(x.shape)
        # Turn (gene_len x batch_size) into (batch_size x input_size x gene_len) for CNN, where input_size = 1
        # x = x.transpose(0, 1)
        # print(x.shape)
        x = x.reshape(batch_size, 1, gene_len)
        # print(x.shape)
        out = self.layer1(x)
        # print(out.shape)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.dense1_bn(out)
        return out

# Convolutional neural network (two convolutional layers)


class ConvNet(nn.Module):

    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


def loss_fn(outputs, labels):
    """
    Compute the cross entropy loss given outputs and labels.

    Args:
        outputs: (Variable) dimension batch_size x 6 - output of the model
        labels: (Variable) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]

    Returns:
        loss (Variable): cross entropy loss for all images in the batch

    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """
    num_examples = outputs.size()[0]
    return -torch.sum(outputs[range(num_examples), labels]) / num_examples


def checkNaNInf(x):
    torch.isnan(x).any() or x.eq(
        float('inf')).any() or x.eq(float('-inf')).any()


def isfinite(x):
    """
    Quick pytorch test that there are no nan's or infs.

    note: torch now has torch.isnan
    url: https://gist.github.com/wassname/df8bc03e60f81ff081e1895aabe1f519
    """
    not_inf = ((x + 1) != x)
    not_nan = (x == x)
    return not_inf & not_nan


def isfinite_list(x):
    """
    Quick pytorch test that there are no nan's or infs.

    note: torch now has torch.isnan
    url: https://gist.github.com/wassname/df8bc03e60f81ff081e1895aabe1f519
    """
    out = 1
    if x is not None:
        not_inf = ((x + 1) != x)
        not_nan = (x == x)
        out = not_inf & not_nan
        out = out.all()
    return out


def negative_log_partial_likelihood(survival, risk, debug=False):
    """Return the negative log-partial likelihood of the prediction
    y_true contains the survival time
    risk is the risk output from the neural network
    censor is the vector of inputs that are censored
    regularization is the regularization constant (not used currently in model)

    Uses the torch backend to perform calculations

    Sorts the surv_time by sorted reverse time
    """

    # calculate negative log likelihood from estimated risk\

    # temp = survival
    # temp[:, 1] = risk

    # print(torch.stack([survival[:, 0], risk]))
    _, idx = torch.sort(survival[:, 0], descending=True)
    censor = survival[idx, 1]
    risk = risk[idx]
    # print(temp[idx, :])
    # print(survival[idx, :])
    epsilon = 0.00001
    max_value = 10
    alpha = 0.1
    risk = torch.reshape(risk, [-1])  # flatten
    # risk = risk - torch.mean(risk)
    # risk[risk > max_value] = max_value
    hazard_ratio = torch.exp(risk)

    # cumsum on sorted surv time accounts for concordance
    log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0) + epsilon)
    log_risk = torch.reshape(log_risk, [-1])
    uncensored_likelihood = risk - log_risk

    # apply censor mask: 1 - dead, 0 - censor
    censored_likelihood = uncensored_likelihood * censor
    num_observed_events = torch.sum(censor)
    neg_likelihood = - torch.sum(censored_likelihood) / \
        num_observed_events
    # if(not isfinite(neg_likelihood) or debug):
    #     import ipdb
    #     ipdb.set_trace()

    # print(type(neg_likelihood))

    # if (neg_likelihood != neg_likelihood):
    #     print(neg_likelihood)
    #     print(censor[np.isnan(censor)])
    #     print(risk[np.isnan(risk)])
    #     raise ValueError("nan found")

    # return torch.exp(neg_likelihood)
    return neg_likelihood


def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.

    Args:
        outputs: (np.ndarray) dimension batch_size x 6 - log softmax output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]

    Returns: (float) accuracy in [0,1]
    """
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs == labels) / float(labels.size)


def negative_log_partial_likelihood_loss(risk, censor, debug=False):
    '''
    function to change the order of censor and risk
    '''

    return negative_log_partial_likelihood(censor, risk, debug)


def c_index(predicted_risk, survival):
    if survival is None:
        return 0
    # calculate the concordance index
    ci = np.nan  # just to know that concordance index cannot be estimated
    # print(r2python.cbind(np.reshape(predicted_risk, (-1, 1)), survival))

    na_inx = ~(np.isnan(survival[:, 0]) | np.isnan(survival[:, 1]) | np.isnan(predicted_risk))
    predicted_risk, survival = predicted_risk[na_inx], survival[na_inx]
    if len(predicted_risk) > 0 and sum(survival[:, 1] == 1) > 2:
        survival_time, censor = survival[:, 0], survival[:, 1]
        epsilon = 0.001
        partial_hazard = np.exp(-(predicted_risk + epsilon))
        censor = censor.astype(int)
        ci = concordance_index(survival_time, partial_hazard, censor)
    return ci


# maintain all metrics required in this dictionary- these are used in the
# training and evaluation loops
def calculate_auc(pred, y):
    if y is None:
        return 0
    # print(y)
    # print("pred")
    # print(pred)
    na_inx = ~(isnan(pred) | isnan(y))
    pred, y = pred[na_inx], y[na_inx]
    try:
        fpr, tpr, thresholds = smetrics.roc_curve(y, pred, pos_label=1)
        auc = smetrics.auc(fpr, tpr)
    except:
        auc = 0.5
    if isnan(auc):
        auc = 0.5
    return auc


def spearman_corr(pred, y):
    # import ipdb
    # ipdb.set_trace()
    try:
        out = spearmanr(pred, y, nan_policy='omit')
        out = out.correlation
    except:
        out = 0.

    return out


metrics = {
    'auc': calculate_auc,
    'c_index': c_index,
    'correlation': spearman_corr
    # could add more metrics such as accuracy for each token type
}


def isnan(x):
    return x != x


def max_na(xx):
    return (0 if len(xx) == 0 else max(xx))


def mean_na(xx):
    xx = np.array(xx)
    return np.mean(xx[~np.isnan(xx)])


def create_lossfns_mask(params):

    survival_output_size = int(len(params.survival_indices) / 2)
    linear_output_size = survival_output_size + len(params.continuous_phenotype_indices)
    binary_output_size = len(params.binary_phenotype_indices)
    params.loss_fns = [negative_log_partial_likelihood_loss] * survival_output_size + \
        [nn.MSELoss()] * len(params.continuous_phenotype_indices) + \
        [nn.BCELoss()] * len(params.binary_phenotype_indices)

    params.mask = np.concatenate([params.survival_indices, params.continuous_phenotype_indices, params.binary_phenotype_indices])

    return params


def regularized_loss(model, params, index="all"):

    loss_curr = 0.

    for name, parameters in model.named_parameters():
        if name.endswith('weight') and name.endswith('linear'):
            if index is "all":
                loss_curr = loss_curr + \
                    torch.norm(parameters.view(-1), 1) * params.l1_regularizer \
                    + torch.norm(parameters.view(-1), 2) * params.l2_regularizer
            else:
                loss_curr = loss_curr + \
                    torch.norm(parameters[index].view(-1), 1) * params.l1_regularizer \
                    + torch.norm(parameters[index].view(-1), 2) * params.l2_regularizer
    return loss_curr


def calculate_loss(labels, net_outputs, loss_fns):
    '''
    define loss function :
    '''

    # total_loss = torch.zeros(1)
    total_loss = 0.

    # ## survival output
    # survival = labels[:,0:2]
    # na_inx = ~( np.isnan(survival[:,1]) | nbasep.isnan(survival[:,1]))
    # survival, net_output = survival[na_inx,:], net_outputs[na_inx,0]
    # if(len(label) > 1 ):
    #     loss_curr = loss_fns[0](net_output, label)
    # else:
    #     loss_curr = 0

    # total_loss = total_loss + loss_curr

    # label, net_output, loss_fn = labels[:,i], net_outputs[:, i], loss_fns[i]

    len_fns = len(loss_fns)
    label_inx = 0

    for i in range(len_fns):
        net_output, loss_fn = net_outputs[:, i], loss_fns[i]
        # print(loss_fn)
        # print(loss_fn.__name__ is 'negative_log_partial_likelihood_loss')
        if hasattr(loss_fn, '__name__'):
            if loss_fn.__name__ is 'negative_log_partial_likelihood_loss':
                label = labels[:, label_inx:(label_inx + 2)]
                label_inx = label_inx + 2
                # print(label.shape)
                na_inx = ~(isnan(label[:, 0]) | isnan(label[:, 1]))
                label, net_output = label[na_inx, :], net_output[na_inx]
                # print(label)
        else:
            label = labels[:, label_inx]
            label_inx = label_inx + 1
            na_inx = ~isnan(label)
            label, net_output = label[na_inx], net_output[na_inx]

        if(len(label) > 1):
            # in case of survival label is censor; censored data assumed to
            # sorted based on patient event time.
            loss_curr = loss_fn(net_output, label)
        else:
            # loss_curr = torch.zeros(1)
            loss_curr = 0.
        # print(loss_curr.item())
        # manually tries to value response rate
        # if i >= len_fns - 2:
            # loss_curr = loss_curr * 10

        total_loss = total_loss + loss_curr

    return total_loss


def define_metrics(params):
    '''
    Create the metrics if not defined
    Format  ['name of. variable', 'metrics', 'index of predicted data', 'index of real data'. ]
    '''

    survival_output_size = int(len(params.survival_indices) / 2)
    if isinstance(params.metrics[0], basestring):
        params.metrics = range(len(params.loss_fns))

    label_inx = 0
    if isinstance(params.metrics[0], int):
        metrics = []
        metrics_type = ["c_index"] * survival_output_size + \
            ["correlation"] * len(params.continuous_phenotype_indices) + \
            ["auc"] * len(params.binary_phenotype_indices)
        for i in range(len(metrics_type)):
            # i is output_batch
            loss_fn = params.loss_fns[i]
            metrics.append([params.header[label_inx], metrics_type[i], i, label_inx])
            if hasattr(loss_fn, '__name__'):
                if loss_fn.__name__ is 'negative_log_partial_likelihood_loss':
                    label_inx = label_inx + 2
            else:
                label_inx = label_inx + 1
        params.metrics = [metrics[tt] for tt in params.metrics]
    # tracer()

    return params


def update_loss_parameters(labels, net_outputs, embedding_model, outputs, embedding_optimizer, outputs_optimizer, params, train_optimizer_mask=[1, 1]):
    '''
    define loss function and update parameters
    '''

    def update_parameters(loss, train_optimizer_mask, embedding_model, outputs):
        # clear previous gradients, compute gradients of all variables wrt loss
        # tracer()
        embedding_optimizer.zero_grad()
        outputs_optimizer.zero_grad()
        loss.backward()
        # xx = [p.grad for p in outputs.parameters()]
        # print(xx)
        # tracer()

        # if not all([isfinite(p.grad).all() for p in outputs.parameters()]):
        # tracer()
        # performs updates using calculated gradients
        if train_optimizer_mask[0]:
            embedding_optimizer.step()
        if train_optimizer_mask[1]:
            outputs_optimizer.step()

    total_loss = 0.
    loss_fns = params.loss_fns
    len_fns = len(loss_fns)

    label_inx = 0
    if sum(train_optimizer_mask):
        is_train = True
    else:
        is_train = False

    # loss_for_training = list(set(range(len(loss_fns))) - set(params.loss_excluded_from_training))
    # import ipdb
    # ipdb.set_trace()
    for i in range(len(loss_fns)):
        net_output, loss_fn = net_outputs[:, i], loss_fns[i]
        if hasattr(loss_fn, '__name__'):
            if loss_fn.__name__ is 'negative_log_partial_likelihood_loss':
                label = labels[:, label_inx:(label_inx + 2)]
                label_inx = label_inx + 2
                # print(label.shape)
                na_inx = ~(isnan(label[:, 0]) | isnan(label[:, 1]))
                label, net_output = label[na_inx, :], net_output[na_inx]
        else:
            label = labels[:, label_inx]
            label_inx = label_inx + 1
            na_inx = ~isnan(label)
            label, net_output = label[na_inx], net_output[na_inx]

        if(len(label) > 1) and not any(np.array(params.loss_excluded_from_training) == i):
            # in case of survival label is censor; censored data assumed to
            # sorted based on patient event time.
            loss_curr = loss_fn(net_output, label)
            loss_curr = loss_curr + regularized_loss(outputs, params, i)
        else:
            # loss_curr = torch.zeros(1)
            loss_curr = 0.

        if isnan(loss_curr):
            loss_curr = 0.

        # print(net_output)
        # print(loss_curr)
        # tracer()

        if is_train and params.pipeline_optimization and loss_curr != 0.:
            update_parameters(loss_curr, train_optimizer_mask, embedding_model, outputs)

        total_loss = total_loss + loss_curr

    # tracer()
    if is_train and not params.pipeline_optimization and total_loss != 0.:
        update_parameters(total_loss, train_optimizer_mask, embedding_model, outputs)
    if total_loss != 0:
        loss_val = total_loss.item()
    else:
        loss_val = 0.

    return loss_val


def update_loss_parameters_vectorized(labels, net_outputs, models, optimizers, params, train_optimizer_mask):
    '''
    define loss function and update parameters
    '''

    def update_parameters(loss):
        "perorm parameter updates"
        # clear previous gradients, compute gradients of all variables wrt loss
        for optimizer in optimizers:
            optimizer.zero_grad()

        loss.backward()
        # performs updates using calculated gradients
        for mask, optimizer in zip(train_optimizer_mask, optimizers):
            if mask:
                optimizer.step()

    total_loss = 0.
    loss_fns = params.loss_fns
    len_fns = len(loss_fns)

    label_inx = 0
    if sum(train_optimizer_mask):
        is_train = True
    else:
        is_train = False

    # loss_for_training = list(set(range(len(loss_fns))) - set(params.loss_excluded_from_training))
    # import ipdb
    # ipdb.set_trace()
    for i in range(len(loss_fns)):
        net_output, loss_fn = net_outputs[:, i], loss_fns[i]
        if hasattr(loss_fn, '__name__'):
            if loss_fn.__name__ is 'negative_log_partial_likelihood_loss':
                label = labels[:, label_inx:(label_inx + 2)]
                label_inx = label_inx + 2
                # print(label.shape)
                na_inx = ~(isnan(label[:, 0]) | isnan(label[:, 1]))
                label, net_output = label[na_inx, :], net_output[na_inx]
        else:
            label = labels[:, label_inx]
            label_inx = label_inx + 1
            na_inx = ~isnan(label)
            label, net_output = label[na_inx], net_output[na_inx]

        if(len(label) > 1) and not any(np.array(params.loss_excluded_from_training) == i):
            # in case of survival label is censor; censored data assumed to
            # sorted based on patient event time.
            loss_curr = loss_fn(net_output, label)
            loss_curr = loss_curr + regularized_loss(models[2], params, i)
        else:
            # loss_curr = torch.zeros(1)
            loss_curr = 0.

        if isnan(loss_curr):
            loss_curr = 0.

        # print(net_output)
        # print(loss_curr)
        # tracer()

        if is_train and params.pipeline_optimization and loss_curr != 0.:
            update_parameters(loss_curr)

        total_loss = total_loss + loss_curr

    # tracer()
    if is_train and not params.pipeline_optimization and total_loss != 0.:
        update_parameters(total_loss)
    if total_loss != 0:
        loss_val = total_loss.item()
    else:
        loss_val = 0.

    return loss_val


def tracer():
    import ipdb
    ipdb.set_trace()
