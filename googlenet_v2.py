import numpy as np
import functools
import chainer.links as L
import chainer.functions as F
from collections import defaultdict
import nutszebra_chainer


class Inception(nutszebra_chainer.Model):

    def __init__(self, in_channel, conv1x1=64, reduce3x3=96, conv3x3=128, reduce5x5=16, conv5x5=32, pool_proj=32, pass_through=False, proj='max', stride=1):
        super(Inception, self).__init__()
        modules = []
        if pass_through is False:
            modules.append(('conv1x1', L.Convolution2D(in_channel, conv1x1, 1, stride, 0)))
            modules.append(('bn_conv1x1', L.BatchNormalization(conv1x1)))
        modules.append(('reduce3x3', L.Convolution2D(in_channel, reduce3x3, 1, 1, 0)))
        modules.append(('bn_reduce3x3', L.BatchNormalization(reduce3x3)))
        modules.append(('conv3x3', L.Convolution2D(reduce3x3, conv3x3, 3, stride, 1)))
        modules.append(('bn_conv3x3', L.BatchNormalization(conv3x3)))
        modules.append(('reduce5x5', L.Convolution2D(in_channel, reduce5x5, 1, 1, 0)))
        modules.append(('bn_reduce5x5', L.BatchNormalization(reduce5x5)))
        modules.append(('conv5x5_1', L.Convolution2D(reduce5x5, conv5x5, 3, 1, 1)))
        modules.append(('bn_conv5x5_1', L.BatchNormalization(conv5x5)))
        modules.append(('conv5x5_2', L.Convolution2D(conv5x5, conv5x5, 3, stride, 1)))
        modules.append(('bn_conv5x5_2', L.BatchNormalization(conv5x5)))
        if pass_through is False:
            modules.append(('pool_proj', L.Convolution2D(in_channel, pool_proj, 1, 1, 0)))
            modules.append(('bn_pool_proj', L.BatchNormalization(pool_proj)))
        # register layers
        [self.add_link(*link) for link in modules]
        self.modules = modules
        self.proj = proj
        self.pass_through = pass_through
        self.stride = stride

    def weight_initialization(self):
        for name, link in self.modules:
            if 'bn' not in name:
                self[name].W.data = self.weight_relu_initialization(link)
                self[name].b.data = self.bias_initialization(link, constant=0)

    @staticmethod
    def conv_bn_relu(x, conv, bn, train=False):
        return F.relu(bn(conv(x), test=not train))

    @staticmethod
    def max_or_ave(word='ave'):
        if word == 'ave':
            return F.average_pooling_2d
        return F.max_pooling_2d

    def __call__(self, x, train=False):
        func = Inception.conv_bn_relu
        pool = Inception.max_or_ave(self.proj)
        b = func(x, self.reduce3x3, self.bn_reduce3x3, train)
        b = func(b, self.conv3x3, self.bn_conv3x3, train)
        c = func(x, self.reduce5x5, self.bn_reduce5x5, train)
        c = func(c, self.conv5x5_1, self.bn_conv5x5_1, train)
        c = func(c, self.conv5x5_2, self.bn_conv5x5_2, train)
        d = pool(x, 3, self.stride, 1)
        if self.pass_through is False:
            d = func(d, self.pool_proj, self.bn_pool_proj, train)
            a = func(x, self.conv1x1, self.bn_conv1x1, train)
            return F.concat((a, b, c, d), axis=1)
        return F.concat((b, c, d), axis=1)

    @staticmethod
    def _conv_count_parameters(conv):
        return functools.reduce(lambda a, b: a * b, conv.W.data.shape)

    def count_parameters(self):
        count = 0
        for name, link in self.modules:
            if 'bn' not in name:
                count += Inception._conv_count_parameters(link)
        return count


class Googlenet(nutszebra_chainer.Model):

    def __init__(self, category_num):
        super(Googlenet, self).__init__()
        modules = []
        modules += [('conv1', L.Convolution2D(3, 64, (7, 7), (2, 2), (3, 3)))]
        modules += [('bn_conv1', L.BatchNormalization(64))]
        modules += [('conv2_1x1', L.Convolution2D(64, 64, (1, 1), (1, 1), (0, 0)))]
        modules += [('bn_conv2_1x1', L.BatchNormalization(64))]
        modules += [('conv2_3x3', L.Convolution2D(64, 192, (3, 3), (1, 1), (1, 1)))]
        modules += [('bn_conv2_3x3', L.BatchNormalization(192))]
        modules += [('inception3a', Inception(192, 64, 64, 64, 64, 96, 32, pass_through=False, proj='ave', stride=1))]
        modules += [('inception3b', Inception(256, 64, 64, 96, 64, 96, 64, pass_through=False, proj='ave', stride=1))]
        modules += [('inception3c', Inception(320, 0, 128, 160, 64, 96, 0, pass_through=True, proj='max', stride=2))]
        modules += [('inception4a', Inception(576, 224, 64, 96, 96, 128, 128, pass_through=False, proj='ave', stride=1))]
        modules += [('inception4b', Inception(576, 192, 96, 128, 96, 128, 128, pass_through=False, proj='ave', stride=1))]
        modules += [('inception4c', Inception(576, 160, 128, 160, 128, 160, 128, pass_through=False, proj='ave', stride=1))]
        modules += [('inception4d', Inception(608, 96, 128, 192, 160, 192, 128, pass_through=False, proj='ave', stride=1))]
        modules += [('inception4e', Inception(608, 0, 128, 192, 192, 256, 0, pass_through=True, proj='ave', stride=2))]
        modules += [('inception5a', Inception(1056, 352, 192, 320, 160, 224, 128, pass_through=False, proj='ave', stride=1))]
        modules += [('inception5b', Inception(1024, 352, 192, 320, 192, 224, 128, pass_through=False, proj='max', stride=1))]
        modules += [('linear', L.Linear(1024, category_num))]
        # register layers
        [self.add_link(*link) for link in modules]
        self.modules = modules
        self.name = 'googlenet_{}'.format(category_num)

    def count_parameters(self):
        count = 0
        for name, link in self.modules:
            if 'inception' in name:
                count += link.count_parameters()
            elif 'bn' not in name:
                count += functools.reduce(lambda a, b: a * b, link.W.data.shape)
        return count

    def weight_initialization(self):
        for name, link in self.modules:
            if 'inception' in name:
                self[name].weight_initialization()
            elif 'bn' not in name:
                self[name].W.data = self.weight_relu_initialization(self[name])
                self[name].b.data = self.bias_initialization(self[name], constant=0)

    @staticmethod
    def conv_bn_relu(x, conv, bn, train=False):
        return F.relu(bn(conv(x), test=not train))

    def __call__(self, x, train=True):
        func = Googlenet.conv_bn_relu
        h = func(x, self.conv1, self.bn_conv1, train)
        h = F.max_pooling_2d(h, ksize=(3, 3), stride=(2, 2), pad=(1, 1))
        h = func(h, self.conv2_1x1, self.bn_conv2_1x1, train)
        h = func(h, self.conv2_3x3, self.bn_conv2_3x3, train)
        h = F.max_pooling_2d(h, ksize=(3, 3), stride=(2, 2), pad=(1, 1))
        h = self.inception3a(h, train)
        h = self.inception3b(h, train)
        h = self.inception3c(h, train)
        h = self.inception4a(h, train)
        h = self.inception4b(h, train)
        h = self.inception4c(h, train)
        h = self.inception4d(h, train)
        h = self.inception4e(h, train)
        h = self.inception5a(h, train)
        h = self.inception5b(h, train)
        num, categories, y, x = h.data.shape
        # global average pooling
        h = F.reshape(F.average_pooling_2d(h, (y, x)), (num, categories))
        h = F.relu(self.linear(h))
        return h

    def calc_loss(self, y, t):
        loss = F.softmax_cross_entropy(y, t)
        return loss

    def accuracy(self, y, t, xp=np):
        y.to_cpu()
        t.to_cpu()
        indices = np.where((t.data == np.argmax(y.data, axis=1)) == True)[0]
        accuracy = defaultdict(int)
        for i in indices:
            accuracy[t.data[i]] += 1
        indices = np.where((t.data == np.argmax(y.data, axis=1)) == False)[0]
        false_accuracy = defaultdict(int)
        false_y = np.argmax(y.data, axis=1)
        for i in indices:
            false_accuracy[(t.data[i], false_y[i])] += 1
        return accuracy, false_accuracy
