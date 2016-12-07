import numpy as np
import functools
import chainer.links as L
import chainer.functions as F
from collections import defaultdict
import nutszebra_chainer


class Conv_BN_ReLU(nutszebra_chainer.Model):

    def __init__(self, in_channel, out_channel, filter_size=(3, 3), stride=(1, 1), pad=(1, 1)):
        super(Conv_BN_ReLU, self).__init__(
            conv=L.Convolution2D(in_channel, out_channel, filter_size, stride, pad),
            bn=L.BatchNormalization(out_channel),
        )

    def weight_initialization(self):
        self.conv.W.data = self.weight_relu_initialization(self.conv)
        self.conv.b.data = self.bias_initialization(self.conv, constant=0)

    def __call__(self, x, train=False):
        return F.relu(self.bn(self.conv(x), test=not train))

    def count_parameters(self):
        return functools.reduce(lambda a, b: a * b, self.conv.W.data.shape)


class Inception(nutszebra_chainer.Model):

    def __init__(self, in_channel, conv1x1=64, reduce3x3=96, conv3x3=128, reduce5x5=16, conv5x5=32, pool_proj=32, pass_through=False, proj='max', stride=1):
        super(Inception, self).__init__()
        modules = []
        if pass_through is False:
            modules.append(('conv1x1', Conv_BN_ReLU(in_channel, conv1x1, 1, stride, 0)))
        modules.append(('reduce3x3', Conv_BN_ReLU(in_channel, reduce3x3, 1, 1, 0)))
        modules.append(('conv3x3', Conv_BN_ReLU(reduce3x3, conv3x3, 3, stride, 1)))
        modules.append(('reduce5x5', Conv_BN_ReLU(in_channel, reduce5x5, 1, 1, 0)))
        modules.append(('conv5x5_1', Conv_BN_ReLU(reduce5x5, conv5x5, 3, 1, 1)))
        modules.append(('conv5x5_2', Conv_BN_ReLU(conv5x5, conv5x5, 3, stride, 1)))
        if pass_through is False:
            modules.append(('pool_proj', Conv_BN_ReLU(in_channel, pool_proj, 1, 1, 0)))
        # register layers
        [self.add_link(*link) for link in modules]
        self.modules = modules
        self.proj = proj
        self.pass_through = pass_through
        self.stride = stride

    def weight_initialization(self):
        for name, link in self.modules:
            link.weight_initialization()

    @staticmethod
    def max_or_ave(word='ave'):
        if word == 'ave':
            return F.average_pooling_2d
        return F.max_pooling_2d

    def __call__(self, x, train=False):
        pool = Inception.max_or_ave(self.proj)
        b = self.reduce3x3(x, train)
        b = self.conv3x3(b, train)
        c = self.reduce5x5(x, train)
        c = self.conv5x5_1(c, train)
        c = self.conv5x5_2(c, train)
        d = pool(x, 3, self.stride, 1)
        if self.pass_through is False:
            d = self.pool_proj(d, train)
            a = self.conv1x1(x, train)
            return F.concat((a, b, c, d), axis=1)
        return F.concat((b, c, d), axis=1)

    def count_parameters(self):
        count = 0
        for name, link in self.modules:
            count += link.count_parameters()
        return count


class Googlenet(nutszebra_chainer.Model):

    def __init__(self, category_num):
        super(Googlenet, self).__init__()
        modules = []
        modules += [('conv1', Conv_BN_ReLU(3, 64, (7, 7), (2, 2), (3, 3)))]
        modules += [('conv2_1x1', Conv_BN_ReLU(64, 64, (1, 1), (1, 1), (0, 0)))]
        modules += [('conv2_3x3', Conv_BN_ReLU(64, 192, (3, 3), (1, 1), (1, 1)))]
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
            if 'linear' in name:
                count += functools.reduce(lambda a, b: a * b, link.W.data.shape)
            else:
                count += link.count_parameters()
        return count

    def weight_initialization(self):
        for name, link in self.modules:
            if 'linear' in name:
                self[name].W.data = self.weight_relu_initialization(self[name])
                self[name].b.data = self.bias_initialization(self[name], constant=0)
            else:
                self[name].weight_initialization()

    def __call__(self, x, train=True):
        h = self.conv1(x, train)
        h = F.max_pooling_2d(h, ksize=(3, 3), stride=(2, 2), pad=(1, 1))
        h = self.conv2_1x1(h, train)
        h = self.conv2_3x3(h, train)
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
