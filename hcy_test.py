import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import sys
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
sys.path.append("..")
root='/Users/lucifer/Desktop/pytorch/Datasets/FashionMNIST'
mnist_train = torchvision.datasets.FashionMNIST(root='/Users/lucifer/Desktop/pytorch/Datasets/FashionMNIST', train=True, download=False, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='/Users/lucifer/Desktop/pytorch/Datasets/FashionMNIST', train=False, download=False, transform=transforms.ToTensor())
def dot(a,b):
    if a.requires_grad:
        an=a.detach().numpy()
    else:
        an=a.numpy()
    if b.requires_grad:
        bn=b.detach().numpy()
    else:
        bn=b.numpy()
    return torch.from_numpy(np.dot(an,bn))
def linreg(X, w, b):  # 本函数已保存在d2lzh_pytorch包中方便以后使用
    return torch.mm(X, w) + b
def linreg_dot(X,w,b):
    return dot(X,w)+b
def squared_loss(y_hat, y):  
    return (y_hat - y.view(y_hat.size())) ** 2 / 2
def sgd(params, lr, batch_size):  # 本函数已保存在d2lzh_pytorch包中方便以后使用
    for param in params:
        param.data -= lr * param.grad / batch_size # 注意这里更改param时用的param.data
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        if isinstance(net, torch.nn.Module):
            net.eval() # 评估模式, 这会关闭dropout
            acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            net.train() # 改回训练模式
        else: # 自定义的模型
            if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                # 将is_training设置成False
                acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item() 
            else:
                acc_sum += (net(X).argmax(dim=1) == y).float().sum().item() 
        n += y.shape[0]
    return acc_sum / n
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)]) # 最后一次可能不足一个batch
        yield  features.index_select(0, j), labels.index_select(0, j)
    #index_select()的第一个参数代表切割的维度，第二个参数是索引，确定切割的范围
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]
def use_svg_display():
    # 用矢量图显示
    display.set_matplotlib_formats('svg')
def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize
def train_2d(trainer):
    x1,x2,s1,s2=-5,-2,0,0
    results=[(x1,x2)]
    for i in range(20):
        x1,x2,s1,s2=trainer(x1,x2,s1,s2)
        results.append((x1,x2))
    print('epoch %d, x1 %f, x2 %f' % (i + 1, x1, x2))
    return results
def show_trace_2d(f, results):  
    plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = np.meshgrid(np.arange(-5.5, 1.0, 0.1), np.arange(-3.0, 1.0, 0.1))
    plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    plt.xlabel('x1')
    plt.ylabel('x2')

def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(3.5, 2.5)):
    set_figsize(figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)
def get_data(root='data/airfoil_self_noise.dat'):  
    data = np.genfromtxt(root, delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return torch.tensor(data[:1500, :-1], dtype=torch.float32), \
    torch.tensor(data[:1500, -1], dtype=torch.float32) # 前1500个样本(每个样本5个特征)

def corr2d(X, K):  
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y
def show_fashion_mnist(images, labels):
    use_svg_display()
    # 这里的_表示我们忽略（不使用）的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()
def load_data_fashion_mnist(batch_size,resize=None,root=root):
    if sys.platform.startswith('win'):
        num_workers = 0  # 0表示不用额外的进程来加速读取数据
    else:
        num_workers = 4
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())
    transform = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=False, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=False, transform=transform) 
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_iter,test_iter
def load_features_and_label_fashion_mnist():
    features=[]
    labels=[]
    test_features=[]
    test_labels=[]
    for feature, label in mnist_train:
        features.append(feature)
        labels.append(label)
    features=torch.stack(features)
    labels=torch.tensor(labels)
    for feature, label in mnist_train:
        test_features.append(feature)
        test_labels.append(label)
    test_features=torch.stack(test_features)
    test_labels=torch.tensor(test_labels)
    return features,labels,test_features,test_labels
def train_module(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()
            
            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            
            l.backward()
            if optimizer is None:
                sgd(params, lr, batch_size)
            else:
                optimizer.step() 
            
            
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))
def train_module_time(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        print("running {}th epoch".format(epoch))
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            start = time.time()
            y_hat = net(X)
            l = loss(y_hat, y).sum()
            
            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            
            l.backward()
            if optimizer is None:
                sgd(params, lr, batch_size)
            else:
                optimizer.step() 
            
            
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
            end = time.time()
            print("循环运行时间:%.2f秒"%(end-start))
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))
# 本函数已保存在d2lzh_pytorch包中方便以后使用
def train_optim_module(optimizer_fn, states, hyperparams, features, labels,
              batch_size=10, num_epochs=2):
    # 初始化模型
    net, loss = linreg, squared_loss
    
    w = torch.nn.Parameter(torch.tensor(np.random.normal(0, 0.01, size=(features.shape[1], 1)), dtype=torch.float32),
                           requires_grad=True)
    b = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=True)

    def eval_loss():
        return loss(net(features, w, b), labels).mean().item()

    ls = [eval_loss()]
    data_iter = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(features, labels), batch_size, shuffle=True)
    
    for _ in range(num_epochs):
        start = time.time()
        for batch_i, (X, y) in enumerate(data_iter):
            l = loss(net(X, w, b), y).mean()  # 使用平均损失
            
            # 梯度清零
            if w.grad is not None:
                w.grad.data.zero_()
                b.grad.data.zero_()
                
            l.backward()
            optimizer_fn([w, b], states, hyperparams)  # 迭代模型参数
            if (batch_i + 1) * batch_size % 100 == 0:
                ls.append(eval_loss())  # 每100个样本记录下当前训练误差
    # 打印结果和作图
    print('loss: %f, %f sec per epoch' % (ls[-1], time.time() - start))
    set_figsize()
    plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
    plt.xlabel('epoch')
    plt.ylabel('loss')
class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)
class GlobalAvgPool2d(nn.Module):
    # 全局平均池化层可通过将池化窗口形状设置成输入的高和宽实现
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])
class linear_regression:
    def __init__(self,features,labels,num_inputs=2,lr=0.01,num_epoches=10,batch_size=10):
        self.lr=lr
        self.num_epoches=num_epoches
        self.features=features
        self.labels=labels
        self.batch_size=batch_size
        self.w = torch.tensor(np.random.normal(loc=0, scale=0.01, size=(num_inputs, 1)), dtype=torch.float32)
        self.b = torch.zeros(1, dtype=torch.float32)
        self.w.requires_grad_(requires_grad=True)
        self.b.requires_grad_(requires_grad=True) 
    def data_iter(self,features,labels):
        num_examples = len(features)
        indices = list(range(num_examples))
        random.shuffle(indices)  # 样本的读取顺序是随机的
        for i in range(0, num_examples, self.batch_size):
            j = torch.LongTensor(indices[i: min(i + self.batch_size, num_examples)]) # 最后一次可能不足一个batch
            yield  features.index_select(0, j), labels.index_select(0, j)
    def linreg(self,X, w, b):
        return torch.mm(X, w) + b
    def squared_loss(self,y_hat, y):  
        return (y_hat - y.view(y_hat.size())) ** 2 / 2
    def sgd(self,params, lr, batch_size):
        for param in params:
            param.data -= lr * param.grad / batch_size # 注意这里更改param时用的param.data
    def fit(self):
        for i in range(self.num_epoches):
            for X,y in self.data_iter(self.features,self.labels):
                l=self.squared_loss(self.linreg(X,self.w,self.b),y).sum()
                l.backward()
                self.sgd([self.w,self.b],self.lr,self.batch_size)
                self.w.grad.data.zero_()
                self.b.grad.data.zero_()
            train_loss=self.squared_loss(self.linreg(self.features,self.w,self.b),self.labels)
            print('epoch %d, loss %f' % (i + 1, train_loss.mean().item()))
    
class softmax_regression:
    def __init__(self,num_inputs,num_outputs,features=None,labels=None,test_features=None,test_labels=None,lr=0.01,num_epoches=10,batch_size=10,itered=True,train_iter=None,test_iter=None):
        self.lr=lr
        self.num_epoches=num_epoches
        self.num_inputs=num_inputs
        self.num_outputs=num_outputs
        self.features=features
        self.labels=labels
        self.testfeatures=test_features
        self.testlabels=test_labels
        self.batch_size=batch_size
        self.w = torch.tensor(np.random.normal(loc=0, scale=0.01, size=(num_inputs, num_outputs)), dtype=torch.float32)
        self.b = torch.zeros(num_outputs, dtype=torch.float32)
        self.w.requires_grad_(requires_grad=True)
        self.b.requires_grad_(requires_grad=True)
        self.itered=itered
        self.train_iter=train_iter
        self.test_iter=test_iter
    def data_iter(self,features,labels):
        num_examples = len(features)
        indices = list(range(num_examples))
        random.shuffle(indices)  # 样本的读取顺序是随机的
        for i in range(0, num_examples, self.batch_size):
            j = torch.LongTensor(indices[i: min(i + self.batch_size, num_examples)]) # 最后一次可能不足一个batch
            yield  features.index_select(0, j), labels.index_select(0, j)
    def softmax(self,X):
        X_exp=X.exp()
        partition=X_exp.sum(dim=1,keepdim=True)
        return X_exp/partition
    def net(self,X, W, b):
        return self.softmax(torch.mm(X.view(-1,self.num_inputs),W)+b)
    def cross_entropy(self,y_hat, y):
        return - torch.log(y_hat.gather(1, y.view(-1, 1)))
    def accuracy(self,y_hat,y):
        return (y_hat.argmax(dim=1)==y).float().mean().item()
    def evaluate_accuracy(self,data_iter,net):
        acc_sum,n=0.0,0
        for X,y in data_iter:
            acc_sum+=(self.net(X,self.w,self.b).argmax(dim=1)==y).float().sum().item()
            n+=y.shape[0]
        return acc_sum/n
    def sgd(self,params, lr, batch_size):
        for param in params:
            param.data -= lr * param.grad / batch_size # 注意这里更改param时用的param.data
    def fit(self):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for epoch in range(self.num_epoches):
            if self.itered==True:
                for X,y in self.train_iter:
                    l=self.cross_entropy(self.net(X,self.w,self.b),y).sum()
                    l.backward()
                    
                    self.sgd([self.w,self.b],self.lr,self.batch_size)
                    self.w.grad.data.zero_()
                    self.b.grad.data.zero_()
            
                    train_l_sum += l.item()
                    train_acc_sum += (self.net(X,self.w,self.b).argmax(dim=1) == y).sum().item()
                    n += y.shape[0]
                test_acc = self.evaluate_accuracy(self.test_iter, self.net)
                print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'%
                      (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))
            else:
                for X,y in self.data_iter(self.features,self.labels):
                    l=self.cross_entropy(self.net(X,self.w,self.b),y).sum()
                    l.backward()
                    self.sgd([self.w,self.b],self.lr,self.batch_size)
                    self.w.grad.data.zero_()
                    self.b.grad.data.zero_()
                    train_l_sum += l.item()
                    train_acc_sum += (self.net(X,self.w,self.b).argmax(dim=1) == y).sum().item()
                    n += y.shape[0]
                test_acc = self.evaluate_accuracy(self.data_iter(self.testfeatures,self.testlabels),self.net)
                print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'%
                      (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5), # in_channels, out_channels, kernel_size
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2), # kernel_size, stride
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16*4*4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 96, 11, 4), # in_channels, out_channels, kernel_size, stride, padding
            nn.ReLU(),
            nn.MaxPool2d(3, 2), # kernel_size, stride
            # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            # 连续3个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，进一步增大了输出通道数。
            # 前两个卷积层后不使用池化层来减小输入的高和宽
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
         # 这里全连接层的输出个数比LeNet中的大数倍。使用丢弃层来缓解过拟合
        self.fc = nn.Sequential(
            nn.Linear(256*5*5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            # 输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
            nn.Linear(4096, 10),
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output
class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims):
        super(BatchNorm, self).__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成0和1
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # 不参与求梯度和迭代的变量，全在内存上初始化成0
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.zeros(shape)
    def batch_norm(self,is_training, X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # 判断当前模式是训练模式还是预测模式
        if not is_training:
            # 如果是在预测模式下，直接使用传入的移动平均所得的均值和方差
            X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
        else:
            assert len(X.shape) in (2, 4)
            if len(X.shape) == 2:
                # 使用全连接层的情况，计算特征维上的均值和方差
                mean = X.mean(dim=0)
                var = ((X - mean) ** 2).mean(dim=0)
            else:
                # 使用二维卷积层的情况，计算通道维上（axis=1，即每个通道上的平均）的均值和方差。这里我们需要保持
                # X的形状以便后面可以做广播运算
                mean = X.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
                var = ((X - mean) ** 2).mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            # 训练模式下用当前的均值和方差做标准化
            X_hat = (X - mean) / torch.sqrt(var + eps)
            # 更新移动平均的均值和方差
            moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
            moving_var = momentum * moving_var + (1.0 - momentum) * var
        Y = gamma * X_hat + beta  # 拉伸和偏移
        return Y, moving_mean, moving_var
    def forward(self, X):
        # 如果X不在内存上，将moving_mean和moving_var复制到X所在显存上
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # 保存更新过的moving_mean和moving_var, Module实例的traning属性默认为true, 调用.eval()后设成false
        Y, self.moving_mean, self.moving_var = self.batch_norm(self.training, 
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9)
        return Y
class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)