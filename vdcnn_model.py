import torch.nn as nn
from torch.nn.init import kaiming_normal_


class ConvolutionBlock(nn.Module):

    def __init__(self, input_dim = 128, n_filters=256, kernel_size=3, padding=1, stride=1, shortcut=False,
                 downsampling=None):
        super(ConvolutionBlock, self).__init__()

        self.downsampling = downsampling
        self.shortcut = shortcut

        self.conv1 = nn.Conv1d(input_dim, n_filters, kernel_size=kernel_size, padding=padding, stride=stride)
        self.batchnorm1 = nn.BatchNorm1d(n_filters)
        self.relu1= nn.ReLU()

        self.conv2 = nn.Conv1d(n_filters, n_filters, kernel_size=kernel_size, padding=padding, stride=stride)
        self.batchnorm2 = nn.BatchNorm1d(n_filters)
        self.relu2 = nn.ReLU()

    def forward(self, input):

        residual = input
        output = self.conv1(input)
        output = self.batchnorm1(output)
        #output = self.batchnorm1(output)
        output = self.relu1(output)
        output = self.conv2(output)
        output = self.batchnorm2(output)

        # understood form figure in the paper
        # use output after conv and input to conv
        # simple:  we skip a conv block

        if self.shortcut:
            if self.downsampling is not None:
                residual = self.downsampling(input)

            output += residual

        output = self.relu2(output)

        return output


# vdcnn
class Vdcnn(nn.Module):

    def __init__(self, n_classes=2, num_embedding=69, embedding_dim=16, depth=9, max_seq_length=1014,
                n_fc_neurons=2048, shortcut=False):

        super(Vdcnn, self).__init__()

        self.max_seq_length = max_seq_length
        self.n_classes = n_classes
        self.depth = depth

        layers=[]
        fc_layers=[]
        base_num_features=64

        self.embed = nn.Embedding(num_embedding,embedding_dim,padding_idx=0, max_norm=None,
                                  norm_type=2, scale_grad_by_freq=False, sparse=False)

        # 1st layer
        layers.append(nn.Conv1d(embedding_dim, base_num_features, kernel_size= 3,padding=1))

        if depth == 9:
            num_conv_block = [0,0,0,0]
        elif depth == 17:
            num_conv_block = [1,1,1,1]
        elif depth == 29:
            num_conv_block = [4,4,1,1]
        elif depth == 49:
            num_conv_block=[7,7,4,2]

        # 64 filter layers
        layers.append(ConvolutionBlock(input_dim=base_num_features, n_filters=base_num_features,kernel_size=3,
                                       padding=1, shortcut=shortcut))

        for _ in range(num_conv_block[0]):
            layers.append(
                ConvolutionBlock(input_dim=base_num_features, n_filters=base_num_features, kernel_size=3,
                                 padding=1, shortcut=shortcut))

        layers.append(nn.MaxPool1d(kernel_size=3,stride=2, padding=1))

        # down-sampling
        # using vgg; first stride =1
        ds = nn.Sequential(nn.Conv1d(base_num_features, 2 * base_num_features, kernel_size=1, stride=1, bias=False),
                           nn.BatchNorm1d(2 * base_num_features))

        # 128 filters layer
        layers.append(
            ConvolutionBlock(input_dim=base_num_features, n_filters=2 * base_num_features, kernel_size=3, padding=1,
                      shortcut=shortcut, downsampling=ds))

        for _ in range(num_conv_block[1]):
            layers.append(
                ConvolutionBlock(input_dim=2 * base_num_features, n_filters=2 * base_num_features, kernel_size=3,
                                 padding=1, shortcut=shortcut))

        layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

        ds = nn.Sequential(nn.Conv1d(2 * base_num_features, 4 * base_num_features, kernel_size=1, stride=1, bias=False),
                           nn.BatchNorm1d(4 * base_num_features))

        # 256 filters
        layers.append(
            ConvolutionBlock(input_dim=2 * base_num_features, n_filters=4 * base_num_features, kernel_size=3, padding=1,
                      shortcut=shortcut, downsampling=ds))

        for _ in range(num_conv_block[2]):
            layers.append(
                ConvolutionBlock(input_dim=4 * base_num_features, n_filters=4 * base_num_features, kernel_size=3, padding=1,
                          shortcut=shortcut))

        layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

        ds = nn.Sequential(nn.Conv1d(4 * base_num_features, 8 * base_num_features, kernel_size=1, stride=1, bias=False),
                           nn.BatchNorm1d(8 * base_num_features))

        # 512 filters
        layers.append(
            ConvolutionBlock(input_dim=4 * base_num_features, n_filters=8 * base_num_features, kernel_size=3, padding=1,
                      shortcut=shortcut, downsampling=ds))

        for _ in range(num_conv_block[3]):
            layers.append(
                ConvolutionBlock(input_dim=8 * base_num_features, n_filters=8 * base_num_features, kernel_size=3, padding=1,
                          shortcut=shortcut))

        # k max pooling
        layers.append(nn.AdaptiveMaxPool1d(8))
        # output : k x 512; k=8
        fc_layers.extend([nn.Linear(8 * 8 * base_num_features, n_fc_neurons), nn.ReLU()])
        fc_layers.extend([nn.Linear(n_fc_neurons, n_fc_neurons), nn.ReLU()])
        fc_layers.extend([nn.Linear(n_fc_neurons, n_classes)])

        self.layers = nn.Sequential(*layers)
        self.fc_layers = nn.Sequential(*fc_layers)
        self.sigmoid = nn.Sigmoid()
        self.__init_weights()

    def __init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, input):

        output = self.embed(input)
        output = output.transpose(1, 2)
        output = self.layers(output)
        output = output.view(output.size(0), -1)
        output = self.fc_layers(output)
        output = self.sigmoid(output)

        return output


