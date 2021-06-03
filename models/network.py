import torch.nn as nn
import torch
from utils.model_utils import init_model
from torchsummary import summary

class Network_v1(nn.Module):
    def __init__(self):
        super(Network_v1,self).__init__()

        # Define layers
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=5,stride=1,padding=1)
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=2,padding=1)
        self.conv3 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=2,padding=1)

        self.max_pool = nn.MaxPool2d(3,2)

        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.bn2 = nn.BatchNorm2d(num_features=128)
        self.bn3 = nn.BatchNorm2d(num_features=256)

        self.fc1 = nn.Linear(in_features=256*8*8,out_features=3)
        # self.fc2 = nn.Linear(in_features=64,out_features=32)
        # self.fc3 = nn.Linear(in_features=32,out_features=2)
        #
        # self.bn_fc_1 = nn.BatchNorm1d(num_features=64)
        # self.bn_fc_2 = nn.BatchNorm1d(num_features=32)
        #
        self.lrelu = nn.LeakyReLU(inplace=False)
        # self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        # Layer 1
        x = self.lrelu(self.conv1(x))
        x = self.bn1(x)

        # Maxpool
        x = self.max_pool(x)

        # Layer 2
        x = self.lrelu(self.conv2(x))
        x = self.bn2(x)

        # Maxpool
        x = self.max_pool(x)

        # Layer 3
        x = self.lrelu(self.conv3(x))
        x = self.bn3(x)

        x = x.view(-1,256*8*8)
        x = self.fc1(x)

        # x = self.sigmoid(x)

        return x

def __main__():
    model = Network_v1().cuda()
    init_model(model)

    input = torch.randn(1, 3, 224, 224)
    summary(model, (3, 224, 224))
    # output = model(input)

if __name__ == "__main___":
    __main__()
