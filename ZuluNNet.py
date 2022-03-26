import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ZuluNNet(nn.Module):
    def __init__(self,game,args):
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args
        super(ZuluNNet,self).__init__()
        self.conv1 = nn.Conv2d(1,args.num_channels,3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(args.num_channels,args.num_channels,3,stride=1,padding=1)
        self.conv3 = nn.Conv2d(args.num_channels,args.num_channels,3,stride=1)
        self.conv4 = nn.Conv2d(args.num_channels,args.num_channels,3,stride=1)

        self.bn1 = nn.BatchNorm2d(args.num_channels)
        self.bn2 = nn.BatchNorm2d(args.num_channels)
        self.bn3 = nn.BatchNorm2d(args.num_channels)
        self.bn4 = nn.BatchNorm2d(args.num_channels)
        self.bn_metadata_1 = nn.BatchNorm2d(64)

        self.fc_metadata_1 = nn.Linear(args.num_metadata, 64)

        self.fc1 = nn.Linear(args.num_channels * (self.board_x-4) * (self.board_y-4) + 64, 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)



        self.fc2 = nn.Linear(1024+64,512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512,self.action_size)

        self.fc4 = nn.Linear(512,1)

    def forward(self,s,data):
        #s shape: (batch,3,8)
        s = s.view(-1,1,self.board_x,self.board_y)
        s = F.relu(self.bn1(self.conv1(s)))
        s = F.relu(self.bn2(self.conv2(s)))
        s = F.relu(self.bn3(self.conv3(s)))
        s = F.relu(self.bn4(self.conv4(s)))
        s = s.view(-1,self.args.num_channels * (self.board_x-4) * (self.board_y-4))
        data = F.relu(self.bn_metadata_1(self.fc_metadata_1(data)))
        s = torch.concat([s,data],dim=-1)
        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args.dropout,
                      training=self.training)
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args.dropout, training=self.training)  # batch_size x 512

        pi = self.fc3(s)
        v = self.fc4(s)

        return F.log_softmax(pi,dim=1), torch.tanh(v)