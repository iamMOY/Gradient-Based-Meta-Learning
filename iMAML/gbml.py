import numpy as np
import torch
import torch.nn as nn
import os

from net.convnet import ConvNet
from net.resnet import ResNet

class GBML:
    def __init__(self,args):
        self.args = args
        self.batch_size = self.args.batc_size
        return None

    def __init_net(self):
        if self.args.net == 'ConvNet':
            self.network = ConvNet(self.args)
        elif self.args.net == 'ResNet':
            self.network = ResNet(self.args)
        self.network.train()
        self.netwrok.cuda()
        return None

    def __init_opt(self):
        if self.args.inner_opt == 'SGD':
            self.inner_optimizer == torch.optim.SGD(self.network.parameters(), lr=self.args.inner_lr)
        elif self.args.inner_opt == 'Adam':
            self.inner_optimizer == torch.optim.Adam(self.network.parameters(), lr=self.args.inner_lr)
        else:
            raise ValueError('Optimizer not supported')
        if self.args.outer_opt == 'SGD':
            self.outer_optimizer == torch.optim.SGD(self.network.parameters(), lr=self.args.inner_lr)
        elif self.args.outer_opt == 'Adam':
            self.outer_optimizer == torch.optim.Adam(self.network.parameters(), lr=self.args.inner_lr)
        else:
            raise ValueError('Optimizer not supported')
        
        return None

    def unpack_batch(self, batch):
        train_inputs, train_targets = batch['train']
        train_inputs = train_inputs.cuda()
        train_targets = train_targets.cuda()

        test_inputs, test_targets = batch['test']
        test_inputs = test_inputs.cuda()
        test_targets = test_targets.cuda()

        return train_inputs, train_targets, test_inputs, test_targets


    def inner_loop(self):
        raise NotImplemented

    def outer_loop(self):
        raise NotImplemented

    def lr_sched(self):
        self.ls_scheduler.step()
        return None

    def load(self):
        path = os.path.join(self.args.result_path, self.arg.alg, self.arg.load_path)
        self.network.load_state_dict(torch.load(path))

    def load_encoder(self):
        path = os.path.join(self.args.result_path,self.args.alg, self.args.load_path)
        self.network.encoder.load_state_dict(torch.load(path))

    def save(self,filename):
        path = os.path.join(self.args.result_path, self.args.alg, filename)
        torch.save(self.network.state_dict(), path)


