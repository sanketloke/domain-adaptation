import numpy as np
import torch
import os
from collections import OrderedDict
from pdb import set_trace as st
from torch.autograd import Variable
import util.util as util
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import sys
from networks import network_classification_model
import torch.nn.functional as F

import evaluation.metrics


class FCNModel(BaseModel):
    
    def name(self):
        return 'FCNModel'


    #Cite Source!
    def cross_entropy2d(self,inputV, target, weight=None, size_average=True):
        # input: (n, c, h, w), target: (n, h, w)
        n, c, h, w = inputV.size()
        # log_p: (n, c, h, w)
        log_p = F.log_softmax(inputV)
        # log_p: (n*h*w, c)
        log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
        log_p = log_p.view(-1, c)
        # target: (n*h*w,)
        mask = target >= 0
        target = target[mask]
        loss = F.nll_loss(log_p, target.long(), weight=weight, size_average=False)
        if size_average:
            #print mask.data.sum()
            loss /= mask.data.sum()
        return loss

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # define tensors
        self.mode = 'all'
        self.input_real_A = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)
        self.input_real_C_givenA = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)
        self.input_real_B = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)
        self.input_real_C_givenB = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)
        self.input_real_A_unlabeled = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)
        self.input_real_B_unlabeled = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)

        self.netG_BC = network_classification_model('FCN16',self.gpu_ids,num_classes=self.opt.num_classes)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_BC, 'G_BC', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            # initialize optimizers
            self.optimizer_G_BC= torch.optim.Adam( (self.netG_BC.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            print('---------- Networks initialized -------------')
            networks.print_network(self.netG_BC)
            print('-----------------------------------------------')


    def set_input(self, input, inputType):
        self.inputType =inputType

        if self.inputType in 'BC' :
            input_B1=input['B_image']
            self.input_real_B.resize_(input_B1.size()).copy_(input_B1)
            input_B1label=input['B_label']
            self.input_real_C_givenB.resize_(input_B1label.size()).copy_(input_B1label)
        if self.inputType in 'AC':
            input_B1=input['A_image']
            self.input_real_B.resize_(input_B1.size()).copy_(input_B1)
            input_B1label=input['A_label']
            self.input_real_C_givenB.resize_(input_B1label.size()).copy_(input_B1label)
        else:
            print 'Please provide proper input for inputType'
            raise Exception('inputType inconsistent')


    def forward(self):
        if self.inputType in 'AC':
            self.real_A = Variable(self.input_real_A)
            self.fake_C_given_A = self.netG_AC.forward(self.real_A)
            self.real_C_givenA=Variable(self.input_real_C_givenA)
        if self.inputType in 'BC':
            self.real_B = Variable(self.input_real_B)
            self.fake_C_given_B = self.netG_BC.forward(self.real_B)
            self.real_C_givenB=Variable(self.input_real_C_givenB)


    def test(self):
        self.real_B = Variable(self.input_real_B)
        self.fake_C_given_B = self.netG_BC.forward(self.real_B)
        self.real_C_givenB=Variable(self.input_real_C_givenB)
        l1=self.fake_C_given_B.cpu().data[0].numpy().argmax(0)
        l2=self.input_real_C_givenB
        return evaluation.metrics.get_all_scores(l1,l2.cpu().long().numpy(),self.opt.num_classes)


    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        if self.inputType in 'AC':
            self.optimizer_G_AC.zero_grad()
            self.backward_G_BC()
            self.optimizer_G_AC.step()

        if self.inputType in 'BC':
            self.optimizer_G_BC.zero_grad()
            self.backward_G_BC()
            self.optimizer_G_BC.step()


    def backward_G_BC(self):
        fake_C= self.netG_BC.forward(self.real_B)
        self.loss_G_BC_L1= self.cross_entropy2d(fake_C,self.real_C_givenB)
        self.loss_G_BC_L1.backward()


    def get_current_visuals(self):
        real_B = util.tensor2im(self.real_B.data)
        fake_C=self.netG_BC.forward(self.real_B)
        from data.custom_transforms import ToLabelTensor
        labels = __import__('data.labels')
        mod_fakeC=ToLabelTensor(labels.labels.labels).label2image(fake_C.cpu().data[0].numpy().argmax(0))
        mod_realC=ToLabelTensor(labels.labels.labels).label2image(self.real_C_givenB.cpu().int().data[0].numpy())
        return OrderedDict([('real_B', real_B), ('real_C_givenB', mod_realC),('fake_C_givenB',mod_fakeC)])
        

    def save(self, label):
        use_gpu = self.gpu_ids is not None
        self.save_network(self.netG_BC, 'G_BC', label, use_gpu)


    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd 
        for param_group in self.optimizer_D_AB.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G_BC.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G_AMid.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr



    def get_current_errors(self):
        if self.inputType in 'AC':
            return OrderedDict([('G_AC_L1', self.loss_G_AC_L1.data[0])])
        elif self.inputType in 'BC':
            return OrderedDict([('G_GAN', self.loss_G_BC_L1.data[0])
            ])
        else:
            D_A = self.loss_D_A.data[0]
            G_A = self.loss_G_A.data[0]
            return OrderedDict([('D_A', D_A), ('G_A', G_A)])