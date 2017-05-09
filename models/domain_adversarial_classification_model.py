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
import copy
import evaluation.metrics


class WildDomainAdaptationModel(BaseModel):
    
    def name(self):
        return 'WildDomainAdaptationModel'


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
        copyfeat4=copy.deepcopy(self.netG_BC.feat4)
        copyfeat5=copy.deepcopy(self.netG_BC.feat5)
        self.netG_AMidAhead=torch.nn.Sequential(copyfeat4,copyfeat5)
        self.netG_AMid=torch.nn.Sequential(self.netG_BC.feats,self.netG_AMidAhead[0],self.netG_AMidAhead[1])
        self.netG_BMid=torch.nn.Sequential(self.netG_BC.feats,self.netG_BC.feat4,self.netG_BC.feat5)
        self.netG_AC = network_classification_model('FCN16',self.gpu_ids,feats=self.netG_AMid[0],feat4=self.netG_AMid[1],feat5=self.netG_AMid[2],num_classes=self.opt.num_classes)
        #self.netG_AC= torch.nn.Sequential(self.netG_AMid,self.netG_BC.fconn,self.netG_BC.score_fconn,self.netG_BC.score_feat4)
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_AB = networks.define_D(512, opt.ndf,
                                         opt.which_model_netD,
                                         opt.n_layers_D, use_sigmoid, self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_BC, 'G_BC', which_epoch)
            self.load_network(self.netG_AMid,'AMid',which_epoch)
            #self.netG_AMid==torch.nn.Sequential(self.netG_BC.feats,self.netG_AMidAhead[0],self.netG_AMidAhead[1])
            self.netG_AC=  network_classification_model('FCN16',self.gpu_ids,feats=self.netG_AMid[0],feat4=self.netG_AMid[1],feat5=self.netG_AMid[2],fconn=self.netG_BC.fconn, score_fconn=self.netG_BC.score_fconn,score_feat4=self.netG_BC.score_feat4,num_classes=self.opt.num_classes)

            if self.isTrain:
                self.load_network(self.netD_AB, 'D_AB', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            self.fake_AC_pool=ImagePool(opt.pool_size)
            self.fake_BC_pool=ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_G_BC= torch.optim.Adam( (self.netG_BC.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizer_G_AMid=torch.optim.Adam(self.netG_AMid.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizer_D_AB = torch.optim.Adam(self.netD_AB.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G_AC = torch.optim.Adam(self.netD_AB.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            print('---------- Networks initialized -------------')
            networks.print_network(self.netG_BC)
            networks.print_network(self.netD_AB)
            print('-----------------------------------------------')


    def set_input(self, input, inputType):
        self.inputType =inputType

        if self.inputType in 'BC':
            input_B1=input['B_image']
            self.input_real_B.resize_(input_B1.size()).copy_(input_B1)
            input_B1label=input['B_label']
            self.input_real_C_givenB.resize_(input_B1label.size()).copy_(input_B1label)
        
        elif self.inputType in 'AC':
            input_A1=input['A_image']
            input_A1label=input['A_label']
            self.input_real_A.resize_(input_A1.size()).copy_(input_A1)
            self.input_real_C_givenA.resize_(input_A1label.size()).copy_(input_A1label)

        elif self.inputType in 'AB':
            inputAB_image_1=input['AB_image_1']
            inputAB_image_2=input['AB_image_2']
            self.input_real_A_unlabeled.resize_(inputAB_image_1.size()).copy_(inputAB_image_1)
            self.input_real_B_unlabeled.resize_(inputAB_image_2.size()).copy_(inputAB_image_2)
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
        if self.inputType in 'AB':
            self.unlabeled_A=Variable(self.input_real_A_unlabeled)
            self.unlabeled_B=Variable(self.input_real_B_unlabeled)


    def test(self):
        if self.inputType in 'AC':
            self.real_A = Variable(self.input_real_A)
            self.fake_C_given_A = self.netG_AC.forward(self.real_A)
            self.real_C_givenA=Variable(self.input_real_C_givenA)
            l1=self.fake_C_given_A.cpu().data[0].numpy().argmax(0)
            l2=self.input_real_C_givenA
            return evaluation.metrics.get_all_scores(l1,l2.cpu().long().numpy(),self.opt.num_classes)
        if self.inputType in 'BC':
            self.real_B = Variable(self.input_real_B)
            self.fake_C_given_B = self.netG_BC.forward(self.real_B)
            self.real_C_givenB=Variable(self.input_real_C_givenB)
            l1=self.fake_C_given_B.cpu().data[0].numpy().argmax(0)
            l2=self.input_real_C_givenB
            return evaluation.metrics.get_all_scores(l1,l2.cpu().long().numpy(),self.opt.num_classes)
        if self.inputType in 'AB':
            return 

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        if self.inputType in 'AC':
            self.optimizer_G_AC.zero_grad()
            self.backward_G_AC()
            self.optimizer_G_AC.step()

        if self.inputType in 'BC':
            self.optimizer_G_BC.zero_grad()
            self.backward_G_BC()
            self.optimizer_G_BC.step()

        if self.inputType in 'AB':
            self.real_A=self.unlabeled_A
            self.real_B= self.unlabeled_B
            self.optimizer_G_AMid.zero_grad()
            self.backward_G(self.unlabeled_A,self.unlabeled_B)
            self.optimizer_G_AMid.step()
            self.optimizer_D_AB.zero_grad()
            self.backward_D_AB()
            self.optimizer_D_AB.step()



    def backward_G(self,rA,rB):
        # GAN loss
        # D_A(G_A(A))
        self.fake_B = self.netG_AMid.forward(rA)
        pred_fake = self.netD_AB.forward(self.fake_B)
        self.loss_G_A = self.criterionGAN(pred_fake, True)
        # combined loss
        self.loss_G = self.loss_G_A 
        self.loss_G.backward()

   
    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD.forward(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD.forward(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_AB(self):
        self.fake_B = self.netG_AMid.forward(self.unlabeled_A)
        fake_B = self.fake_B_pool.query(self.fake_B)
        real_B= self.netG_BMid.forward(self.unlabeled_B)
        self.loss_D_A = self.backward_D_basic(self.netD_AB, real_B, fake_B)


    def backward_G_AC(self):
        fake_C= self.netG_AC.forward(self.real_A)
        self.loss_G_AC_L1= self.cross_entropy2d(fake_C,self.real_C_givenA)
        self.loss_G_AC_L1.backward()

    def backward_G_BC(self):
        fake_C= self.netG_BC.forward(self.real_B)
        self.loss_G_BC_L1= self.cross_entropy2d(fake_C,self.real_C_givenB)
        self.loss_G_BC_L1.backward()


    def get_current_visuals(self):
        if self.inputType in 'BC':
            real_B = util.tensor2im(self.real_B.data)
            fake_C=self.netG_BC.forward(self.real_B)
            from data.custom_transforms import ToLabelTensor
            labels = __import__('data.labels')
            mod_fakeC=ToLabelTensor(labels.labels.labels).label2image(fake_C.cpu().data[0].numpy().argmax(0))
            mod_realC=ToLabelTensor(labels.labels.labels).label2image(self.real_C_givenB.cpu().int().data[0].numpy())
            return OrderedDict([('real_B', real_B), ('real_C_givenB', mod_realC),('fake_C_givenB',mod_fakeC)])

        if self.inputType in 'AC':
            real_A = util.tensor2im(self.real_A.data)
            fake_C=self.netG_BC.forward(self.real_A)
            from data.custom_transforms import ToLabelTensor
            labels = __import__('data.labels')
            mod_fakeC=ToLabelTensor(labels.labels.labels).label2image(fake_C.cpu().data[0].numpy().argmax(0))
            mod_realC=ToLabelTensor(labels.labels.labels).label2image(self.real_C_givenA.cpu().int().data[0].numpy())
            return OrderedDict([('real_A', real_A), ('real_C_givenA', mod_realC) ,('fake_C_givenA',mod_fakeC)])


    def save(self, label):
        use_gpu = self.gpu_ids is not None
        self.save_network(self.netG_BC, 'G_BC', label, use_gpu)
        self.save_network(self.netG_AMid, 'G_AMid', label, use_gpu)
        self.save_network(self.netD_AB, 'D_AB', label, use_gpu)


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
            return OrderedDict([('G_BC_L1', self.loss_G_BC_L1.data[0]) 
            ])
        else:
            D_A = self.loss_D_A.data[0]
            G_A = self.loss_G_A.data[0]
            return OrderedDict([('D_A', D_A), ('G_A', G_A)])