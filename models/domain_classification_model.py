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


class DomainClassificationModel(BaseModel):
    
    def name(self):
        return 'DomainClassificationModel'


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

        self.input_real_A = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)
        self.input_real_C_givenA = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)
        self.input_real_B = self.Tensor(opt.batchSize, opt.input_nc,
         
         #Commenting                          opt.fineSize, opt.fineSize)
        ###########subha###########self.input_real_C_givenB = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)
        self.input_real_A_unlabeled = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)
        self.input_real_B_unlabeled = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)

        # load/define networks
        self.netG_AB = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                    opt.which_model_netG, opt.norm, self.gpu_ids)

                # load/define networks
        self.netG_BC = network_classification_model('FCN16',self.gpu_ids,num_classes=22)

                # load/define networks
        ###########subha###########self.netG_BA = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                         #           opt.which_model_netG, opt.norm, self.gpu_ids)

        z=list(list(self.netG_AB.children())[0].children()) 
        temp=torch.nn.Sequential(*z[:10])
        self.netG_BC = torch.nn.Sequential(temp,network_classification_model('FCN16',self.gpu_ids,num_classes=22))


        self.netG_ABC= torch.nn.Sequential(self.netG_AB, self.netG_BC)
        
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            ###########subha########### self.netD_A = networks.define_D(opt.output_nc, opt.ndf,
                                  #       opt.which_model_netD,
                                  #       opt.n_layers_D, use_sigmoid, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf,
                                         opt.which_model_netD,
                                         opt.n_layers_D, use_sigmoid, self.gpu_ids)
        
            self.netD_C = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                         opt.which_model_netD,
                                         opt.n_layers_D, use_sigmoid, self.gpu_ids)


        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_AB, 'G_AB', which_epoch)
           ###########subha###########self.load_network(self.netG_BA, 'G_BA', which_epoch)
            self.load_network(self.netG_BC, 'G_BC', which_epoch)
            self.netG_ABC=torch.nn.Sequential(self.netG_AB,self.netG_BC)
            if self.isTrain:
                self.load_network(self.netD_A, 'D_A', which_epoch)
           ###########subha###########     self.load_network(self.netD_B, 'D_B', which_epoch)
                self.load_network(self.netD_C, 'D_C', which_epoch)

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

            self.optimizer_G_AC =torch.optim.Adam((self.netG_ABC.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G_BC= torch.optim.Adam( (self.netG_BC.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_C = torch.optim.Adam(self.netD_C.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

           ###########subha###########  self.optimizer_G_BA=torch.optim.Adam(self.netG_BA.parameters(),
                                  #              lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_AB.parameters(),self.netG_BA.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_C = torch.optim.Adam(self.netD_C.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

            print('---------- Networks initialized -------------')
            networks.print_network(self.netG_AB)
           ###########subha###########  networks.print_network(self.netG_BA)
            networks.print_network(self.netG_BC)
            networks.print_network(self.netD_A)
            networks.print_network(self.netD_B)
            networks.print_network(self.netD_C)
            print('-----------------------------------------------')


    def set_input(self, input, inputType):
        self.inputType =inputType

        if self.inputType in 'BC':
            input_B1=input['B_image']
            self.input_real_B.resize_(input_B1.size()).copy_(input_B1)
            input_B1label=input['B_label']
            self.input_real_C_givenB.resize_(input_B1label.size()).copy_(input_B1label)
        
       ###########subha###########  elif self.inputType in 'AC':
            ###########subha########### input_A1=input['A_image']
            ###########subha########### input_A1label=input['A_label']
            ###########subha########### self.input_real_A.resize_(input_A1.size()).copy_(input_A1)
            ###########subha########### self.input_real_C_givenA.resize_(input_A1label.size()).copy_(input_A1label)
        
        elif self.inputType in 'AB':
            inputAB_image_1=input['AB_image_1']
            inputAB_image_2=input['AB_image_2']
            self.input_real_A_unlabeled.resize_(inputAB_image_1.size()).copy_(inputAB_image_1)
            self.input_real_B.resize_(inputAB_image_2.size()).copy_(inputAB_image_2)
        else:
            print 'Please provide proper input for inputType'
            raise Exception('inputType inconsistent')


    def forward(self):
        ###########subha########### if self.inputType in 'AC':
          ###########subha###########   self.real_A = Variable(self.input_real_A)
           ###########subha###########  self.fake_C_given_A = self.netG_ABC.forward(self.real_A)
            ###########subha########### self.real_C_givenA=Variable(self.input_real_C_givenA)

        if self.inputType in 'BC':
            self.real_B = Variable(self.input_real_B)
            self.fake_C_given_B = self.netG_ABC.forward(self.real_B)
            self.real_C_givenB=Variable(self.input_real_C_givenB)

        if self.inputType in 'AB':
            self.unlabeled_A=Variable(self.input_real_A_unlabeled)
            self.unlabeled_B=Variable(self.input_real_B_unlabeled)

            self.fake_B = self.netG_AB.forward(self.unlabeled_A)
            self.rec_A= self.netG_AB.forward(self.fake_B)

            self.fake_A = self.netG_AB.forward(self.unlabeled_B)
            self.rec_B= self.netG_AB.forward(self.fake_A)

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        #if self.inputType in 'AC':
            # self.optimizer_D_C.zero_grad()
            # self.backward_D_C_givenG_AC()
            # self.optimizer_D_C.step()

            ###########subha########### self.optimizer_G_AC.zero_grad()
           ###########subha###########  self.backward_G_AC()
            ###########subha########### self.optimizer_G_AC.step()

           ###########subha###########  if self.opt.reconstruction_classifier>0:
           ###########subha###########      self.optimizer_G_BA.zero_grad()
           ###########subha###########      self.backward_G_BA()
           ###########subha########### self.optimizer_G_BA.step()

        if self.inputType in 'BC':
            # self.optimizer_D_C.zero_grad()
            # self.backward_D_C_givenG_BC()
            # self.optimizer_D_C.step()

            self.optimizer_G_BC.zero_grad()
            self.backward_G_BC()
            self.optimizer_G_BC.step()

        if self.inputType in 'AB':
            self.optimizer_G.zero_grad()
            self.backward_G(self.unlabeled_A,self.unlabeled_B)
            self.optimizer_G.step()
           ###########subha###########  self.optimizer_D_A.zero_grad()
          ###########subha###########   self.backward_D_A()
           ###########subha###########  self.optimizer_D_A.step()
            # D_B
            self.optimizer_D_B.zero_grad()
            self.backward_D_B()
            self.optimizer_D_B.step()



    def backward_G(self,rA,rB):
        print "Update the Cycle GAN"
        lambda_idt = self.opt.identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            self.idt_A = self.netG_AB.forward(rB)
            self.loss_idt_A = self.criterionIdt(self.idt_A, rB) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed.
            self.idt_B = self.netG_BA.forward(rA)
            self.loss_idt_B = self.criterionIdt(self.idt_B, rA) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss
        # D_A(G_A(A))
        self.fake_B = self.netG_AB.forward(rA)
        pred_fake = self.netD_A.forward(self.fake_B)
        self.loss_G_A = self.criterionGAN(pred_fake, True)
        

        # D_B(G_B(B))
        self.fake_A = self.netG_BA.forward(rB)
        pred_fake = self.netD_B.forward(self.fake_A)
        self.loss_G_B = self.criterionGAN(pred_fake, True)


        # Forward cycle loss
        self.rec_A = self.netG_BA.forward(self.fake_B)
        self.loss_cycle_A = self.criterionCycle(self.rec_A, rA) * lambda_A
        
        # Backward cycle loss
       ###########subha###########  self.rec_B = self.netG_AB.forward(self.fake_A)
        ###########subha########### self.loss_cycle_B = self.criterionCycle(self.rec_B, rB) * lambda_B
        

        # combined loss
        ###########I was not sure which to remove########################################
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
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

   ###########subha###########  def backward_D_A(self):
        ###########subha########### self.fake_B = self.netG_AB.forward(self.unlabeled_A)
       ###########subha###########  fake_B = self.fake_B_pool.query(self.fake_B)
       ###########subha###########  self.loss_D_A = self.backward_D_basic(self.netD_A, self.unlabeled_B, fake_B)

    def backward_D_B(self):
        self.fake_A = self.netG_BA.forward(self.unlabeled_B)
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B =  self.backward_D_basic(self.netD_B, self.unlabeled_A, fake_A)


    # def backward_D_C_givenG_X(self,netG,realA,realC,pool):
    #     fake_C= netG.forward(realA)
    #     fake_AC=pool.query(torch.cat((realA,fake_C),1))
    #     pred_fake=self.netD_C.forward(fake_AC.detach())
    #     loss_D_X_fake=self.criterionGAN(pred_fake,False)

    #     #Real
    #     real_AC=torch.cat((realA,realC),1)
    #     pred_real= self.netD_C.forward(real_AC)
    #     loss_D_X_real=self.criterionGAN(pred_real,True)

    #     self.loss_D_X=(loss_D_X_real + loss_D_X_fake) * 0.5;
    #     self.loss_D_X.backward()
    #     return (loss_D_X_real,loss_D_X_fake)

    # def backward_D_C_givenG_AC(self):
    #     q= self.backward_D_C_givenG_X(self.netG_ABC,self.real_A,self.real_C_givenA,self.fake_AC_pool)
    #     self.loss_D_AC_real=q[0]
    #     self.loss_D_AC_fake=q[1]

    # def backward_D_C_givenG_BC(self):
    #     q=self.backward_D_C_givenG_X(self.netG_BC , self.real_B,self.real_C_givenB,self.fake_BC_pool) 
    #     self.loss_D_BC_real=q[0]
    #     self.loss_D_BC_real=q[1]


    def backward_G_BA(self):
        ###########subha########### fake_B=netG_AB.forward(self.real_A)
        rec_A=netG_BA.forward(fake_B)
        self.loss_G_BA_L1= self.criterionL1(fake_B,self.real_A)
        self.loss_G_BA_L1.backward()


    def backward_G_AC(self):
        fake_C= self.netG_ABC.forward(self.real_A)
        # fake_AC=torch.cat((self.real_A,fake_C),1)
        # pred_fake = self.netD_C.forward(fake_AC)
        # self.loss_G_AC_GAN = self.criterionGAN(pred_fake,True)
        self.loss_G_AC_L1= self.cross_entropy2d(fake_C,self.real_C_givenA)
        # self.loss_G_AC = self.loss_G_AC_GAN +self.loss_G_AC_L1
        self.loss_G_AC_L1.backward()

    def backward_G_BC(self):
        fake_C= self.netG_BC.forward(self.real_B)
        # fake_BC=torch.cat((self.real_B,fake_C),1)
        # pred_fake = self.netD_C.forward(fake_BC)
        # self.loss_G_BC_GAN = self.criterionGAN(pred_fake,True)
        self.loss_G_BC_L1= self.cross_entropy2d(fake_C,self.real_C_givenB)
        # self.loss_G_BC = self.loss_G_BC_GAN +self.loss_G_BC_L1
        self.loss_G_BC_L1.backward()



    def get_current_visuals(self):
        if self.inputType in 'BC':
            real_B = util.tensor2im(self.real_B.data)
            fake_C=self.netG_BC.forward(self.real_B)
            fake_C = util.tensor2im(fake_C.data)
            real_C_givenB = util.tensor2im(self.real_C_givenB.data)
            return OrderedDict([('real_B', real_B), ('fake_C', fake_C), ('real_C_givenB', real_C_givenB)])

        if self.inputType in 'AC':
            real_A = util.tensor2im(self.real_A.data)
            fake_C=self.netG_BC.forward(self.real_A)
            fake_C = util.tensor2im(fake_C.data)
            real_C_givenA = util.tensor2im(self.real_C_givenA.data)
            return OrderedDict([('real_A', real_A), ('fake_C', fake_C), ('real_C_givenA', real_C_givenA)])

        if self.inputType in 'AB':
            real_A = util.tensor2im(self.real_A.data)
            fake_B = util.tensor2im(self.fake_B.data)
            rec_A  = util.tensor2im(self.rec_A.data)
            real_B = util.tensor2im(self.real_B.data)
            fake_A = util.tensor2im(self.fake_A.data)
            rec_B  = util.tensor2im(self.rec_B.data)
            if self.opt.identity > 0.0:
                idt_A = util.tensor2im(self.idt_A.data)
                idt_B = util.tensor2im(self.idt_B.data)
                return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A), ('idt_B', idt_B),
                                    ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B), ('idt_A', idt_A)])
            else:
                return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A),
                                    ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B)])


    def save(self, label):
        use_gpu = self.gpu_ids is not None
       ###########subha###########  self.save_network(self.netG_AB, 'G_AB', label, use_gpu)
        self.save_network(self.netG_BC, 'G_BC', label, use_gpu)
        self.save_network(self.netG_BA, 'netG_BA', label, use_gpu)
       ###########subha###########  self.save_network(self.netD_A, 'D_A', label, use_gpu)
        self.save_network(self.netD_B, 'D_B', label, use_gpu)
        self.save_network(self.netD_C, 'D_C', label, use_gpu)


    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D_C.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G_BC.param_groups:
            param_group['lr'] = lr

        ###########subha########### for param_group in self.optimizer_D_A.param_groups:
           ###########subha###########  param_group['lr'] = lr
        for param_group in self.optimizer_D_B.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G_AC.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G_BA.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G_ABC.param_groups:
            param_group['lr'] = lr


        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr



    def get_current_errors(self):
        print 'Supervised Loss'
        if self.inputType in 'AC':
            return OrderedDict([('G_AC_L1', self.loss_G_AC_L1.data[0]),
                    ('loss_G_BA_L1', self.loss_G_BA_L1.data[0]),
            ])
        elif self.inputType in 'BC':
            return OrderedDict([('G_GAN', self.loss_G_BC_L1.data[0])
            ])
        else:
            D_A = self.loss_D_A.data[0]
            G_A = self.loss_G_A.data[0]
            Cyc_A = self.loss_cycle_A.data[0]
            D_B = self.loss_D_B.data[0]
            G_B = self.loss_G_B.data[0]
            Cyc_B = self.loss_cycle_B.data[0]
            if self.opt.identity > 0.0:
                idt_A = self.loss_idt_A.data[0]
                idt_B = self.loss_idt_B.data[0]
                return OrderedDict([('D_A', D_A), ('G_A', G_A), ('Cyc_A', Cyc_A), ('idt_A', idt_A),
                                    ('D_B', D_B), ('G_B', G_B), ('Cyc_B', Cyc_B), ('idt_B', idt_B)])
            else:
                return OrderedDict([('D_A', D_A), ('G_A', G_A), ('Cyc_A', Cyc_A),
                                    ('D_B', D_B), ('G_B', G_B), ('Cyc_B', Cyc_B)])


        # #Supervised Loss if paired data has been added using setInput function
        # if real_C_givenB not None:
        #     self.optimizer_D_C.zero_grad()
        #     self.backward_D_C_givenG_BC()
        #     self.optimizer_D_C.step()

        # if real_C_givenB not None:
        #     self.optimizer_G_BC.zero_grad()
        #     self.backward_G_BC()
        #     self.optimizer_G_BC.step()

        # if real_C_givenA not None:
        #     self.optimizer_D_C.zero_grad()
        #     self.backward_D_C_givenG_AC()
        #     self.optimizer_D_C.step()

        # if real_C_givenA not None:
        #     self.optimizer_G_AC.zero_grad()
        #     self.backward_G_AC()
        #     self.optimizer_G_AC.step()

        # self.optimizer_G.zero_grad()
        # self.backward_G(real_A,unlabeled_B)
        # self.optimizer_G.step()
        # # D_A
        # self.optimizer_D_A.zero_grad()
        # self.backward_D_A(rB)
        # self.optimizer_D_A.step()
        # # D_B
        # self.optimizer_D_B.zero_grad()
        # self.backward_D_B(rA)
        # self.optimizer_D_B.step()


        # self.optimizer_G.zero_grad()
        # self.backward_G(unlabeled_A,real_B)
        # self.optimizer_G.step()
        # self.optimizer_D_A.zero_grad()
        # self.backward_D_A(rB)
        # self.optimizer_D_A.step()

        # # D_B
        # self.optimizer_D_B.zero_grad()
        # self.backward_D_B(rA)
        # self.optimizer_D_B.step()





    # def backward_D_C_givenG_AC(self):
    #     print "Backward Prop for D (for domain A and B)"
    #     #Check if supervised domain A-> domain C pair exists

    #     fake_C= self.netG_ABC.forward(self.real_A)
    #     fake_AC=self.fake_AC_pool.query(torch.cat((self.real_A,fake_C)))
    #     pred_fake=self.netD_C.forward(fake_AC.detach())
    #     self.loss_D_AC_fake=self.criterionGAN(pred_fake,False)

    #     #Real
    #     real_AC=torch.cat((real_A,real_C_givenA),1)
    #     pred_real= self.netD_C.forward(real_AC)
    #     self.loss_D_AC_real=self.criterionGAN(pred_real,True)

    #     self.loss_D_AC=(self.loss_D_AC_real + self.loss_D_AC_fake) * 0.5;
    #     self.loss_D_AC.backward()

                        
    # def backward_D_C_givenG_BC(self):
    #     fake_C=self.netG_BC.forward(self.real_B)
    #     fake_BC=self.fake_AC_pool.query(torch.cat(self.real_A,self.fake_C))
    #     pred_fake=self.netD_C.forward(fake_BC.detach())
    #     self.loss_D_BC_fake = self.criterionGAN(pred_fake,False)

    #     real_BC=torch.cat((self.real_B,self.real_C_givenB),1)
    #     pred_real=self.fake_BC_pool.query(torch.cat(self.real_B,self.fake_C))
    #     self.loss_D_BC_real=self.criterionGAN(pred_real,True)

    #     self.loss_D_BC=(self.loss_D_AC_real + self.loss_D_BC_fake) *0.5

    #     self.loss_D_BC.backward()
