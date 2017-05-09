import time
from options.train_options import TrainOptions
opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch
import pickle
from data.custom_transforms import ToLabelTensor
# with open("opt.obj",'wb') as f:
#     pickle.dump(opt,f)

from data.segmentation import SegmentationDataset
from models.models import create_model
from data.unaligned_data_loader import UnalignedDataLoader
import torch.utils.data
import torchvision.transforms as transforms
#from models.models import create_model
from util.visualizer import Visualizer
from pdb import set_trace as st
import numpy as np
import gc

import evaluation.metrics

labels = __import__('data.labels')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data.custom_transforms import DownSizeLabelTensor

ds1= DownSizeLabelTensor(opt.factor)
size= ds1.findDecreasedResolution(opt.fineSize)/2
transform = transforms.Compose([
                                       transforms.CenterCrop(opt.fineSize),
                                       transforms.Scale(size),
                                       transforms.ToTensor(),
                                    ])

target_transform = transforms.Compose([
                                       transforms.CenterCrop(opt.fineSize),transforms.ToTensor(),ToLabelTensor(labels.labels.labels)
                                    ])

target_transform2 = transforms.Compose([
                                       transforms.CenterCrop(opt.fineSize),transforms.ToTensor(),ToLabelTensor(labels.labels.labels)
                                    ])

domainAdata= SegmentationDataset(root=opt.dataroot + '/' + opt.domain_A , split_ratio=opt.split_ratio_A,
                        transform=transform, target_transform=target_transform, return_paths=True)
domainBdata= SegmentationDataset(root=opt.dataroot + '/' + opt.domain_B ,  split_ratio=opt.split_ratio_B,
                        transform=transform, target_transform=target_transform2, return_paths=True)
domainAdataloader = torch.utils.data.DataLoader(
            domainAdata,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

domainBdataloader = torch.utils.data.DataLoader(
    domainBdata,
    batch_size=opt.batchSize,
    shuffle=not opt.serial_batches,
    num_workers=int(opt.nThreads))


cycle_data_loader=UnalignedDataLoader()
cycle_data_loader.initialize(opt,transform,transform)


dataset = cycle_data_loader.load_data()
num_train = len(cycle_data_loader)
print('#training images = %d' % num_train)
print ('Finetune:'+str(opt.finetune))
print ('Split Ratio A:'+str(opt.split_ratio_A))
print ('Split Ratio B:'+str(opt.split_ratio_B))
print ('Split Ratio AB:'+str(opt.split_ratio_AB))
print ('Experiment Name:'+opt.name)
print ('Iterations'+str(opt.niter))
print ('Iterations Decay'+str(opt.niter_decay))
opt.continue_train=True
model = create_model(opt)
visualizer = Visualizer(opt)

print 'Pretraining Done!!'
print 'Starting Combined Training' 
model.mode='all'
avgtimetaken=[]
total_steps=0
# for epoch in range(1,opt.niter + opt.niter_decay + 1): # 
#     epoch_start_time = time.time()
#     domainBdata_iter = domainBdataloader.__iter__()
#     iter=0
#     print epoch
#     for i in range(0,len(domainBdataloader)):
#       s=time.time()
#       batch_n= next(domainBdata_iter)
#       data={}
#       data['B_image'] = batch_n[0][0]
#       data['B_label'] = ds1.downsize(ds1.downsize(batch_n[1][0]).data).data
#       print i
#       iter_start_time = time.time()
#       total_steps += opt.batchSize
#       epoch_iter = total_steps % num_train
#       model.set_input(data,'BC')
#       model.optimize_parameters()
#       e=time.time()
#       avgtimetaken.append(e-s)

#       if total_steps % opt.display_freq == 0:
#           visualizer.display_current_results(model.get_current_visuals(), epoch)
#       if total_steps % opt.print_freq == 0:
#           errors = model.get_current_errors()
#           visualizer.print_current_errors(epoch, total_steps, errors, iter_start_time)
#           if opt.display_id > 0:
#               visualizer.plot_current_errors(epoch, total_steps, opt, errors)

#       if total_steps % opt.save_latest_freq == 0:
#           print('saving the latest model (epoch %d, total_steps %d)' %
#                 (epoch, total_steps))
#           model.save('latest')
    
#     if epoch % opt.save_epoch_freq == 0:
#         print('saving the model at the end of epoch %d, iters %d' %
#               (epoch, total_steps))
#         model.save('latest')
#         model.save(epoch)

#     print('End of epoch %d / %d \t Time Taken: %d sec' %
#           (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

#     if epoch > opt.niter + opt.niter_decay*0.75:
#         model.update_learning_rate()
print 'Done'



# print 'Training Target Domain to Source Domain Adversarially'
# for epoch in range(1,20+ 1): # 
#     epoch_start_time = time.time()
#     domainABdata_iter = dataset.__iter__()
#     iter=0
#     for i in range(0,num_train):
#       s=time.time()
#       batch_n= next(domainABdata_iter)
#       data={}
#       data['AB_image_1'] = batch_n['A']
#       data['AB_image_2'] = batch_n['B']
#       iter_start_time = time.time()
#       total_steps += opt.batchSize
#       epoch_iter = total_steps % num_train
#       model.set_input(data,'AB')
#       model.optimize_parameters()
#       e=time.time()
#       avgtimetaken.append(e-s)

#       if total_steps % opt.print_freq == 0:
#           errors = model.get_current_errors()
#           visualizer.print_current_errors(epoch, total_steps, errors, iter_start_time)
    

#       if total_steps % opt.save_latest_freq == 0:
#           print('saving the latest model (epoch %d, total_steps %d)' %
#                 (epoch, total_steps))
#           model.save('latest')
    
#     if epoch % opt.save_epoch_freq == 0:
#         print('saving the model at the end of epoch %d, iters %d' %
#               (epoch, total_steps))
#         model.save('latest')
#         model.save(epoch)

#     print('End of epoch %d / %d \t Time Taken: %d sec' %
#           (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))



if opt.finetune>1:
  print 'FineTuning'
  for epoch in range(1,opt.niter + opt.niter_decay + 1): # 
      epoch_start_time = time.time()
      domainAdata_iter = domainAdataloader.__iter__()
      iter=0
      for i in range(0,len(domainAdataloader)):
        s=time.time()
        batch_n= next(domainAdata_iter)
        data={}
        data['A_image'] = batch_n[0][0]
        data['A_label'] = ds1.downsize(ds1.downsize(batch_n[1][0]).data).data
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter = total_steps % num_train
        model.set_input(data,'AC')
        model.optimize_parameters()
        e=time.time()
        avgtimetaken.append(e-s)
        if total_steps % opt.display_freq == 0:
            visualizer.display_current_results(model.get_current_visuals(), epoch)
        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            visualizer.print_current_errors(epoch, total_steps, errors, iter_start_time)
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, total_steps, opt, errors)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')
      
      if epoch % opt.save_epoch_freq == 0:
          print('saving the model at the end of epoch %d, iters %d' %
                (epoch, total_steps))
          model.save('latest')
          model.save(epoch)

      print('End of epoch %d / %d \t Time Taken: %d sec' %
            (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

      if epoch > opt.niter + opt.niter_decay*0.75:
          model.update_learning_rate()