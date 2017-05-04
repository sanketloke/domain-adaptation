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

model = create_model(opt)
visualizer = Visualizer(opt)


data_sample_len= (len(domainAdata) + len(domainBdata) + len(cycle_data_loader))/opt.batchSize




data_samples = ['AB', 'AC', 'BC']

p=[float(len(domainAdata)/opt.batchSize)/ data_sample_len, (float(len(domainBdata))/opt.batchSize) / data_sample_len  , ( float(len(cycle_data_loader)) /opt.batchSize) / data_sample_len ]


mean_pixel_acc_epoch, mean_class_acc_epoch, mean_class_iou_epoch, per_class_acc_epoch, per_class_iou_epoch=[],[],[],[],[]

#mean_pixel_acc_test_epoch, mean_class_acc_test_epoch, mean_class_iou_test_epoch, per_class_acc_test_epoch, per_class_iou_test_epoch=[],[],[],[],[]
test_epoch_results=[]
mean_pixel_acc, mean_class_acc, mean_class_iou, per_class_acc, per_class_iou=0,0,0,np.zeros((opt.num_classes)),np.zeros((opt.num_classes))
avgcountAC=0
avgcountBC=0
total_steps=0
print 'Dataset A Size:'+str(len(domainAdata))
print 'Dataset B Size:'+str(len(domainBdata))
print 'Dataset AB Size:' +str(len(cycle_data_loader))
avgtimetaken=[]


model.mode='cycle'

print 'Pretraining CycleGAN'
for epoch in range(1, opt.niter + opt.niter_decay + 1): #opt.niter + opt.niter_decay + 1
    epoch_start_time = time.time()
    dataset_iter = dataset.__iter__()
    gc.collect()
    print 'Starting new epoch'
    iter=0
    for batch in dataset:
      if iter%20==0:
        print 'Epoch :'+str(epoch)+' Iteratiom:'+str(iter)
      data={}
      data['AB_image_1'] = batch['A']
      data['AB_image_2'] = batch['B']
      iter+=1
      iter_start_time = time.time()
      total_steps += opt.batchSize
      epoch_iter = total_steps % num_train
      model.set_input(data,'AB')
      model.optimize_parameters()
      if total_steps % opt.display_freq == 0:
          visualizer.display_current_results(model.get_current_visuals(), epoch)
      if total_steps % opt.print_freq == 0:
          errors = model.get_current_errors()
          visualizer.print_current_errors(epoch, epoch_iter, errors, iter_start_time)
          if opt.display_id > 0:
              visualizer.plot_current_errors(epoch, epoch_iter, opt, errors)

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

    if epoch > opt.niter:
        model.update_learning_rate()

print 'Pretraining Done!!'
print 'Starting Combined Training' 

for epoch in range(1,opt.niter + opt.niter_decay + 1): # opt.niter + opt.niter_decay + 1
    epoch_start_time = time.time()
    domainAdata_iter = domainAdataloader.__iter__()
    domainBdata_iter = domainBdataloader.__iter__()
    dataset_iter = dataset.__iter__()
    data_sample_list = np.random.choice(data_samples, data_sample_len , p) 
    gc.collect()
    if avgcountAC>0:
      mean_pixel_acc_epoch.append(mean_pixel_acc_AC_train/avgcountAC)
      mean_class_acc_epoch.append(mean_class_acc_AC_train/avgcountAC)
      mean_class_iou_epoch.append(mean_class_iou_AC_train/avgcountAC)
      per_class_acc_epoch.append(per_class_acc_AC_train/avgcountAC)
      per_class_iou_epoch.append(per_class_iou_AC_train/avgcountAC)
    if avgcountBC>0:
      mean_pixel_acc_epoch.append(mean_pixel_acc_BC_train/avgcountBC)
      mean_class_acc_epoch.append(mean_class_acc_BC_train/avgcountBC)
      mean_class_iou_epoch.append(mean_class_iou_BC_train/avgcountBC)
      per_class_acc_epoch.append(per_class_acc_BC_train/avgcountBC)
      per_class_iou_epoch.append(per_class_iou_BC_train/avgcountBC)


    mean_pixel_acc_AC_train, mean_class_acc_AC_train, mean_class_iou_AC_train, per_class_acc_AC_train, per_class_iou_AC_train=0,0,0,np.zeros((opt.num_classes)),np.zeros((opt.num_classes))
    mean_pixel_acc_BC_train, mean_class_acc_BC_train, mean_class_iou_BC_train, per_class_acc_BC_train, per_class_iou_BC_train=0,0,0,np.zeros((opt.num_classes)),np.zeros((opt.num_classes))

    avgcountAC=0
    avgcountBC=0
    iter=0
    for i in data_sample_list:
      s=time.time()
      try:
        if i in 'AB':
          batch_n= next(dataset_iter)
          data={}
          data['AB_image_1'] = batch_n['A']
          data['AB_image_2'] = batch_n['B']
        elif i in 'BC':
          batch_n= next(domainBdata_iter)
          data={}
          data['B_image'] = batch_n[0][0]
          data['B_label'] = ds1.downsize(ds1.downsize(batch_n[1][0]).data).data
        else:
          batch_n= next(domainAdata_iter)
          data={}
          data['A_image'] = batch_n[0][0]
          data['A_label'] = ds1.downsize(ds1.downsize(batch_n[1][0]).data).data
      except:
        continue
      iter_start_time = time.time()
      total_steps += opt.batchSize
      epoch_iter = total_steps % num_train
      print total_steps
      model.set_input(data,i)
      model.optimize_parameters()
      e=time.time()
      avgtimetaken.append(e-s)
      if i in 'AC' :
        a,b,c,d,e=model.test()
        mean_pixel_acc +=a
        mean_class_acc +=b
        mean_class_iou +=c 
        per_class_acc +=d
        per_class_iou  +=e
        avgcountAC+=1
      if i in 'BC':
        a,b,c,d,e=model.test()
        mean_pixel_acc +=a
        mean_class_acc +=b
        mean_class_iou +=c 
        per_class_acc +=d
        per_class_iou  +=e
        avgcountBC+=1
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
    
    if epoch%opt.test_epoch_freq==0:
        print('testing the model at the end of epoch %d, iters %d' %
                (epoch, total_steps))

        domainAdata_test= SegmentationDataset(root=opt.dataroot + '/' + opt.domain_A , split_ratio=opt.split_ratio_A,
                          transform=transform, target_transform=target_transform, return_paths=True)
        domainBdata_test= SegmentationDataset(root=opt.dataroot + '/' + opt.domain_B ,  split_ratio=opt.split_ratio_B,
                                transform=transform, target_transform=target_transform2, return_paths=True)
        
        domainAdata_test.mode='test'
        domainBdata_test.mode ='test'
        domainAdataloader_test = torch.utils.data.DataLoader(
                    domainAdata,
                    batch_size=opt.batchSize,
                    shuffle=not opt.serial_batches,
                    num_workers=int(opt.nThreads))

        domainBdataloader_test = torch.utils.data.DataLoader(
            domainBdata,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))
        domainAdata_iter_test = domainAdataloader.__iter__()
        domainBdata_iter_test = domainBdataloader.__iter__()
        mean_pixel_acc_test_A, mean_class_acc_test_A, mean_class_iou_test_A, per_class_acc_test_A, per_class_iou_test_A=0,0,0,np.zeros((opt.num_classes)),np.zeros((opt.num_classes))
        for i in range(0,len(domainAdata_test)):
            batch_n= next(domainAdata_iter_test)
            data={}
            data['A_image'] = batch_n[0][0]
            data['A_label'] = ds1.downsize(ds1.downsize(batch_n[1][0]).data).data
            model.set_input(data,'AC')
            a,b,c,d,e=model.test()
            mean_pixel_acc_test_A +=a
            mean_class_acc_test_A +=b
            mean_class_iou_test_A +=c 
            per_class_acc_test_A +=d
            per_class_iou_test_A  +=e
        mean_pixel_acc_test_A /= len(domainAdata_test)
        mean_class_acc_test_A /=len(domainAdata_test)
        mean_class_iou_test_A /=len(domainAdata_test)
        per_class_acc_test_A /=len(domainAdata_test)
        per_class_iou_test_A  /=len(domainAdata_test)

        print 'Mean Pixel Accuracy (Domain A):'+str(mean_pixel_acc_test_A)
        print 'Mean Class Accuracy (Domain A):'+str(mean_class_acc_test_A)
        print 'Mean Class IoU (Domain A):'+str(mean_class_iou_test_A)
        print 'Per Class Accuracy (Domain A):'+str(per_class_acc_test_A)
        print 'Per Class IoU (Domain A):'+str(per_class_iou_test_A)

        mean_pixel_acc_test_B, mean_class_acc_test_B, mean_class_iou_test_B, per_class_acc_test_B, per_class_iou_test_B=0,0,0,np.zeros((opt.num_classes)),np.zeros((opt.num_classes))
        for i in range(0,len(domainBdata_test)):
            batch_n= next(domainBdata_iter_test)
            data={}
            data['B_image'] = batch_n[0][0]
            data['B_label'] = ds1.downsize(ds1.downsize(batch_n[1][0]).data).data
            model.set_input(data,'BC')
            a,b,c,d,e=model.test()
            mean_pixel_acc_test_B +=a
            mean_class_acc_test_B +=b
            mean_class_iou_test_B +=c 
            per_class_acc_test_B +=d
            per_class_iou_test_B  +=e
        mean_pixel_acc_test_B /= len(domainBdata_test)
        mean_class_acc_test_B /=len(domainBdata_test)
        mean_class_iou_test_B /=len(domainBdata_test)
        per_class_acc_test_B /=len(domainBdata_test)
        per_class_iou_test_B  /=len(domainBdata_test)

        print 'Mean Pixel Accuracy (Domain B):'+str(mean_pixel_acc_test_B)
        print 'Mean Class Accuracy (Domain B):'+str(mean_class_acc_test_B)
        print 'Mean Class IoU (Domain B):'+str(mean_class_iou_test_B)
        print 'Per Class Accuracy (Domain B):'+str(per_class_acc_test_B)
        print 'Per Class IoU (Domain B):'+str(per_class_iou_test_B)
        
        test_epoch_results.append([[mean_pixel_acc_test_A, mean_class_acc_test_A, mean_class_iou_test_A, per_class_acc_test_A, per_class_iou_test_A],[mean_pixel_acc_test_B, mean_class_acc_test_B, mean_class_iou_test_B, per_class_acc_test_B, per_class_iou_test_B]])
        with open("results.obj",'wb') as f:
          pickle.dump(test_epoch_results,f)


    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    if epoch > opt.niter:
        model.update_learning_rate()

with open("results.obj",'wb') as f:
    pickle.dump(test_epoch_results,f)