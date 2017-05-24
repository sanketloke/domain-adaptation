import time
from options.train_options import TrainOptions
opt = TrainOptions().parse() 
#opt.dataroot='/home/sloke/repos/nips2017/left8bit/gtacityscapes/test'
opt.split_ratio_A=1
opt.split_ratio_B=1
 # set CUDA_VISIBLE_DEVICES before import torch
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

opt.continue_train=True
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



#mean_pixel_acc_test_epoch, mean_class_acc_test_epoch, mean_class_iou_test_epoch, per_class_acc_test_epoch, per_class_iou_test_epoch=[],[],[],[],[]
test_epoch_results=[]
mean_pixel_acc, mean_class_acc, mean_class_iou, per_class_acc, per_class_iou=0,0,0,np.zeros((opt.num_classes)),np.zeros((opt.num_classes))
avgcountAC=0
avgcountBC=0
total_steps=0

avgtimetaken=[]

model = create_model(opt)
visualizer = Visualizer(opt)

domainAdata_test= SegmentationDataset(root=opt.dataroot + '/' + opt.domain_A , split_ratio=opt.split_ratio_A,
                  transform=transform, target_transform=target_transform, return_paths=True)
domainBdata_test= SegmentationDataset(root=opt.dataroot + '/' + opt.domain_B ,  split_ratio=opt.split_ratio_B,
                        transform=transform, target_transform=target_transform2, return_paths=True)

print 'Dataset A Size:'+str(len(domainAdata_test))
print 'Dataset B Size:'+str(len(domainBdata_test))

domainAdataloader_test = torch.utils.data.DataLoader(
            domainAdata_test,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

domainBdataloader_test = torch.utils.data.DataLoader(
    domainBdata_test,
    batch_size=opt.batchSize,
    shuffle=not opt.serial_batches,
    num_workers=int(opt.nThreads))
domainAdata_iter_test = domainAdataloader_test.__iter__()
domainBdata_iter_test = domainBdataloader_test.__iter__()
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
    print 'Mean Pixel Accuracy (Domain A):'+str(a)
    print 'Mean Class Accuracy (Domain A):'+str(b)
    print 'Mean Class IoU (Domain A):'+str(c)
    print 'Per Class Accuracy (Domain A):'+str(d)
    print 'Per Class IoU (Domain A):'+str(e)
    print 'Iteration:'+str(i)
    print 'Model:'+opt.name
    if total_steps % opt.display_freq == 0:
          visualizer.display_current_results(model.get_current_visuals(), i)
mean_pixel_acc_test_A /= len(domainAdata_test)
cycle_data_loader=UnalignedDataLoader()
cycle_data_loader.initialize(opt,transform,transform)


# dataset = cycle_data_loader.load_data()
# q=0
# for i in dataset:
#     batch_n= i#next(domainAdata_iter_test)
#     q=q+1
#     data={}
#     data['AB_image_1'] = i['A']#batch_n[0][0]
#     data['AB_image_2'] = i['B']#ds1.downsize(ds1.downsize(batch_n[1][0]).data).data
#     model.set_input(data,'AB')
#     model.test()
#     if total_steps % opt.display_freq == 0:
#           visualizer.display_current_results(model.get_current_visuals(), q)
# mean_pixel_acc_test_A /= len(domainAdata_test)

# mean_class_acc_test_A /=len(domainAdata_test)
# mean_class_iou_test_A /=len(domainAdata_test)
# per_class_acc_test_A /=len(domainAdata_test)
# per_class_iou_test_A  /=len(domainAdata_test)
# print 'Final Results:'
# print 'Mean Pixel Accuracy (Domain A):'+str(mean_pixel_acc_test_A)
# print 'Mean Class Accuracy (Domain A):'+str(mean_class_acc_test_A)
# print 'Mean Class IoU (Domain A):'+str(mean_class_iou_test_A)
# print 'Per Class Accuracy (Domain A):'+str(per_class_acc_test_A)
# print 'Per Class IoU (Domain A):'+str(per_class_iou_test_A)


# print 'Mean Pixel Accuracy (Domain A):'+str(mean_pixel_acc_test_A)
# print 'Mean Class Accuracy (Domain A):'+str(mean_class_acc_test_A)
# print 'Mean Class IoU (Domain A):'+str(mean_class_iou_test_A)
# print 'Per Class Accuracy (Domain A):'+str(per_class_acc_test_A)
# print 'Per Class IoU (Domain A):'+str(per_class_iou_test_A)

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
    print 'Mean Pixel Accuracy (Domain B):'+str(a)
    print 'Mean Class Accuracy (Domain B):'+str(b)
    print 'Mean Class IoU (Domain B):'+str(c)
    print 'Per Class Accuracy (Domain B):'+str(d)
    print 'Per Class IoU (Domain B):'+str(e)
    print 'Iteration:'+str(i)
    print 'Model:'+opt.name
    if total_steps % opt.display_freq == 0:
          visualizer.display_current_results(model.get_current_visuals(), i)
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
