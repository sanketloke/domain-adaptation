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

labels = __import__('data.labels')



transform = transforms.Compose([
                                       transforms.Scale(opt.loadSize),
                                       transforms.CenterCrop(opt.fineSize),
                                       transforms.ToTensor(),
                                    ])


target_transform = transforms.Compose([
                                       transforms.Scale(opt.loadSize),
                                       transforms.CenterCrop(opt.fineSize),transforms.ToTensor(),#ToLabelTensor(labels.labels.labels),
                                    ])



domainAdata= SegmentationDataset(root=opt.dataroot + '/' + opt.domain_A ,
                        transform=transform, target_transform=target_transform, return_paths=True)
domainBdata= SegmentationDataset(root=opt.dataroot + '/' + opt.domain_B ,
                        transform=transform, target_transform=target_transform, return_paths=True)
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
cycle_data_loader.initialize(opt,transform,target_transform)


dataset = cycle_data_loader.load_data()
num_train = len(cycle_data_loader)
print('#training images = %d' % num_train)


model = create_model(opt)
visualizer = Visualizer(opt)


data_sample_len= (len(domainAdata) + len(domainBdata) + len(cycle_data_loader))/opt.batchSize





data_samples = ['AB', 'AC', 'BC']

p=[float(len(domainAdata)/opt.batchSize)/ data_sample_len, (float(len(domainBdata))/opt.batchSize) / data_sample_len  , ( float(len(cycle_data_loader)) /opt.batchSize) / data_sample_len ]

domainAdata_iter = domainAdataloader.__iter__()
domainBdata_iter = domainBdataloader.__iter__()
dataset_iter = dataset.__iter__()

total_steps=0

for epoch in range(1, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()



    data_sample_list = np.random.choice(data_samples, data_sample_len , p) 
    for i in data_sample_list:
      if i in 'AB':
        batch_n= next(dataset_iter)
        data={}
        data['AB_image_1'] = batch_n['A']
        data['AB_image_2'] = batch_n['B']
      elif i in 'BC':
        batch_n= next(domainBdata_iter)
        data={}
        data['B_image'] = batch_n[0][0]
        data['B_label'] = batch_n[1][0]
      else:
        batch_n= next(domainAdata_iter)
        data={}
        data['A_image'] = batch_n[0][0]
        data['A_label'] = batch_n[1][0]
      
      iter_start_time = time.time()
      total_steps += opt.batchSize
      epoch_iter = total_steps % num_train
      model.set_input(data,i)
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

