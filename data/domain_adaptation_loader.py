import torch.utils.data
import torchvision.transforms as transforms
from data.base_data_loader import BaseDataLoader
from data.image_folder import ImageFolder
from builtins import object
from pdb import set_trace as st
from custom_transforms import ToLabelTensor

class PairedData(object):


    def cityscapes_assign_trainIds(self, label):
        """
        Map the given label IDs to the train IDs appropriate for training
        Use the label mapping provided in labels.py from the cityscapes scripts
        """
        labels = __import__('labels')
        
        label = np.array(label, dtype=np.float32)
        if sys.version_info[0] < 3:
            for k, v in id2trainId.iteritems():
                label[label == k] = v
        else:
            for k, v in self.id2trainId.items():
                label[label == k] = v
        return label

    def __init__(self, domainAdata,domainBdata, data_loader_AB_images_1 , data_loader_AB_images_2): #,dataset
        self.domainAdata = domainAdata
        self.domainBdata = domainBdata
        self.data_loader_AB_images_1 = data_loader_AB_images_1
        self.data_loader_AB_images_2 = data_loader_AB_images_2
        #self.dataset = dataset
        self.__iter__()

    def __iter__(self):
        self.data_loader_A_images_iter = iter(self.data_loader_A_images)
        self.data_loader_B_images_iter = iter(self.data_loader_B_images)
        self.data_loader_A_labels_iter = iter(self.data_loader_A_labels)
        self.data_loader_B_labels_iter = iter(self.data_loader_B_labels)
        self.data_loader_AB_images_1_iter = iter(self.data_loader_AB_images_1)
        self.data_loader_AB_images_2_iter = iter(self.data_loader_AB_images_2)
        return self

    def __next__(self):
        A_image, A_image_paths= next(self.data_loader_A_images_iter)
        B_image, B_image_paths=next(self.data_loader_B_images_iter)
	
        A_label, A_label_paths=next(self.data_loader_A_labels_iter)
        B_label, B_label_paths=   next(self.data_loader_B_labels_iter)

        AB_image_1, AB_image_1_paths=next(self.data_loader_AB_images_1_iter)
        AB_image_2, AB_image_1_paths=   next(self.data_loader_AB_images_2_iter)


        return {'A_image': A_image, 'A_image_paths': A_image_paths,
                'B_image': B_image, 'B_image_paths': B_image_paths, 'A_label': A_label, 'A_label_paths': A_label_paths,
                'B_label': B_label, 'B_label_paths': B_label_paths,'AB_image_1': AB_image_1, 'AB_image_1_paths': AB_image_1_paths,
                'AB_image_2': AB_image_2, 'AB_image_1_paths': AB_image_1_paths}


    def get_pair(self,typePair,transform = True):
        if typePair is 'AC':
            A_image, A_image_paths= next(self.data_loader_A_images_iter)
            A_label, A_label_paths=next(self.data_loader_A_labels_iter)
            A_label_temp =  self.image2label(A_label[0])
            return {'image': A_image, 'image_path': A_image_paths, 'label': A_label, 'label_path': A_label_paths}
        elif typePair is 'BC':
            B_image, B_image_paths=next(self.data_loader_B_images_iter)
            B_label, B_label_paths=   next(self.data_loader_B_labels_iter)
            return {'image': B_image, 'image_path': B_image_paths, 'label': B_label, 'label_path': B_label_paths}
        elif typePair is 'AB':
            return {'imageA': AB_image_1, 'imageApath': AB_image_1_paths,
                'imageB': AB_image_2, 'imageBpath': AB_image_1_paths}

class DataLoader(BaseDataLoader):
    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        transform = transforms.Compose([
                                       transforms.Scale(opt.loadSize),
                                       transforms.CenterCrop(opt.fineSize),
                                       transforms.ToTensor()])

        target_transform= transforms.Compose([
                                       transforms.Scale(opt.loadSize),
                                       transforms.CenterCrop(opt.fineSize),transforms.ToTensor(),ToLabelTensor(labels.labels.labels),
                                    ])
        domainAdata= SegmentationDataset(root=opt.dataroot + '/' + opt.domain_A ,
                                transform=transform, target_transform=target_transform, return_paths=True)
        domainBdata= SegmentationDataset(root=opt.dataroot + '/' + opt.domain_B ,
                                transform=transform, target_transform=target_transform, return_paths=True)

       # Dataset AB
        domain_AB_images_1 = ImageFolder(root=opt.dataroot + '/' + opt.domain_A + '/images',
                                transform=transform, return_paths=True,sort=False)

        # Dataset AB
        domain_AB_images_2 = ImageFolder(root=opt.dataroot + '/' + opt.domain_B + '/images',
                                transform=transform, return_paths=True,sort=False)

        train_test_ratio_domainB= opt.ratio_domainA

        train_test_ratio_domainA= opt.ratio_domainB

        domainAdataloader = torch.utils.data.DataLoader(
            domainAdata,
            batch_size=self.opt.batchSize,
            shuffle=not self.opt.serial_batches,
            num_workers=int(self.opt.nThreads))

        domainBdataloader = torch.utils.data.DataLoader(
            domainBdata,
            batch_size=self.opt.batchSize,
            shuffle=not self.opt.serial_batches,
            num_workers=int(self.opt.nThreads))

        data_loader_AB_images_1 = torch.utils.data.DataLoader(
            domain_AB_images_1 ,
            batch_size=self.opt.batchSize,
            shuffle=not self.opt.serial_batches,
            num_workers=int(self.opt.nThreads))

        data_loader_AB_images_2  = torch.utils.data.DataLoader(
            domain_AB_images_2 ,
            batch_size=self.opt.batchSize,
            shuffle=not self.opt.serial_batches,
            num_workers=int(self.opt.nThreads))
        

        self.domainAdataloader = domainAdataloader
        self.domainBdataloader = domainBdataloader
        self.domain_AB_images_1 = domain_AB_images_1
        self.domain_AB_images_2 = domain_AB_images_2 

        self.paired_data = PairedData(data_loader_A_images, data_loader_A_labels, data_loader_B_images, data_loader_B_labels, data_loader_AB_images_1 , data_loader_AB_images_2) # self.dataset

    def name(self):
        return 'UnalignedDataLoader'

    def load_data(self):
        return self.paired_data

    def __len__(self):
        return (len(self.domain_A_images),len(domain_B_images))


    def __iter__(self):
        self.domainAdataloader_iter = iter(domainAdataloader)
        self.domainBdataloader_iter = iter(domainBdataloader)
        self.domain_AB_images_1_iter = iter(domain_AB_images_1)
        self.domain_AB_images_2_iter = iter(domain_AB_images_2)

    