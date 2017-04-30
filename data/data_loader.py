import combined_data_loader.CombinedDataLoader
def CreateDataLoader(opt):
    data_loader=CombinedDataLoader()
    data_loader.initialize(opt)
    return data_loader
    # data_loader = None
    # if opt.domain_adapt_flag==1:
    #     from data.combined_data_loader import CombinedDataLoader
    #     data_loader=CombinedDataLoader()
    # if opt.align_data > 0:
    #     from data.aligned_data_loader import AlignedDataLoader
    #     data_loader = AlignedDataLoader()
    # else:
    #     from data.unaligned_data_loader import UnalignedDataLoader
    #     data_loader = UnalignedDataLoader()
    # print(data_loader.name())
    # data_loader.initialize(opt)
    # return data_loader
