from  cycle_pix2pix_model import CycleGANPix2PixModel
from cycle_classification_model import CycleGANClassificationModel
from domain_adversarial_classification_model import WildDomainAdaptationModel
from fcn_model import FCNModel
def create_model(opt):
    model = None
    if opt.model=='fcnwild':
        model = WildDomainAdaptationModel()
        model.initialize(opt)
        print("model [%s] was created" % (model.name()))
        return model
    if opt.model=='cycle_gan_seg':
        model=CycleGANClassificationModel()
        model.initialize(opt)
        print("model [%s] was created" % (model.name()))
        return model
    if opt.model=='fcnonly':
        model=FCNModel()
        model.initialize(opt)
        print("model [%s] was created" % (model.name()))
        return model
    # print(opt.model)
    # if opt.model == 'cycle_gan':
    #     from .cycle_gan_model import CycleGANModel
    #     assert(opt.align_data == False)
    #     model = CycleGANModel()
    # if opt.model == 'pix2pix':
    #     from .pix2pix_model import Pix2PixModel
    #     assert(opt.align_data == True)
    #     model = Pix2PixModel()
    # model.initialize(opt)
    # print("model [%s] was created" % (model.name()))
    # return model

