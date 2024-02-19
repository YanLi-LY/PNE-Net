import time

import torch
from torchvision import utils as vutils
import torch.nn.parallel
import torch.nn as nn

from Utils.metrics import *
from Utils.option import *
from Utils.utils import *
from DataLoader.dataloader import *
from Models.backbone import Net as Net

device = torch.device(opt.device)
trainLogger = create_log(if_train=False)
save_testResults_dir = f'./Output/{opt.testset_name}_testResults/'
create_dir(save_testResults_dir)

def test():
    test_dataset = test_Dataset(opt.testset_path, mode='test', size='whole img')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    backbone_model = Net(gps=opt.gps, blocks=opt.blocks)
    ema_model = Net(gps=opt.gps, blocks=opt.blocks)
    backbone_model = backbone_model.to(device)
    ema_model = ema_model.to(device)

    ckp = torch.load(opt.test_model_path)
    backbone_model.load_state_dict(ckp['backbone_model'])
    ema_model.load_state_dict(ckp['ema_model'])

    backbone_model.eval()
    ema_model.eval()
    torch.cuda.empty_cache()
    ssims_1, psnrs_1 = [], []
    ssims_2, psnrs_2 = [], []
    for test_batch_id, test_data in enumerate(test_loader):
        input, target, name = test_data
        input = input.to(device)
        target = target.to(device)

        with torch.no_grad():
            pred_1 = backbone_model(input)
            pred_2 = ema_model(input)
        per_ssim_1 = ssim(pred_1, target).item()
        per_psnr_1 = psnr(pred_1, target)
        ssims_1.append(per_ssim_1)
        psnrs_1.append(per_psnr_1)

        per_ssim_2 = ssim(pred_2, target).item()
        per_psnr_2 = psnr(pred_2, target)
        ssims_2.append(per_ssim_2)
        psnrs_2.append(per_psnr_2)

        trainLogger.write(f'\n{name}_ssim_1 : {per_ssim_1:.4f} | {name}_psnr_1 : {per_psnr_1:.4f}\n'
                          f'{name}_ssim_2 : {per_ssim_2:.4f} | {name}_psnr_2 : {per_psnr_2:.4f}')

        vutils.save_image(torch.cat((input, pred_1, pred_2, target), 0), os.path.join(save_testResults_dir, f'{name}.png'), normalize=True)

    avg_ssim_1 = np.mean(ssims_1)
    avg_psnr_1 = np.mean(psnrs_1)
    avg_ssim_2 = np.mean(ssims_2)
    avg_psnr_2 = np.mean(psnrs_2)
    print(f'\navg_ssim_1 : {avg_ssim_1:.4f} | avg_psnr_1 : {avg_psnr_1:.4f}\n'
          f'avg_ssim_2 : {avg_ssim_2:.4f} | avg_psnr_2 : {avg_psnr_2:.4f}')
    trainLogger.write(f'\navg_ssim_1 : {avg_ssim_1:.4f} | avg_psnr_1 : {avg_psnr_1:.4f}\n'
                      f'avg_ssim_2 : {avg_ssim_2:.4f} | avg_psnr_2 : {avg_psnr_2:.4f}')


if __name__ == "__main__":
    test()
    trainLogger.close()





