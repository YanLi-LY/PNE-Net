import sys
import time
import copy
import warnings
import numpy as np

warnings.filterwarnings('ignore')

import torch.nn as nn
from torch import optim
import torch.nn.parallel
from torch.autograd import Variable
from torchvision import utils as vutils

from Utils.option import *
from Utils.utils import *
from Utils.metrics import *
from DataLoader.dataloader import *
from Models.backbone import Net as Net
from Losses.ContrastiveLoss import LossNetwork as ContrastLoss


device = torch.device(opt.device)
create_dir(opt.save_model_dir)
trainLogger = create_log(if_train=True)


def train():
    train_start_time = time.time()
    start_epoch = 0
    max_ssim_1, max_psnr_1 = 0, 0
    max_ssim_2, max_psnr_2 = 0, 0

    train_syn_dataset = syn_Dataset(opt.trainset_path, mode='train', size=opt.crop_size)
    test_dataset = test_Dataset(opt.testset_path, mode='test', size='whole img')
    train_syn_loader = DataLoader(train_syn_dataset, batch_size=opt.train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False)
    total_iters = len(train_syn_loader)

    backbone_model = Net(gps=opt.gps, blocks=opt.blocks)
    ema_model = Net(gps=opt.gps, blocks=opt.blocks)
    backbone_model = backbone_model.to(device)
    ema_model = ema_model.to(device)

    criterion = []
    criterion.append(nn.L1Loss().to(device))
    if opt.contrastloss:
        criterion.append(ContrastLoss(device).to(device))

    optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, backbone_model.parameters()), lr=opt.lr, betas=(0.9, 0.999), eps=1e-08)

    if opt.load_double and os.path.exists(opt.pre_model_path):
        print(f'load_pretrained_double_model from {opt.pre_model_path}')
        ckp = torch.load(opt.pre_model_path, map_location=torch.device(opt.device))
        start_epoch = ckp['epoch']
        optimizer.load_state_dict(ckp['optimizer'])
        backbone_model.load_state_dict(ckp['backbone_model'])
        ema_model.load_state_dict(ckp['ema_model'])
        max_ssim_1 = ckp['max_ssim_1']
        max_psnr_1 = ckp['max_psnr_1']
        max_ssim_2 = ckp['max_ssim_2']
        max_psnr_2 = ckp['max_psnr_2']
        print(f'start_epoch:{start_epoch + 1} start training ---')
        print(f'max_ssim_1: {max_ssim_1}, max_panr_1: {max_psnr_1}, max_ssim_2: {max_ssim_2}, max_psnr_2: {max_psnr_2}')

    else:
        print('train from scratch *** ')

    for param in ema_model.parameters():
        param.requires_grad = False
    for param in backbone_model.parameters():
        param.requires_grad = True

    pre_epoch_dehaze_list = {}
    now_epoch_dehaze_list = {}
    negatives = {}
    for epoch in range(start_epoch+1, opt.num_epochs+1):

        pre_epoch_dehaze_list.clear()
        pre_epoch_dehaze_list = now_epoch_dehaze_list.copy()
        now_epoch_dehaze_list.clear()

        one_epoch_start_time = time.time()
        epoch_loss = []
        backbone_model.train()

        lr = adjust_learning_rate(epoch, optimizer)
        for batch_id, data in enumerate(train_syn_loader):

            optimizer.zero_grad()

            if not opt.ema:
                hazy_img_syn, clear_img_syn = data
                hazy_img_syn = hazy_img_syn.to(device)
                clear_img_syn = clear_img_syn.to(device)

                syn_output = backbone_model(hazy_img_syn)

                l1_su = criterion[0](syn_output, clear_img_syn)
                loss = l1_su

            else:
                hazy_img_syn_1, hazy_img_syn_2, clear_img_syn, aux_negs, id, extension = data

                hazy_img_syn_1 = hazy_img_syn_1.to(device)
                hazy_img_syn_2 = hazy_img_syn_2.to(device)
                clear_img_syn = clear_img_syn.to(device)
                aux_negs = aux_negs.to(device)

                a1_out = backbone_model(hazy_img_syn_1)
                with torch.no_grad():
                    a2_out = ema_model(hazy_img_syn_2)

                # syn consist loss
                syn_consist_loss = criterion[0](a1_out, a2_out.detach_())

                # syn l1 loss
                syn_l1_loss = criterion[0](a1_out, clear_img_syn)

                # syn contrast loss
                if epoch == start_epoch+1:
                    pre_dehaze_1 = hazy_img_syn_1
                    pre_dehaze_2 = hazy_img_syn_2
                    syn_contrast_loss = criterion[1](a1_out, clear_img_syn, pre_dehaze_1, pre_dehaze_2, aux_negs)
                    for j in range(opt.train_batch_size):
                        negatives[str(id[j]) + '_1'] = pre_dehaze_1[j, :, :, :].unsqueeze(0).detach().cpu()
                        negatives[str(id[j]) + '_2'] = pre_dehaze_2[j, :, :, :].unsqueeze(0).detach().cpu()
                else:
                    syn_contrast_loss_list = []
                    for i in range(opt.train_batch_size):
                        pre_neg_1 = negatives.get(str(id[i]) + '_1')
                        pre_neg_1 = pre_neg_1.to(device)
                        pre_neg_2 = negatives.get(str(id[i]) + '_2')
                        pre_neg_2 = pre_neg_2.to(device)

                        pre_dehaze_1 = pre_epoch_dehaze_list.get(str(id[i]) + '_1')
                        pre_dehaze_2 = pre_epoch_dehaze_list.get(str(id[i]) + '_2')
                        pre_dehaze_1 = pre_dehaze_1.to(device)
                        pre_dehaze_2 = pre_dehaze_2.to(device)

                        ssim_neg_gt_1 = ssim(pre_neg_1, clear_img_syn)
                        ssim_dehaze_gt_1 = ssim(pre_dehaze_1, clear_img_syn)
                        psnr_neg_gt_1 = psnr(pre_neg_1, clear_img_syn)
                        psnr_dehaze_gt_1 = psnr(pre_dehaze_1, clear_img_syn)
                        ssim_neg_gt_2 = ssim(pre_neg_2, clear_img_syn)
                        ssim_dehaze_gt_2 = ssim(pre_dehaze_2, clear_img_syn)
                        psnr_neg_gt_2 = psnr(pre_neg_2, clear_img_syn)
                        psnr_dehaze_gt_2 = psnr(pre_dehaze_2, clear_img_syn)

                        if (ssim_dehaze_gt_1 < ssim_neg_gt_1 and psnr_dehaze_gt_1 < psnr_neg_gt_1) and (ssim_dehaze_gt_2 < ssim_neg_gt_2 and psnr_dehaze_gt_2 < psnr_neg_gt_2):
                            syn_contrast_loss_list.append(criterion[1](a1_out[i, :, :, :].unsqueeze(0), clear_img_syn[i, :, :, :].unsqueeze(0), pre_dehaze_1, pre_dehaze_2, aux_negs[i, :, :, :].unsqueeze(0)))
                            syn_contrast_loss = 1 / opt.train_batch_size * sum(syn_contrast_loss_list)
                            negatives = pre_epoch_dehaze_list.copy()
                        else:
                            syn_contrast_loss_list.append(criterion[1](a1_out[i, :, :, :].unsqueeze(0), clear_img_syn[i, :, :, :].unsqueeze(0), pre_neg_1, pre_neg_2, aux_negs[i, :, :, :].unsqueeze(0)))
                            syn_contrast_loss = 1 / opt.train_batch_size * sum(syn_contrast_loss_list)

                # syn total loss
                syn_loss = syn_consist_loss + syn_l1_loss + 0.01 * syn_contrast_loss

                # model total loss
                total_loss = syn_loss

                for j in range(opt.train_batch_size):
                    now_epoch_dehaze_list[str(id[j]) + '_1'] = a1_out[j, :, :, :].unsqueeze(0).detach().cpu()
                    now_epoch_dehaze_list[str(id[j]) + '_2'] = a2_out[j, :, :, :].unsqueeze(0).detach().cpu()


            epoch_loss.append(total_loss.item())
            total_loss.backward()
            optimizer.step()

            if opt.ema:
                # if max_ssim_1 > 0.45 and max_psnr_1 > 14.5:
                #     opt.momentum = 0.9
                state_dict_backbone = backbone_model.state_dict()
                state_dict_ema_model = ema_model.state_dict()
                for (k_backbone, v_backbone), (k_ema, v_ema) in zip(state_dict_backbone.items(), state_dict_ema_model.items()):
                    assert k_backbone == k_ema
                    assert v_backbone.shape == v_ema.shape
                    if 'num_batches_tracked' in k_ema:
                        v_ema.copy_(v_backbone)
                    else:
                        v_ema.copy_(v_ema * opt.momentum + (1. - opt.momentum) * v_backbone) # momentum=0.999

            if not opt.ema:
                print(f'\rtrain loss : {total_loss.item():.5f} | syn_loss : {syn_loss.item():.5f}'
                    f'| epoch : {epoch}/{opt.num_epochs} | iter : {batch_id + 1}/{total_iters} | lr : {lr :.5f} | time_used : {(time.time() - train_start_time) / 3600 :.1f} hour',
                    end='', flush=True)
            else:
                print(f'\rtrain loss : {total_loss.item():.5f} | syn_loss : {syn_loss.item():.5f}'
                    f'| epoch : {epoch}/{opt.num_epochs} | iter : {batch_id + 1}/{total_iters} | lr : {lr :.5f} | time_used : {(time.time() - train_start_time) / 3600 :.1f} hour', end='', flush=True)

            if epoch % opt.eval_epoch == 0 and (batch_id + 1) == total_iters:
                one_epoch_time = (time.time() - one_epoch_start_time) / 3600
                avg_loss = np.mean(epoch_loss)
                with torch.no_grad():
                    ssim_1_eval, psnr_1_eval, ssim_2_eval, psnr_2_eval = test(backbone_model, ema_model, test_loader)


                print(f'\nepoch : {epoch}  | epoch train loss : {avg_loss.item():.5f} | lr : {lr :.5f} | epoch {epoch} | train time : {one_epoch_time :.1f} hour\n'
                    f'| ssim_1 : {ssim_1_eval:.4f} | psnr_1 : {psnr_1_eval:.4f} | ssim_2 : {ssim_2_eval:.4f} | psnr_2 : {psnr_2_eval:.4f}')
                trainLogger.write(f'\nepoch : {epoch}  | epoch train loss : {avg_loss.item():.5f} | lr : {lr :.5f} | epoch {epoch} | train time : {one_epoch_time :.1f} hour\n'
                    f'| ssim_1 : {ssim_1_eval:.4f} | psnr_1 : {psnr_1_eval:.4f} | ssim_2 : {ssim_2_eval:.4f} | psnr_2 : {psnr_2_eval:.4f}')

                torch.save({
                    'epoch': epoch,
                    'max_psnr_1': max_psnr_1,
                    'max_ssim_1': max_ssim_1,
                    'max_psnr_2': max_psnr_2,
                    'max_ssim_2': max_ssim_2,
                    'optimizer': optimizer.state_dict(),
                    'backbone_model': backbone_model.state_dict(),
                    'ema_model': ema_model.state_dict()
                }, opt.save_last_model_path)


                if ssim_1_eval >= max_ssim_1 and psnr_1_eval >= max_psnr_1:
                    max_ssim_1 = max(max_ssim_1, ssim_1_eval)
                    max_psnr_1 = max(max_psnr_1, psnr_1_eval)
                    if opt.ema:
                        torch.save({
                            'epoch': epoch,
                            'max_psnr_1': max_psnr_1,
                            'max_ssim_1': max_ssim_1,
                            'max_psnr_2': max_psnr_2,
                            'max_ssim_2': max_ssim_2,
                            'optimizer': optimizer.state_dict(),
                            'backbone_model': backbone_model.state_dict(),
                            'ema_model': ema_model.state_dict()
                        }, opt.save_best_backmodel_path)
                        print(f'\n>>>>>>>>>>>>>>>best backbone model saved at epoch : {epoch} | max_ssim_1 : {max_ssim_1:.4f} | max_psnr_1 : {max_psnr_1:.4f} | max_ssim_2 : {max_ssim_2:.4f} | max_psnr_2 : {max_psnr_2:.4f}')
                        trainLogger.write(f'\n>>>>>>>>>>>>>>>best backbone model saved at epoch : {epoch} | max_ssim_1 : {max_ssim_1:.4f} | max_psnr_1 : {max_psnr_1:.4f} | max_ssim_2 : {max_ssim_2:.4f} | max_psnr_2 : {max_psnr_2:.4f}')

                if ssim_2_eval >= max_ssim_2 and psnr_2_eval >= max_psnr_2 :
                    max_ssim_2 = max(max_ssim_2, ssim_2_eval)
                    max_psnr_2 = max(max_psnr_2, psnr_2_eval)
                    if opt.ema:
                        torch.save({
                                    'epoch': epoch,
                                    'max_psnr_1': max_psnr_1,
                                    'max_ssim_1': max_ssim_1,
                                    'max_psnr_2': max_psnr_2,
                                    'max_ssim_2': max_ssim_2,
                                    'optimizer': optimizer.state_dict(),
                                    'backbone_model': backbone_model.state_dict(),
                                    'ema_model': ema_model.state_dict()
                        }, opt.save_best_eammodel_path)
                        print(f'\n>>>>>>>>>>>>>>>best ema model saved at epoch : {epoch} | max_ssim_1 : {max_ssim_1:.4f} | max_psnr_1 : {max_psnr_1:.4f} | max_ssim_2 : {max_ssim_2:.4f} | max_psnr_2 : {max_psnr_2:.4f}')
                        trainLogger.write(f'\n>>>>>>>>>>>>>>>best ema model saved at epoch : {epoch} | max_ssim_1 : {max_ssim_1:.4f} | max_psnr_1 : {max_psnr_1:.4f} | max_ssim_2 : {max_ssim_2:.4f} | max_psnr_2 : {max_psnr_2:.4f}')


def test(backbone_model, ema_model, test_loader):
    backbone_model.eval()
    ema_model.eval()
    with torch.cuda.device(opt.device):
        torch.cuda.empty_cache()
    ssims_1, psnrs_1 = [], []
    ssims_2, psnrs_2 = [], []
    for val_batch_id, test_data in enumerate(test_loader):
        input, target, name = test_data
        input = input.to(device)
        target = target.to(device)
        if opt.ema:
            pred_1 = backbone_model(input)
            pred_2 = ema_model(input)
            ssim_1 = ssim(pred_1, target).item()
            psnr_1 = psnr(pred_1, target)
            ssim_2 = ssim(pred_2, target).item()
            psnr_2 = psnr(pred_2, target)
            ssims_1.append(ssim_1)
            psnrs_1.append(psnr_1)
            ssims_2.append(ssim_2)
            psnrs_2.append(psnr_2)

        else:
            pred_1 = backbone_model(input)
            ssim_1 = ssim(pred_1, target).item()
            psnr_1 = psnr(pred_1, target)
            ssims_1.append(ssim_1)
            psnrs_1.append(psnr_1)

    return np.mean(ssims_1), np.mean(psnrs_1), np.mean(ssims_2), np.mean(psnrs_2)


if __name__ == '__main__':

    trainLogger.write('\n')
    trainLogger.write('=' * 100)
    opt_k = vars(opt)
    for k in opt_k.keys():
        trainLogger.write(f'\n---{k}: {opt_k[k]}')
    trainLogger.write('\n')
    trainLogger.write('=' * 100)
    trainLogger.write('\n')

    train()
    close_log(trainLogger)