import os
import warnings
import argparse


warnings.filterwarnings('ignore')

parser=argparse.ArgumentParser()
# path
parser.add_argument('--trainset_name', type=str, default='')
parser.add_argument('--testset_name', type=str, default='')
parser.add_argument('--trainset_path', type=str, default='')
parser.add_argument('--testset_path', type=str, default='')
parser.add_argument('--pre_model_path', type=str, default='', help='the pre_train model .pk')
parser.add_argument('--save_model_dir', type=str, default='./Checkpoints/', help='save train model dir.pk')
parser.add_argument('--save_best_model_path', type=str, default='', help='save train model path .pk')
parser.add_argument('--save_last_model_path', type=str, default='', help='save train model path .pk')
parser.add_argument('--load_double', type=str, default=True)
parser.add_argument('--test_model_path', type=str, default='', help='the pre_train model .pk')

# hyper-parameters
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--lr_sche', default=True)
parser.add_argument('--gps', type=int, default=3, help='residual_groups')
parser.add_argument('--blocks', type=int, default=20, help='residual_blocks: increase the number of blocks from 19 to 20')
parser.add_argument('--contrastloss', type=str, default=True, help='Contrastive Loss')
parser.add_argument('--consistloss', type=str, default=False)
parser.add_argument('--ema', type=str, default=True, help='use ema')
parser.add_argument('--momentum', type=float, default=0.999, help='ema decay rate')

# self-defined parameters
parser.add_argument('--num_epochs', type=int, default=1000)
parser.add_argument('--eval_epoch', type=int, default=1)
parser.add_argument('--train_batch_size', type=int, default=3)
parser.add_argument('--test_batch_size', type=int, default=1)
parser.add_argument('--device', type=str, default='cuda:1')

parser.add_argument('--crop', type=str, default=True)
parser.add_argument('--crop_size', type=int, default=240)


opt = parser.parse_args()
opt.save_best_eammodel_path = os.path.join(opt.save_model_dir, f'{opt.trainset_name}_best_emamodel.pk')
opt.save_best_backmodel_path = os.path.join(opt.save_model_dir, f'{opt.trainset_name}_best_backmodel.pk')
opt.save_last_model_path = os.path.join(opt.save_model_dir, f'{opt.trainset_name}_last_model.pk')

