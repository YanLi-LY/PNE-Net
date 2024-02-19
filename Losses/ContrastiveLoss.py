import numpy
import torch
import torch.nn.functional as F
from torchvision import models

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X) # relu1_1
        h_relu2 = self.slice2(h_relu1) # relu2_1
        h_relu3 = self.slice3(h_relu2) # relu3_1
        h_relu4 = self.slice4(h_relu3) # relu4_1
        h_relu5 = self.slice5(h_relu4) # relu5_1
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

# --- Perceptual loss network  --- #
class LossNetwork(torch.nn.Module):
    def __init__(self, device):
        super(LossNetwork, self).__init__()
        self.vgg = Vgg19().to(device)
        self.L1 = torch.nn.L1Loss()
        self.weight = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self, dehaze, gt, neg1, neg2, aux_negs):
        loss = []
        dehaze_features = self.vgg(dehaze)
        gt_features = self.vgg(gt)
        neg1_features = self.vgg(neg1)
        neg2_features = self.vgg(neg2)
        aux_negs_features = self.vgg(aux_negs)
        for i in range(len(dehaze_features)):
            dehaze_gt = torch.exp(-self.L1(dehaze_features[i], gt_features[i]) * 1 / 2)
            dehaze_neg1 = torch.exp(-self.L1(dehaze_features[i], neg1_features[i]) * 1 / 2)
            dehaze_neg2 = torch.exp(-self.L1(dehaze_features[i], neg2_features[i]) * 1 / 2)
            dehaze_aux_negs = torch.exp(-self.L1(dehaze_features[i], aux_negs_features[i]) * 1 / 2)

            per_loss = -torch.log(dehaze_gt / (dehaze_gt + dehaze_neg1 + dehaze_neg2 + dehaze_aux_negs))
            loss.append(self.weight[i] * per_loss)

        return sum(loss)
