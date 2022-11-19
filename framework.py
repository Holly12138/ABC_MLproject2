import torch
import torch.nn as nn
from torch.autograd import Variable as V
from loss import DiceLoss
import cv2
import numpy as np


class MyFrame():
    def __init__(self, net, loss, lr=1e-3, evalmode=False):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.net = net().to(self.device)
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=lr)

        self.loss = loss()
        self.old_lr = lr
        if evalmode:
            for i in self.net.modules():
                if isinstance(i, nn.BatchNorm2d):
                    i.eval()

    def set_input(self, img_batch, mask_batch=None, img_id=None):
        self.img = img_batch
        self.mask = mask_batch
        self.img_id = img_id

    def test_one_img_from_path(self, path):
        img = cv2.imread(path)
        img = np.array(img, np.float32)/255.0 * 3.2 - 1.6
        img = V(torch.Tensor(img).to(self.device))

        mask = self.net.forward(img).squeeze().cpu().data.numpy()
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        return mask

    def forward(self, volatile=False):
        self.img = V(self.img.to(self.device), volatile=volatile)
        if self.mask is not None:
            self.mask = V(self.mask.to(self.device), volatile=volatile)

    def optimize(self, eval=False):
        self.forward()
        if not eval:
            self.optimizer.zero_grad()
            self.net.train()
        else:
            self.net.eval()
        pred = self.net.forward(self.img)
        # pred = torch.argmax(pred, dim=1).type(torch.float32)
        # pred = torch.unsqueeze(pred,dim=1)
        # pred =torch.LongTensor(pred)
        gt=torch.squeeze(self.mask).type(torch.LongTensor).to(self.device)
        # print("!!!!!!:", pred.dtype)
        # loss = self.loss(self.mask, pred)
        pred_y=torch.argmax(pred,dim=1)
        pred_y=pred_y.to(dtype=torch.float32).requires_grad_(True)
        # print(pred_y.dtype)
        loss_dice=DiceLoss()
        loss_d =loss_dice(pred_y,gt)
        loss_fn=nn.CrossEntropyLoss()
        loss=loss_fn(pred,gt)+0.5*loss_d
        # print(loss_fn(pred,gt),loss_d)

        if not eval:
            loss.backward()
            self.optimizer.step()
        with torch.no_grad():
            pred_cpu = pred.clone()
            gt_cpu = gt.clone()
            pred_cpu = pred_cpu.cpu().detach().numpy()
            gt_cpu = gt_cpu.cpu().detach().numpy()
            dice = self.Dice(pred_cpu, gt_cpu)
        return loss.item(),dice

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path))

    def update_lr(self, new_lr, mylog, factor=False):
        if factor:
            new_lr = self.old_lr / new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        print(mylog, 'update learning rate: %f -> %f' % (self.old_lr, new_lr))
        print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
        self.old_lr = new_lr

    def Dice(self, input, target, eps=1):
        # 抹平了，弄成一维的
        input =np.argmax(input, axis=1)
        input_flatten = input.flatten()
        target_flatten = target.flatten()
        # 计算交集中的数量
        overlap = np.sum(input_flatten * target_flatten)
        # 返回值，让值在0和1之间波动
        return np.clip(((2. * overlap) / (np.sum(target_flatten) + np.sum(input_flatten) + eps)), 1e-4, 0.9999)

