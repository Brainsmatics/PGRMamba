import torch
from torch import nn
import numpy as np

class GCA_Loss():
    def __init__(self, outputs, label, device, lambda_conf=10, thread=10, pred_conf_thread=0.5):
        super(GCA_Loss, self).__init__()
        self.outputs = outputs
        self.label = label
        self.labmda_conf = lambda_conf
        self.thread = thread
        self.device = device
        self.pred_conf_thread = pred_conf_thread

    def focal_bce_loss(self, pred, target, alpha=0.90, gamma=2.0):
        bce_loss = nn.BCELoss(reduction='none')(pred, target) 
        p_t = target * pred + (1 - target) * (1 - pred)
        focal_weight = (1 - p_t) ** gamma
        alpha_factor = target * alpha + (1 - target) * (1 - alpha)
        return alpha_factor * focal_weight * bce_loss

    def _get_target(self, label, output_shape=None, thread=1):
        if output_shape is None:
            output_shape = (64, 64, 64)
        depth, height, width = output_shape
        mask = torch.zeros([depth, height, width])
        noobj_mask = torch.ones([depth, height, width])
        tx = torch.zeros([depth, height, width])
        ty = torch.zeros([depth, height, width])
        tz = torch.zeros([depth, height, width])
        if label is not None:
            label = np.array(label)
            if label.size > 0:
                if label.ndim == 1:
                    label = label.reshape(1, -1)
                # print(label)
                for j in range(label.shape[0]):
                    if label[j][2] == 256:
                        label[j][2] -= 0.1
                    if label[j][1] >256:
                        label[j][1] -= 0.1
                    if label[j][0] >256:
                        label[j][0] -= 0.1
                    x, y, z = int(label[j][0]/4), int(label[j][1]/4), int(label[j][2]/4)
                    z = max(0, min(z, depth - 1))
                    x = max(0, min(x, height - 1))
                    y = max(0, min(y, width - 1))
                    mask[z, x, y] = 1

                    tx[z, x, y] = float(label[j][0]/4 - x)
                    ty[z, x, y] = float(label[j][1]/4 - y)
                    tz[z, x, y] = float(label[j][2]/4 - z)
                    noobj_mask[z, x, y] = 0

        return mask, noobj_mask, tx, ty, tz

    def _get_distance(self, pred, label0):
        label0_expand = label0.to(pred.device).expand_as(pred)
        d1 = torch.square(label0_expand-pred)
        distance = torch.sqrt(torch.sum(d1, 1))
        return distance



    def _get_ignore0(self, noobj_mask, label, output, scale_factor = [0.65, 2]):
        if label is not None:
            label = np.array(label)
            if label.size == 0:
                return noobj_mask
            if label.ndim == 1:
                label = label.reshape(1, -1)
            tx = torch.sigmoid(output[0])
            ty = torch.sigmoid(output[1])
            tz = torch.sigmoid(output[2])
            depth, height, width = tx.shape
            device = tx.device
            grid_x = torch.arange(height, device=device).view(1, height, 1).expand(depth, height, width)
            grid_y = torch.arange(width, device=device).view(1, 1, width).expand(depth, height, width)
            grid_z = torch.arange(depth, device=device).view(depth, 1, 1).expand(depth, height, width)
            pred_x, pred_y, pred_z = (tx+grid_x)*4*scale_factor[0], (ty+grid_y)*4*scale_factor[0], (tz+grid_z)*4*scale_factor[1]
            pred = torch.stack([pred_x.view(-1), pred_y.view(-1), pred_z.view(-1)]).T
            label0 = torch.tensor(label)/1.0
            label0 = label0[:,:3]
            for i in range(label0.shape[0]):
                distance = self._get_distance(pred, label0[i])
                distance = distance.view(tx.shape)
                mask = distance < self.thread
                noobj_mask[mask] = 0
        return noobj_mask



    def forward(self, input, target = None):
        pass

    def get_loss(self):
        device = self.device
        loss = 0
        total_loss_x, total_loss_y, total_loss_z = 0, 0, 0
        total_loss_conf = 0
        alpha = 1.9 
        delta = 3.0

        for i in range(self.outputs.shape[0]):
            label0 = self.label[i]
            output_i = self.outputs[i]
            output_shape = output_i.shape[1:]
            mask, noobj_mask, tx, ty, tz = self._get_target(label0, output_shape=output_shape)
            noobj_mask = self._get_ignore0(noobj_mask, label0, output_i)

            tx, ty, tz = tx.to(device), ty.to(device), tz.to(device)
            mask, noobj_mask = mask.to(device), noobj_mask.to(device)

            pred_x, pred_y, pred_z = torch.sigmoid(output_i[0]), torch.sigmoid(output_i[1]), torch.sigmoid(output_i[2])
            conf = torch.sigmoid(output_i[3])
            loss_x = nn.BCELoss()(pred_x*mask, tx*mask)
            loss_y = nn.BCELoss()(pred_y*mask, ty*mask)
            loss_z = nn.BCELoss()(pred_z*mask, tz*mask)
            
            reg_loss_base = (loss_x + loss_y + loss_z)

            
            with torch.no_grad():
                dist_sq = (pred_x - tx)**2 + (pred_y - ty)**2 + (pred_z - tz)**2 
                quality_target = torch.exp(-torch.sqrt(dist_sq + 1e-7)) * mask

                current_batch_mse = dist_sq[mask.bool()].mean() if mask.sum() > 0 else torch.tensor(0.0).to(device)
                if not hasattr(self, 'running_avg_mse'): self.running_avg_mse = torch.tensor(0.05).to(device)
                self.running_avg_mse = 0.9 * self.running_avg_mse + 0.1 * current_batch_mse.detach()
                beta = dist_sq / (self.running_avg_mse + 1e-7)
                r = beta / (alpha * (delta ** (beta - alpha)))

            weighted_reg_loss = (r * reg_loss_base * mask).sum() / (mask.sum() + 1e-6)

            loss_conf_pos = self.focal_bce_loss(conf * mask, quality_target)
            loss_conf_neg = self.focal_bce_loss(conf * noobj_mask, torch.zeros_like(conf))
            loss_conf_sum = (loss_conf_pos.sum() + loss_conf_neg.sum()) / (mask.sum() + noobj_mask.sum() + 1.0)

            loss += (weighted_reg_loss  + self.labmda_conf * loss_conf_sum)
            
            total_loss_x += loss_x
            total_loss_y += loss_y
            total_loss_z += loss_z
            total_loss_conf += loss_conf_sum
            
        return loss, total_loss_x, total_loss_y, total_loss_z, total_loss_conf





