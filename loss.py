import torch.nn as nn
import torch as tc
import numpy as np
import math

class DistanceBasedLogisticLoss(nn.Module):
    def __init__(self, num_pos: int, num_imgs: int):
        super(DistanceBasedLogisticLoss, self).__init__()

        self.num_pos = num_pos
        self.num_imgs = num_imgs
        self.tot_patch = num_pos * num_imgs

    def forward(self, y_pred, y_true):
        y_pred_flat = tc.reshape(y_pred, [self.tot_patch, -1])
        #print(y_pred_flat.shape)
        A = y_pred_flat
        B = tc.transpose(A, 0, 1)

        na = tc.tile(tc.unsqueeze(tc.sum(tc.square(A), dim=1), -1), [1, self.tot_patch])
        nb = tc.tile(tc.unsqueeze(tc.sum(tc.square(B), dim=0), 0), [self.tot_patch, 1])

        l2norm = na - tc.tensor(2.)*tc.matmul(A, B) + nb
        eps = 1e-8

        l2norm = tc.where(tc.eye(l2norm.shape[0]).bool(), tc.full_like(l2norm, float('inf')), l2norm)
        #print(l2norm)
        l2norm_softmax = tc.exp(-l2norm) / tc.tile(tc.unsqueeze(tc.sum(tc.exp(-l2norm), dim=1) + tc.tensor(eps), -1), [1, self.tot_patch])
        #print(l2norm_softmax.shape)
        l2norm_vec = tc.reshape(l2norm_softmax, [-1])
        #print(l2norm_vec.shape)
        y_true_vec = tc.reshape(y_true, [-1])


        #print(l2norm_vec.size())
        #print(y_true_vec.size())
        check_labels = l2norm_vec * y_true_vec
        dbl = tc.sum(-tc.log(tc.sum(tc.reshape(check_labels, [self.tot_patch, self.tot_patch]), dim=1) + tc.tensor(eps)))
        if math.isnan(dbl):
            print("dbl is nan")
            dbl = dbl
        return dbl


