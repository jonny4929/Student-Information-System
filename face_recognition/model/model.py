import torch
import torch.nn as nn
import torch.nn.functional as functional
import torchvision.models as models
import numpy
from model.rpn.generate_anchors import generate_anchors
from model.utils.bbox import bbox_transform_inv ,clip_boxes, clip_boxes_batch
from model.utils.bbox import bbox_overlaps_batch , bbox_transform_batch
from model.utils.config import cfg
from model.nms import NMS
torch.set_default_tensor_type(torch.FloatTensor)
device=torch.device('cuda:0')
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torchvision import models
class SSH(nn.Module):
    def __init__(self):
        super(SSH,self).__init__()
        self.baseline=SSH_vgg16base()
        self.m1_anchor_target_layer=anchor_target_layer(8, numpy.array([1,2]), numpy.array([1, ]), 0, name='m1')
        self.m2_anchor_target_layer=anchor_target_layer(16, numpy.array([4,8]), numpy.array([1, ]), 0, name='m2')
        self.m3_anchor_target_layer=anchor_target_layer(32, numpy.array([16,32]), numpy.array([1, ]), 0, name='m3')
        self.m1_proposallayer=proposal_layer(8, numpy.array([1, 2]), numpy.array([1, ]))
        self.m2_proposallayer=proposal_layer(16, numpy.array([4, 8]), numpy.array([1, ]))
        self.m3_proposallayer=proposal_layer(32, numpy.array([16, 32]), numpy.array([1, ]))

        self.softmax1=nn.Softmax(1)
        self.softmax2=nn.Softmax(1)
        self.softmax3=nn.Softmax(1)

    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        return x

    def forward(self,imgdata,im_size,gt_boxes=None,training=False):
        batch_size=imgdata.shape[0]
        self.baseline=self.baseline.to(device)
        imgdata=imgdata.to(device)
        m1_ssh_cls_score, m1_ssh_bbox_pred,m2_ssh_cls_score, m2_ssh_bbox_pred,m3_ssh_cls_score, m3_ssh_bbox_pred=self.baseline(imgdata)
        if training:

            m1_ssh_cls_score, m1_ssh_bbox_pred,m2_ssh_cls_score, m2_ssh_bbox_pred,m3_ssh_cls_score, m3_ssh_bbox_pred=m1_ssh_cls_score.to(torch.device('cpu')), m1_ssh_bbox_pred.to(torch.device('cpu')),m2_ssh_cls_score.to(torch.device('cpu')), m2_ssh_bbox_pred.to(torch.device('cpu')),m3_ssh_cls_score.to(torch.device('cpu')), m3_ssh_bbox_pred.to(torch.device('cpu'))
            m3_ssh_cls_score_reshape_OHEM = self.reshape(m3_ssh_cls_score.detach(), 2)
            m2_ssh_cls_score_reshape_OHEM = self.reshape(m2_ssh_cls_score.detach(), 2)
            m1_ssh_cls_score_reshape_OHEM = self.reshape(m1_ssh_cls_score.detach(), 2)

            # softmax
            m3_ssh_cls_prob_output_OHEM = self.softmax1(m3_ssh_cls_score_reshape_OHEM)
            m2_ssh_cls_prob_output_OHEM = self.softmax2(m2_ssh_cls_score_reshape_OHEM)
            m1_ssh_cls_prob_output_OHEM = self.softmax3(m1_ssh_cls_score_reshape_OHEM)

            # reshape from (batch,2,2*H,W) back to (batch,4,h,w)
            m3_ssh_cls_prob_reshape_OHEM = self.reshape(m3_ssh_cls_prob_output_OHEM, 4)
            m2_ssh_cls_prob_reshape_OHEM = self.reshape(m2_ssh_cls_prob_output_OHEM, 4)
            m1_ssh_cls_prob_reshape_OHEM = self.reshape(m1_ssh_cls_prob_output_OHEM, 4)

            m3_labels, m3_bbox_targets, m3_bbox_inside_weights, m3_bbox_outside_weights = \
                self.m3_anchor_target_layer(m3_ssh_cls_score, gt_boxes, im_size, m3_ssh_cls_prob_reshape_OHEM)

            m2_labels, m2_bbox_targets, m2_bbox_inside_weights, m2_bbox_outside_weights = \
                self.m2_anchor_target_layer(m2_ssh_cls_score, gt_boxes, im_size, m2_ssh_cls_prob_reshape_OHEM)

            m1_labels, m1_bbox_targets, m1_bbox_inside_weights, m1_bbox_outside_weights = \
                self.m1_anchor_target_layer(m1_ssh_cls_score, gt_boxes, im_size, m1_ssh_cls_prob_reshape_OHEM)
            
            m3_ssh_cls_score_reshape = self.reshape(m3_ssh_cls_score, 2)
            m2_ssh_cls_score_reshape = self.reshape(m2_ssh_cls_score, 2)
            m1_ssh_cls_score_reshape = self.reshape(m1_ssh_cls_score, 2)

            m3_ssh_cls_score = m3_ssh_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
            m2_ssh_cls_score = m2_ssh_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
            m1_ssh_cls_score = m1_ssh_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)

            m3_target_labels = m3_labels.view(batch_size, -1)
            m2_target_labels = m2_labels.view(batch_size, -1)
            m1_target_labels = m1_labels.view(batch_size, -1)

            m3_ssh_cls_score = m3_ssh_cls_score.view(-1, 2)
            m2_ssh_cls_score = m2_ssh_cls_score.view(-1, 2)
            m1_ssh_cls_score = m1_ssh_cls_score.view(-1, 2)

            m3_target_labels = m3_target_labels.view(-1).long()
            m2_target_labels = m2_target_labels.view(-1).long()
            m1_target_labels = m1_target_labels.view(-1).long()

            m3_ssh_cls_loss = functional.cross_entropy(m3_ssh_cls_score, m3_target_labels, ignore_index=-1)
            m2_ssh_cls_loss = functional.cross_entropy(m2_ssh_cls_score, m2_target_labels, ignore_index=-1)
            m1_ssh_cls_loss = functional.cross_entropy(m1_ssh_cls_score, m1_target_labels, ignore_index=-1)

            m3_bbox_loss = _smooth_l1_loss(m3_ssh_bbox_pred, m3_bbox_targets,
                                               m3_bbox_inside_weights, m3_bbox_outside_weights, sigma=3, dim=[1, 2, 3])
            m2_bbox_loss = _smooth_l1_loss(m2_ssh_bbox_pred, m2_bbox_targets,
                                               m2_bbox_inside_weights, m2_bbox_outside_weights, sigma=3, dim=[1, 2, 3])
            m1_bbox_loss = _smooth_l1_loss(m1_ssh_bbox_pred, m1_bbox_targets,
                                               m1_bbox_inside_weights, m1_bbox_outside_weights, sigma=3, dim=[1, 2, 3])

            return m3_ssh_cls_loss, m2_ssh_cls_loss, m1_ssh_cls_loss, m3_bbox_loss, m2_bbox_loss, m1_bbox_loss
        else:
            # reshape from (batch,4,h,w) to (batch,2,-1,w)
            m3_ssh_cls_score_reshape = self.reshape(m3_ssh_cls_score, 2)
            m2_ssh_cls_score_reshape = self.reshape(m2_ssh_cls_score, 2)
            m1_ssh_cls_score_reshape = self.reshape(m1_ssh_cls_score, 2)

            # softmax
            m3_ssh_cls_prob_output = self.softmax3(m3_ssh_cls_score_reshape)
            m2_ssh_cls_prob_output = self.softmax2(m2_ssh_cls_score_reshape)
            m1_ssh_cls_prob_output = self.softmax1(m1_ssh_cls_score_reshape)

            # reshape from (batch,2,2*H,W) back to (batch,4,h,w)
            m3_ssh_cls_prob_reshape = self.reshape(m3_ssh_cls_prob_output, 4)
            m2_ssh_cls_prob_reshape = self.reshape(m2_ssh_cls_prob_output, 4)
            m1_ssh_cls_prob_reshape = self.reshape(m1_ssh_cls_prob_output, 4)

            # roi has shape of (batch, top_k, 5)
            # where (batch, top_k, 4) is cls score and
            # (batch, top_k, 0:4) is bbox coordinated
            m3_ssh_roi = self.m3_proposallayer(m3_ssh_cls_prob_reshape.to(torch.device('cpu')), m3_ssh_bbox_pred.to(torch.device('cpu')), im_size)
            m2_ssh_roi = self.m2_proposallayer(m2_ssh_cls_prob_reshape.to(torch.device('cpu')), m2_ssh_bbox_pred.to(torch.device('cpu')), im_size)
            m1_ssh_roi = self.m1_proposallayer(m1_ssh_cls_prob_reshape.to(torch.device('cpu')), m1_ssh_bbox_pred.to(torch.device('cpu')), im_size)

            ssh_roi = torch.cat((m3_ssh_roi, m2_ssh_roi, m1_ssh_roi), dim=1)
            # ssh_roi = torch.cat((m3_ssh_roi,), dim=1)
            return ssh_roi

class SSH_vgg16base(nn.Module):
    def __init__(self):
        super(SSH_vgg16base,self).__init__()
        vgg=models.vgg16(pretrained=False)
        self.layer1=vgg.features[:23]
        self.layer2=vgg.features[23:30]
        self.pool=vgg.features[30]
        for param in self.layer1.parameters():
            param.requires_grad=False
        #for param in self.layer2.parameters():
        #    param.requires_grad=False
        self.conv1=nn.Conv2d(512,128,1)
        self.conv2=nn.Conv2d(512,128,1)
        self.conv3=nn.Conv2d(128,128,3,padding=1)
        self.detection1=detection(128,2)
        self.detection2=detection(512,2)
        self.detection3=detection(512,2)


    def forward(self,x):
        x=self.layer1(x)
        x1=self.conv1(x)
        #print(x.shape,x1.shape)
        x2=self.layer2(x)
        x=self.layer2(x)
        x3=self.pool(x)
        x12=functional.relu(self.conv2(x2))
        #print(x12.shape)
        x12=functional.interpolate(x12,scale_factor=2,mode='bilinear',align_corners=True)
        #print(x1.shape,x12.shape)
        x1=x1+x12
        x1=functional.relu(self.conv3(x1))
        out11,out12=self.detection1(x1)
        out21,out22=self.detection2(x2)
        out31,out32=self.detection3(x3)
        return out11,out12,out21,out22,out31,out32

class context(nn.Module):
    def __init__(self,channel):
        super(context,self).__init__()
        channel=int(channel)
        self.conv1=nn.Conv2d(channel,channel//2,3,padding=1)
        self.conv2=nn.Conv2d(channel//2,channel//2,3,padding=1)
        self.conv3=nn.Conv2d(channel//2,channel//2,3,padding=1)
        self.conv4=nn.Conv2d(channel//2,channel//2,3,padding=1)

    def forward(self,x):
        x=functional.relu(self.conv1(x))
        x1=functional.relu(self.conv2(x))
        x2=functional.relu(self.conv3(x))
        x2=functional.relu(self.conv4(x2))
        x=torch.cat((x1,x2),1)
        return x

class detection(nn.Module):
    def __init__(self,channel,k):
        super(detection,self).__init__()
        channel=int(channel)
        k=int(k)
        self.context=context(channel)
        self.conv1=nn.Conv2d(channel,channel,3,padding=1)
        self.conv2=nn.Conv2d(channel*2,4*k,1)
        self.conv3=nn.Conv2d(channel*2,2*k,1)
    
    def forward(self,x):
        x1=functional.relu(self.conv1(x))
        x2=functional.relu(self.context(x))
        x=torch.cat((x1,x2),1)
        out1=self.conv3(x)
        out2=self.conv2(x)
        return out1,out2

def _filter_boxes( boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, :, 2] - boxes[:, :, 0] + 1
    hs = boxes[:, :, 3] - boxes[:, :, 1] + 1
    keep = ((ws >= min_size.view(-1, 1).expand_as(ws)) & (hs >= min_size.view(-1, 1).expand_as(hs)))
    return keep

class proposal_layer(nn.Module):
    def __init__(self,stride,scale,ratios):
        super(proposal_layer,self).__init__()
        self.anchors=torch.from_numpy(generate_anchors(scales=numpy.array(scale),ratios=numpy.array(numpy.array(ratios))))
        self.anchor_num=self.anchors.shape[0]
        self.stride=stride

    def forward(self,score,delta,img):
        scores=score[:,self.anchor_num:,:,:]
        
        cfg_key = 'TRAIN'
        pre_nms_topN = 100
        post_nms_topN = 0
        nms_thresh = 0.3
        min_size = cfg[cfg_key].RPN_MIN_SIZE
        batch_size=delta.shape[0]
        img=numpy.array([tuple(img)*batch_size])
        h,w=scores.shape[2],scores.shape[3]
        shifts=numpy.array([[i,j,i,j] for j in range(h) for i in range(w)])*self.stride
        shifts=torch.from_numpy(shifts).float()
        shifts = shifts.contiguous().type_as(scores).float()
        self.anchors=self.anchors.type_as(scores)

        A = self.anchor_num
        K = shifts.size(0)
        anchor = self.anchors.view(1, A, 4) + shifts.view(shifts.shape[0], 1, 4)
        anchor = anchor.view(1, K * A, 4).expand(batch_size, K * A, 4)

        deltas = delta.permute(0, 2, 3, 1).contiguous()
        deltas = delta.view(batch_size, -1, 4)

        scores = scores.permute(0, 2, 3, 1).contiguous()
        scores = scores.view(batch_size, -1)

        proposals = bbox_transform_inv(anchor, deltas, batch_size)
        proposals = clip_boxes(proposals, img, batch_size)

        keep = _filter_boxes(proposals,torch.tensor(min_size).float().type_as(score))
        proposals = proposals[:, keep[0], :]
        scores = scores[:, keep[0]]

        _, order = torch.sort(scores, 1, True)
        output = scores.new_zeros(batch_size, pre_nms_topN, 5)
        for i in range(batch_size):
            proposals_single = proposals[i]
            scores_single = scores[i]
            order_single = order[i]
            if pre_nms_topN > 0 and pre_nms_topN < scores.numel():
                order_single = order_single[:pre_nms_topN]

            proposals_single = proposals_single[order_single, :]
            scores_single = scores_single[order_single].view(-1,1)

            keep_idx_i = NMS(torch.cat((proposals_single, scores_single), 1), nms_thresh)
            keep_idx_i = keep_idx_i.long().view(-1)
            if post_nms_topN > 0:
                keep_idx_i = keep_idx_i[:post_nms_topN]
            proposals_single = proposals_single[keep_idx_i, :]
            scores_single = scores_single[keep_idx_i, :]

            num_proposal = proposals_single.size(0)
            output[i,:num_proposal, 4] = scores_single[:,0]
            output[i,:num_proposal,0:4] = proposals_single[:,0:4]
        return output

class anchor_target_layer(nn.Module):
    def __init__(self,stride,scale,ratios,allowed_border,name):
        super(anchor_target_layer,self).__init__()
        self.stride=stride
        self.scale=scale
        self.name=name
        self.anchors=torch.from_numpy(generate_anchors(scales=numpy.array(scale),ratios=numpy.array(ratios))).float()
        self.anchor_num=self.anchors.shape[0]
        self.allowed_border=allowed_border

    def forward(self,score,gt_boxes,im_size,score_OHEM):
        h,w=score.shape[2:4]
        batch_size=gt_boxes.shape[0]
        shifts=numpy.array([[i,j,i,j] for j in range(h) for i in range(w)])*self.stride
        #print(h,w)
        shifts=torch.from_numpy(shifts)
        shifts = shifts.contiguous().type_as(score).float()

        A = self.anchor_num
        K = shifts.size(0)
        self.anchors=self.anchors.type_as(gt_boxes)
        anchors = self.anchors.view(1, A, 4) + shifts.view(K, 1, 4)
        anchor = anchors.view(K * A, 4)
        
        total_anchors = int(K * A)

        keep = ((anchor[:, 0] >= -self.allowed_border) &
                (anchor[:, 1] >= -self.allowed_border) &
                (anchor[:, 2] < int(im_size[1]) + self.allowed_border) & # width
                (anchor[:, 3] < int(im_size[0]) + self.allowed_border))  # height
        inds_inside = torch.nonzero(keep).view(-1)
        #print(anchor)
        #print(inds_inside)
        anchors = anchor[inds_inside, :]
        target_size= inds_inside.size(0)

        labels=gt_boxes.new_full((batch_size,target_size),fill_value=-1)
        bbox_inside_weights = gt_boxes.new_full((batch_size, target_size),fill_value=0)
        bbox_outside_weights = gt_boxes.new_full((batch_size, target_size),fill_value=0)
        #print(anchors)
        overlaps = bbox_overlaps_batch(anchors, gt_boxes)
        #print(overlaps)
        max_overlaps, argmax_overlaps =torch.max(overlaps,2)
        gt_max_overlaps, _ = torch.max(overlaps, 1)

        gt_max_overlaps[gt_max_overlaps == 0] = 1e-5
        keep = torch.sum(overlaps.eq(gt_max_overlaps.view(batch_size, 1, -1).expand_as(overlaps)), 2)
        if torch.sum(keep) > 0:
            labels[keep > 0] = 1

        labels[max_overlaps >= cfg.TRAIN.ANCHOR_POSITIVE_OVERLAP] = 1
        labels[max_overlaps < cfg.TRAIN.ANCHOR_NEGATIVE_OVERLAP] = 0

        num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)

        sum_fg = torch.sum((labels == 1).int(), 1)
        sum_bg = torch.sum((labels == 0).int(), 1)

        for i in range(batch_size):
            # subsample positive labels if we have too many
            if sum_fg[i] > num_fg:
                fg_inds = torch.nonzero(labels[i] == 1).view(-1)
                # torch.randperm seems has a bug on multi-gpu setting that cause the segfault.
                # See https://github.com/pytorch/pytorch/issues/1868 for more details.
                # use numpy instead.
                # rand_num = torch.randperm(fg_inds.size(0)).type_as(gt_boxes).long()
                if cfg.TRAIN.HARD_POSITIVE_MINING and score_OHEM is not None:
                    ohem_scores = score_OHEM[i, self.anchor_num:, :, :]
                    # ohem_score (A,H,W) to (H,W,A)
                    ohem_scores = ohem_scores.permute(1, 2, 0).contiguous()
                    # ohem_score (H*W*A)
                    ohem_scores = ohem_scores.view(-1,1)
                    ohem_scores = ohem_scores[inds_inside]
                    # find lowest predicted score
                    pos_ohem_scores = 1 - ohem_scores[fg_inds]
                    #sort by descending order
                    _, orderd_ohem_score = torch.sort(pos_ohem_scores,dim = 0,descending = True)
                    # sample ohem score
                    ohem_sampled_fgs = fg_inds[orderd_ohem_score[:num_fg]]
                    labels[i][fg_inds] = -1
                    labels[i][ohem_sampled_fgs] = 1
                else:
                    rand_num = torch.from_numpy(numpy.random.permutation(fg_inds.size(0))).type_as(gt_boxes).long()
                    disable_inds = fg_inds[rand_num[:fg_inds.size(0) - num_fg]]
                    labels[i][disable_inds] = -1

            #           num_bg = cfg.TRAIN.RPN_BATCHSIZE - sum_fg[i]
            num_bg = cfg.TRAIN.RPN_BATCHSIZE - torch.sum((labels == 1).int(), 1)[i]

            # subsample negative labels if we have too many print(out_array)
            if sum_bg[i] > num_bg:
                bg_inds = torch.nonzero(labels[i] == 0).view(-1)
                # rand_num = torch.randperm(bg_inds.size(0)).type_as(gt_boxes).long()
                if cfg.TRAIN.HARD_NEGATIVE_MINING and score_OHEM is not None:
                    ohem_scores = score_OHEM[i, self.anchor_num:, :, :]
                    # ohem_score (A,H,W) to (H,W,A)
                    ohem_scores = ohem_scores.permute(1, 2, 0).contiguous()
                    # ohem_score (H*W*A)cv2.imwrite('./img.png',img,)
                    ohem_scores = ohem_scores.view(-1,1)
                    ohem_scores = ohem_scores[inds_inside]
                    # find Highest predicted score
                    neg_ohem_scores = ohem_scores[bg_inds]
                    # sort by descending order
                    _, orderd_ohem_score = torch.sort(neg_ohem_scores, dim = 0, descending=True)
                    # sample ohem score
                    ohem_sampled_bgs = bg_inds[orderd_ohem_score[:num_bg]]
                    labels[i][bg_inds] = -1
                    labels[i][ohem_sampled_bgs] = 0
                else:
                    rand_num = torch.from_numpy(numpy.random.permutation(bg_inds.size(0))).type_as(gt_boxes).long()
                    disable_inds = bg_inds[rand_num[:bg_inds.size(0) - num_bg]]
                    labels[i][disable_inds] = -1

        # sum_fg = torch.sum((labels == 1).int(), 1)
        # sum_bg = torch.sum((labels == 0).int(), 1)
        # print("name={}, fg={}, bg={}".format(self._name,sum_fg[0],sum_bg[0]))

        offset = torch.arange(0, batch_size) * gt_boxes.size(1)

        argmax_overlaps = argmax_overlaps + offset.view(batch_size, 1).type_as(argmax_overlaps)
        bbox_targets = _compute_targets_batch(anchors,
                                              gt_boxes.view(-1, 5)[argmax_overlaps.view(-1), :].view(batch_size, -1, 5))

        # use a single value instead of 4 values for easy index.
        bbox_inside_weights[labels == 1] = cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS[0]

        if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
            num_examples = torch.sum(labels[0] >= 0)
            positive_weights = 1.0 / num_examples.item()
            negative_weights = 1.0 / num_examples.item()
        else:
            assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                    (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))

        bbox_outside_weights[labels == 1] = positive_weights
        bbox_outside_weights[labels == 0] = negative_weights

        labels = _unmap(labels, total_anchors, inds_inside, batch_size, fill=-1)

        bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, batch_size, fill=0)
        bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, batch_size, fill=0)
        bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, batch_size, fill=0)

        outputs = []

        labels = labels.view(batch_size, h, w, A).permute(0, 3, 1, 2).contiguous()
        labels = labels.view(batch_size, 1, A * h, w)
        # outputs.append(labels)

        bbox_targets = bbox_targets.view(batch_size, h, w, A * 4).permute(0, 3, 1, 2).contiguous()
        # outputs.append(bbox_targets)

        anchors_count = bbox_inside_weights.size(1)
        bbox_inside_weights = bbox_inside_weights.view(batch_size, anchors_count, 1).expand(batch_size, anchors_count,
                                                                                            4)

        bbox_inside_weights = bbox_inside_weights.contiguous().view(batch_size, h, w, 4 * A) \
            .permute(0, 3, 1, 2).contiguous()

        # outputs.append(bbox_inside_weights)

        bbox_outside_weights = bbox_outside_weights.view(batch_size, anchors_count, 1).expand(batch_size, anchors_count,
                                                                                              4)
        bbox_outside_weights = bbox_outside_weights.contiguous().view(batch_size, h, w, 4 * A) \
            .permute(0, 3, 1, 2).contiguous()
        # outputs.append(bbox_outside_weights)

        # return outputs
        return labels , bbox_targets, bbox_inside_weights, bbox_outside_weights

        

def _unmap(data, count, inds, batch_size, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """

    if data.dim() == 2:
        ret = torch.Tensor(batch_size, count).fill_(fill).type_as(data)
        ret[:, inds] = data
    else:
        ret = torch.Tensor(batch_size, count, data.size(2)).fill_(fill).type_as(data)
        ret[:, inds, :] = data
    return ret


def _compute_targets_batch(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    return bbox_transform_batch(ex_rois, gt_rois[:, :, :4])
def _smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = torch.abs(in_box_diff)
    smoothL1_sign = (abs_in_box_diff < 1. / sigma_2).detach().float()
    in_loss_box = torch.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                  + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = out_loss_box
    for i in sorted(dim, reverse=True):
        loss_box = loss_box.sum(i)
    loss_box = loss_box.mean()
    return loss_box
def _smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = torch.abs(in_box_diff)
    smoothL1_sign = (abs_in_box_diff < 1. / sigma_2).detach().float()
    in_loss_box = torch.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                  + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = out_loss_box
    for i in sorted(dim, reverse=True):
        loss_box = loss_box.sum(i)
    loss_box = loss_box.mean()
    return loss_box

class Rnet(nn.Module):
    def __init__(self):
        super(Rnet,self).__init__()
        vgg=models.vgg16(pretrained=False)
        self.layer=vgg.features
        self.classifer=vgg.classifier[:-1]
        self.fc=nn.Sequential(
            nn.Linear(4096,256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5,inplace=False)
            )
        self.fc1=nn.Linear(256,2)
        self.fc2=nn.Linear(256,4)
        self.fc3=nn.Linear(256,10)

    def forward(self,x):
        x=self.layer(x)
        x=x.view(-1,25088)
        x=self.classifer(x)
        x=self.fc(x)
        x1=functional.sigmoid(self.fc1(x))
        x2=self.fc2(x)
        x3=self.fc3(x)
        return x1,x2,x3


class Onet(nn.Module):
    def __init__(self):
        super(Onet,self).__init__()
        vgg=models.vgg16(pretrained=False)
        self.layer=vgg.features
        self.layer[-1]=nn.MaxPool2d(2,2,ceil_mode=True)
        self.classifier=nn.Sequential(
            nn.Linear(512*3*4,256),
            nn.ReLU(),
            nn.Dropout(0.7)
        )
        self.fc1=nn.Linear(256,2)
        self.fc2=nn.Linear(256,4)

    def forward(self,x):
        x=self.layer(x)
        x=x.view(-1,512*3*4)
        x=self.classifier(x)
        x1=functional.sigmoid(self.fc1(x))
        x2=self.fc2(x)
        return x1,x2

class Rnet_v2(nn.Module):
    def __init__(self):
        super(Rnet_v2,self).__init__()
        n=models.vgg16(pretrained=False)
        self.layer=n.features[:-1]
        self.classifier=nn.Sequential(
            nn.Linear(512*6*7,1024),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.fc1=nn.Linear(1024,2)
        self.fc2=nn.Linear(1024,4)
        self.fc3=nn.Linear(1024,10)

    def forward(self,x):
        x=self.layer(x)
        x=x.view(-1,512*6*7)
        x=self.classifier(x)
        x1=nn.functional.sigmoid(self.fc1(x))
        x2=self.fc2(x)
        x3=nn.functional.sigmoid(self.fc3(x))
        return x1,x2,x3

class Facenet(nn.Module):
    def __init__(self,classnum):
        super(Facenet,self).__init__()
        self.vgg=models.vgg16(pretrained=False).features
        self.vgg[-1]=nn.MaxPool2d(2,2,ceil_mode=True)
        for param in self.vgg.parameters():
            param.requires_grad=False
        self.classifier=nn.Sequential(
            nn.Linear(512*3*4,1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024,128),
        )
        self.fc=nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128,classnum)
        )

    def forward(self,x):
        x=self.vgg(x)
        x=x.view(-1,512*3*4)
        x=self.classifier(x)
        y=self.fc(x)
        return x,y

class resnet50(nn.Module):
    def __init__(self,classnum):
        super(resnet50,self).__init__()
        self.res=models.resnet50(pretrained=False)
        self.res.fc=nn.Sequential(
            nn.Linear(2048,512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512,128)
        )
        self.fc=nn.Sequential(
            nn.ReLU(),
            nn.Linear(128,classnum)
        )
        for param in self.res.parameters():
            param.requires_grad=False
        for param in self.res.fc.parameters():
            param.requires_grad=True

    
    def forward(self,x):
        x=self.res(x)
        x=torch.nn.functional.normalize(x,dim=1)
        y=self.fc(x)
        return x, y

class softmax_layer(nn.Module):
    def __init__(self):
        super(softmax_layer,self).__init__()
        self.fc=nn.Linear(128,2)

    def forward(self,x):
        x=torch.abs(x[:,0]-x[:,1])
        x=self.fc(x)
        return x

class Facenet_test(nn.Module):
    def __init__(self):
        super(Facenet_test,self).__init__()
        self.layer=models.GoogLeNet(128)

    def forward(self,x):
        x=self.layer(x)
        x=x/(torch.norm(x,dim=1).unsqueeze(1))
        return x

class Facenet_train(nn.Module):
    def __init__(self):
        super(Facenet_train,self).__init__()
        self.layer=models.GoogLeNet(128)

    def forward(self,x):
        x,_,_=self.layer(x)
        x=x/(torch.norm(x,dim=1).unsqueeze(1))
        return x

class Facenet_v2(nn.Module):
    def __init__(self):
        super(Facenet_v2,self).__init__()
        resnet=models.resnet50()
        resnet.fc=nn.Linear(2048,128)
        self.layer=resnet

    def forward(self,x):
        x=self.layer(x)
        x=x/torch.norm(x)
        return x