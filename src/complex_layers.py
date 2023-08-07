"""
    Based on caffe/examples/faster-rcnn/lib/rpn/proposal_layer.py from https://github.com/intel/caffe
    with copyright header below, where LICENSE refers to caffe/examples/faster-rcnn/license.txt in the above repo:
        Copyright (c) 2015 Microsoft
        Licensed under The MIT License [see LICENSE for details]
        Written by Ross Girshick and Sean Bell
"""

import torch
from torchvision.ops import nms


def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)

    return anchors


def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)

    return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    ws = ws.unsqueeze(1)
    hs = hs.unsqueeze(1)
    anchors = torch.hstack((x_ctr - 0.5 * (ws - 1),
                            y_ctr - 0.5 * (hs - 1),
                            x_ctr + 0.5 * (ws - 1),
                            y_ctr + 0.5 * (hs - 1)))

    return anchors


def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = torch.round(torch.sqrt(size_ratios))
    hs = torch.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)

    return anchors


def generate_anchors(base_size=16, ratios=None, scales=2**torch.arange(3, 6)):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """

    if ratios is None:
        ratios = torch.tensor([0.5, 1, 2])
    base_anchor = torch.tensor([1, 1, base_size, base_size]) - 1
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    anchors = torch.vstack([_scale_enum(ratio_anchors[i, :], scales)
                            for i in range(ratio_anchors.size(0))])

    return anchors


def bbox_transform_inv(boxes, deltas):
    if boxes.size(0) == 0:
        return torch.zeros((0, deltas.size(1))).to(deltas)

    boxes = boxes.to(deltas, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * widths.unsqueeze(1) + ctr_x.unsqueeze(1)
    pred_ctr_y = dy * heights.unsqueeze(1) + ctr_y.unsqueeze(1)
    pred_w = torch.exp(dw) * widths.unsqueeze(1)
    pred_h = torch.exp(dh) * heights.unsqueeze(1)

    pred_boxes = torch.zeros(deltas.size()).to(deltas)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes


def clip_boxes(boxes, im_shape, torch_zero):
    """
    Clip boxes to image boundaries.
    """

    # x1 >= 0
    boxes[:, 0::4] = torch.maximum(torch.minimum(boxes[:, 0::4], im_shape[1] - 1), torch_zero)
    # y1 >= 0
    boxes[:, 1::4] = torch.maximum(torch.minimum(boxes[:, 1::4], im_shape[0] - 1), torch_zero)
    # x2 < im_shape[1]
    boxes[:, 2::4] = torch.maximum(torch.minimum(boxes[:, 2::4], im_shape[1] - 1), torch_zero)
    # y2 < im_shape[0]
    boxes[:, 3::4] = torch.maximum(torch.minimum(boxes[:, 3::4], im_shape[0] - 1), torch_zero)

    return boxes


def _filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = torch.where((ws >= min_size) & (hs >= min_size))[0]

    return keep


class ProposalLayerModule(torch.nn.Module):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def __init__(self, param):
        super(ProposalLayerModule, self).__init__()

        self.device = param["device"]
        self.torch_zero = torch.tensor([0], device=self.device)

        self._feat_stride = param['feat_stride']
        anchor_scales = param.get('scales', (8, 16, 32))
        self._anchors = generate_anchors(scales=torch.tensor(anchor_scales))
        self._num_anchors = self._anchors.size(0)

        self.pre_nms_topN = param['pre_nms_topN']
        self.post_nms_topN = param['post_nms_topN']
        self.nms_thresh = param['nms_thresh']
        self.min_size = param['min_size']

    @torch.no_grad()
    def forward(self, scores, bbox_deltas, im_info):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate A anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the A anchors
        # clip predicted boxes to image
        # remove predicted boxes with either height or width < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)

        assert scores.data.size(0) == 1, 'Only single item batches are supported'

        # the first set of _num_anchors channels are bg probs; the second set are the fg probs, which we want
        scores = scores[:, self._num_anchors:, :, :]  # scores.data?

        # 1. Generate proposals from bbox deltas and shifted anchors
        height, width = scores.size()[-2:]

        # Enumerate all shifts
        # shift_x = np.arange(0, width) * self._feat_stride
        shift_x = torch.arange(0, width, device=self.device, requires_grad=False) * self._feat_stride
        # shift_y = np.arange(0, height) * self._feat_stride
        shift_y = torch.arange(0, height, device=self.device, requires_grad=False) * self._feat_stride
        # shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shift_x, shift_y = torch.meshgrid(shift_x, shift_y)
        # shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
        #                     shift_x.ravel(), shift_y.ravel())).transpose()
        shifts = torch.vstack((shift_x.flatten(), shift_y.flatten(),
                               shift_x.flatten(), shift_y.flatten())).transpose(0, 1)

        # Enumerate all shifted anchors:
        #
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = self._num_anchors
        K = shifts.size(0)
        anchors = torch.clone(self._anchors).to(device=self.device)
        anchors = anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).permute((1, 0, 2))
        anchors = anchors.reshape((K * A, 4))

        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:
        #
        # bbox deltas will be (1, 4 * A, H, W) format
        # transpose to (1, H, W, 4 * A)
        # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
        # in slowest to fastest order
        bbox_deltas = bbox_deltas.permute((0, 2, 3, 1)).reshape((-1, 4))

        # Same story for the scores:
        #
        # scores are (1, A, H, W) format
        # transpose to (1, H, W, A)
        # reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)
        scores = scores.permute((0, 2, 3, 1)).reshape((-1, 1))

        # Convert anchors into proposals via bbox transformations
        proposals = bbox_transform_inv(anchors, bbox_deltas)

        # 2. clip predicted boxes to image
        proposals = clip_boxes(proposals, im_info[:2], self.torch_zero)

        # 3. remove predicted boxes with either height or width < threshold
        # (NOTE: convert min_size to input image scale stored in im_info[2])
        keep = _filter_boxes(proposals, self.min_size * im_info[2])
        proposals = proposals[keep, :]
        scores = scores[keep]

        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topN (e.g. 6000)
        order = scores.flatten().argsort(descending=True)
        if self.pre_nms_topN > 0:
            order = order[:self.pre_nms_topN]
        proposals = proposals[order, :]
        scores = scores[order]

        # 6. apply nms (e.g. threshold = 0.7)
        # 7. take after_nms_topN (e.g. 300)
        # 8. return the top proposals (-> RoIs top)
        # keep = nms(np.hstack((proposals, scores)), nms_thresh)
        # nms is based on classless boxes - batched_nms needs class predictions which happens later in the network
        keep = nms(boxes=proposals, scores=scores.squeeze(), iou_threshold=self.nms_thresh)
        if self.post_nms_topN > 0:
            keep = keep[:self.post_nms_topN]
        proposals = proposals[keep, :]

        # Output rois blob
        # If the RPN implementation only supports a single input image, then all batch inds are 0
        batch_inds = torch.zeros((proposals.size(0), 1), dtype=torch.float32, device=self.device)
        out = torch.hstack((batch_inds, proposals.to(torch.float32, copy=False)))

        return out

    def backward(self, grad_outputs):
        """This layer does not propagate gradients."""
        pass
