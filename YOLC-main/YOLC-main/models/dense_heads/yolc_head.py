# Copyright (c) OpenMMLab. All rights reserved.
from operator import gt
import re
import cv2
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import bias_init_with_prob, normal_init
from mmcv.ops import batched_nms, soft_nms, DeformConv2d
from mmcv.runner import force_fp32
from mmcv.ops import nms
from mmdet.core import bbox, multi_apply, MlvlPointGenerator
from mmdet.models import HEADS, build_loss
from mmdet.models.utils import gaussian_radius, gen_gaussian_target
from mmdet.models.utils.gaussian_target import (get_local_maximum, transpose_and_gather_feat)
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
from mmdet.models.dense_heads.dense_test_mixins import BBoxTestMixin
import torch
import torch.nn.functional as F
from kornia.filters import gaussian_blur2d


@HEADS.register_module()
class YOLCHead(BaseDenseHead, BBoxTestMixin):
    """YOLC: You Only Look Clusters for Tiny Object Detection in Aerial Images.
    Paper link <https://arxiv.org/abs/2404.06180>

    Args:
        in_channel (int): Number of channel in the input feature map.
        feat_channel (int): Number of channel in the intermediate feature map.
        num_classes (int): Number of categories excluding the background
            category.
        loss_center_heatmap (dict | None): Config of center heatmap loss.
            Default: GaussianFocalLoss.
        loss_xywh (dict | None): Config of xywh loss. Default: GWDLoss.
        train_cfg (dict | None): Training config. Useless in CenterNet,
            but we keep this variable for SingleStageDetector. Default: None.
        test_cfg (dict | None): Testing config of CenterNet. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channel,
                 feat_channel,
                 num_classes,
                 loss_center_local=dict(
                     type='GaussianFocalLoss', loss_weight=1.0),
                 loss_xywh=dict(type='GWDLoss', loss_weight=2.0),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(YOLCHead, self).__init__(init_cfg)
        self.num_classes = num_classes
        self.local_head = self._build_loc_head(in_channel, num_classes)
        self._build_reg_head(in_channel, feat_channel)

        self.loss_center_local = build_loss(loss_center_local)
        self.loss_xywh_coarse = build_loss(loss_xywh)
        self.loss_xywh_refine = build_loss(loss_xywh)
        loss_l1 = dict(type='L1Loss', loss_weight=0.5)
        self.loss_xywh_coarse_l1 = build_loss(loss_l1)
        self.loss_xywh_refine_l1 = build_loss(loss_l1)

        # 修改 strides 为二维值的列表
        strides = [(32, 32)]  # 使用二维 stride
        self.prior_generator = MlvlPointGenerator(strides, offset=0)

        dcn_base = np.arange(-1, 2).astype(np.float64)
        dcn_base_y = np.repeat(dcn_base, 3)
        dcn_base_x = np.tile(dcn_base, 3)
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape(
            (-1))
        self.dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False

    def _build_loc_head(self, in_channel, out_channel):
        """Build head for high resolution heatmap branch."""
        last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=in_channel,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.BatchNorm2d(in_channel, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channel, self.num_classes * 8, 4, stride=2, padding=1, output_padding=0),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.num_classes * 8, self.num_classes * 8, 4, stride=2, padding=1, output_padding=0),
            nn.Conv2d(self.num_classes * 8, out_channel, kernel_size=1, groups=self.num_classes) # Group Conv
        )
        return last_layer

    def _build_reg_head(self, in_channel, feat_channel):
        """Build head for regression branch."""
        self.reg_conv = nn.Sequential(
            nn.Conv2d(in_channel, feat_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channel, feat_channel, kernel_size=3, padding=1))
        # branch in xywh_init and bbox_offset
        self.xywh_init = nn.Conv2d(feat_channel, 4, kernel_size=1)
        self.bbox_offset = nn.Conv2d(feat_channel, 18, kernel_size=1)
        self.xywh_refine = DeformConv2d(feat_channel, 4, kernel_size=3, padding=1)


    def _build_head(self, in_channel, feat_channel, out_channel):
        """Build head for each branch."""
        layer = nn.Sequential(
            nn.Conv2d(in_channel, feat_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channel, out_channel, kernel_size=1))
        return layer

    def init_weights(self):
        """Initialize weights of the head."""
        for head in [self.local_head, self.reg_conv, self.xywh_init, self.bbox_offset, self.xywh_refine]:
            for m in head.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.ConvTranspose2d):
                    m.weight.data.normal_(0.0, 0.02)

    def get_points(self, featmap_sizes, img_metas, device):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.

        Returns:
            tuple: points of each image, valid flags of each image
        """
        num_imgs = len(img_metas)

        # since feature map sizes of all images are the same, we only compute
        # points center for one time
        multi_level_points = self.prior_generator.grid_priors(
            featmap_sizes, device=device, with_stride=False)
        points_list = [multi_level_points[0].clone() for _ in range(num_imgs)]

        return points_list

    def forward(self, feats):
        """Forward features. Notice CenterNet head does not use FPN.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            center_local_preds (List[Tensor]): center predict heatmaps for
                all levels, the channels number is num_classes.
            wh_preds (List[Tensor]): wh predicts for all levels, the channels
                number is 2.
            offset_preds (List[Tensor]): offset predicts for all levels, the
               channels number is 2.
        """
        center_local_preds, xywh_preds_coarse, xywh_preds_refine = multi_apply(self.forward_single, feats)
    
        # Debugging output
        # print(f"center_local_preds length: {len(center_local_preds)}")
        # print(f"xywh_preds_coarse length: {len(xywh_preds_coarse)}")
        # print(f"xywh_preds_refine length: {len(xywh_preds_refine)}")
        return multi_apply(self.forward_single, feats)

    def forward_single(self, feat):
        """Forward feature of a single level."""
        # print("\nDebug forward_single:")
        # print(f"Input feat: shape={feat.shape}, range=[{feat.min():.4f}, {feat.max():.4f}]")
        
        # 中心点预测
        center_local_pred = self.local_head(feat)
        # print(f"Before sigmoid - center_local_pred: shape={center_local_pred.shape}, "
        #     f"range=[{center_local_pred.min():.4f}, {center_local_pred.max():.4f}]")
        center_local_pred = center_local_pred.sigmoid()
        # print(f"After sigmoid - center_local_pred: shape={center_local_pred.shape}, "
        #     f"range=[{center_local_pred.min():.4f}, {center_local_pred.max():.4f}]")
        
        # 回归特征
        reg_feat = self.reg_conv(feat)
        # print(f"reg_feat: shape={reg_feat.shape}, range=[{reg_feat.min():.4f}, {reg_feat.max():.4f}]")
        
        # 初始预测
        xywh_pred_coarse = self.xywh_init(reg_feat)
        # print(f"xywh_pred_coarse: shape={xywh_pred_coarse.shape}, "
        #     f"range=[{xywh_pred_coarse.min():.4f}, {xywh_pred_coarse.max():.4f}]")
        
        # 获取特征图尺寸和设备
        featmap_sizes = [xywh_pred_coarse.size()[-2:]]
        device = xywh_pred_coarse.device
        
        try:
            # 生成中心点
            center_points = self.prior_generator.grid_priors(
                featmap_sizes, device=device, with_stride=False)[0]
            # print(f"center_points: shape={center_points.shape}, "
            #     f"range=[{center_points.min():.4f}, {center_points.max():.4f}]")
            
            # 处理边界框预测
            bbox_pred = xywh_pred_coarse.detach().permute(0, 2, 3, 1).reshape(
                xywh_pred_coarse.size(0), -1, 4).contiguous()
            # print(f"bbox_pred before offset: shape={bbox_pred.shape}, "
            #     f"range=[{bbox_pred.min():.4f}, {bbox_pred.max():.4f}]")
            
            # 添加中心点偏移
            bbox_pred[:, :, :2] = bbox_pred[:, :, :2] + center_points.unsqueeze(0)
            # print(f"bbox_pred after offset: shape={bbox_pred.shape}, "
            #     f"range=[{bbox_pred.min():.4f}, {bbox_pred.max():.4f}]")
            
            # 生成偏移
            offset = self.bbox_offset(reg_feat).sigmoid()
            # print(f"offset: shape={offset.shape}, range=[{offset.min():.4f}, {offset.max():.4f}]")
            
            # 生成可变形卷积偏移
            dcn_offset = self.gen_dcn_offset(
                bbox_pred.permute(0, 2, 1), offset, center_points)
            # print(f"dcn_offset: shape={dcn_offset.shape}, "
            #     f"range=[{dcn_offset.min():.4f}, {dcn_offset.max():.4f}]")
            
            # 细化预测
            xywh_pred_refine = self.xywh_refine(reg_feat, dcn_offset)
            # print(f"xywh_pred_refine: shape={xywh_pred_refine.shape}, "
            #     f"range=[{xywh_pred_refine.min():.4f}, {xywh_pred_refine.max():.4f}]")
            
        except Exception as e:
            # print(f"Error in forward_single: {str(e)}")
            xywh_pred_refine = torch.zeros_like(xywh_pred_coarse)
        
        # 检查网络权重
        # print("\nNetwork weights:")
        # 检查 local_head 中的每个层
        # for name, module in self.local_head.named_modules():
        #     if hasattr(module, 'weight'):
        #         print(f"local_head.{name} weight: range=[{module.weight.min():.4f}, {module.weight.max():.4f}]")
        
        # # 检查其他组件
        # for name, module in [('reg_conv', self.reg_conv), 
        #                     ('xywh_init', self.xywh_init),
        #                     ('bbox_offset', self.bbox_offset)]:
        #     if hasattr(module, 'weight'):
        #         print(f"{name} weight: range=[{module.weight.min():.4f}, {module.weight.max():.4f}]")
        
        return center_local_pred, xywh_pred_coarse, xywh_pred_refine
    
    def gen_dcn_offset(self, bbox_pred, offset, center_points):
        '''
            bbox_pred: [B, H, W, 4] [x, y, w/2, h/2] detach
            offset : [B, ?, H, W] require_grad is True
            center_points : [HxW, 2], cordinate of anchor points
            
            Return:
                dcn_offset: [B, ?x2, H, W]
        '''
        B, _, H, W = offset.shape
        dcn_offset = offset.new(B, 9*2, H, W)
        bbox_pred = bbox_pred.view(B, 4, H, W)
        bbox_pred[:, 0:2, :, :,] = bbox_pred[:, 0:2, :, :,] - bbox_pred[:, 2:4, :, :,]
        bbox_pred[:, 2:4, :, :,] = 2 * bbox_pred[:, 2:4, :, :,]
        
        dcn_offset[:, 0::2, :, :] = bbox_pred[:, 0, :, :].unsqueeze(1) + bbox_pred[:, 2, :, :].unsqueeze(1) * offset[:, 0::2, :, :]
        dcn_offset[:, 1::2, :, :] = bbox_pred[:, 1, :, :].unsqueeze(1) + bbox_pred[:, 3, :, :].unsqueeze(1) * offset[:, 1::2, :, :]

        dcn_base_offset = self.dcn_base_offset.type_as(dcn_offset)
        dcn_anchor_offset = center_points.view(H, W, 2).repeat(B, 1, 1, 1).repeat(1, 1, 1, 9).permute(0, 3, 1, 2)
        dcn_anchor_offset += dcn_base_offset
        return dcn_offset - dcn_anchor_offset


    @force_fp32(apply_to=('center_local_preds', 'xywh_preds_coarse', 'xywh_preds_refine'))
    def loss(self,
         center_local_preds,
         xywh_preds_coarse,
         xywh_preds_refine,
         gt_bboxes,
         gt_labels,
         img_metas,
         gt_bboxes_ignore=None):
        """Compute losses of the head."""

        device = center_local_preds[0].device
        
        # 初始化损失列表
        center_losses = []
        coarse_losses = []
        coarse_l1_losses = []
        refine_losses = []
        refine_l1_losses = []
        
        for i in range(len(center_local_preds)):
            try:
                center_local_pred = center_local_preds[i]
                xywh_pred_coarse = xywh_preds_coarse[i]
                xywh_pred_refine = xywh_preds_refine[i]
                
                # 获取目标
                target_result, avg_factor = self.get_targets(gt_bboxes, 
                                                        gt_labels,
                                                        center_local_pred.shape,
                                                        img_metas[0]['pad_shape'])
                
                # 确保 avg_factor 有效
                avg_factor = max(float(avg_factor), 1.0)
                
                # 计算中心点热图损失
                center_loss = self.loss_center_local(
                    center_local_pred,
                    target_result['center_heatmap_target'].to(device),
                    avg_factor=avg_factor)
                
                # 处理预测值
                featmap_size = xywh_pred_coarse.shape[-2:]
                B = xywh_pred_coarse.size(0)
                
                # 重新调整目标大小以匹配预测
                xywh_target = target_result['xywh_target'].to(device)
                if xywh_target.shape[-2:] != featmap_size:
                    xywh_target = F.interpolate(
                        xywh_target.permute(0, 3, 1, 2),
                        size=featmap_size,
                        mode='bilinear',
                        align_corners=True)
                    xywh_target = xywh_target.permute(0, 2, 3, 1)
                
                # 调整预测和目标的形状
                bbox_pred_coarse = xywh_pred_coarse.permute(0, 2, 3, 1).reshape(B, -1, 4)
                bbox_pred_refine = xywh_pred_refine.permute(0, 2, 3, 1).reshape(B, -1, 4)
                xywh_target = xywh_target.reshape(B, -1, 4)
                
                # 确保权重形状匹配
                xywh_target_weight = target_result['xywh_target_weight'].to(device)
                if xywh_target_weight.shape[-2:] != featmap_size:
                    xywh_target_weight = F.interpolate(
                        xywh_target_weight.unsqueeze(1),
                        size=featmap_size,
                        mode='nearest').squeeze(1)
                xywh_target_weight = xywh_target_weight.reshape(B, -1)
                
                # 计算回归损失
                coarse_loss = self.loss_xywh_coarse(
                    bbox_pred_coarse,
                    xywh_target,
                    xywh_target_weight,
                    avg_factor=avg_factor)
                
                refine_loss = self.loss_xywh_refine(
                    bbox_pred_refine,
                    xywh_target,
                    xywh_target_weight,
                    avg_factor=avg_factor)
                
                # 确保所有损失都需要梯度
                center_loss.requires_grad_(True)
                coarse_loss.requires_grad_(True)
                refine_loss.requires_grad_(True)
                
                # 添加损失
                center_losses.append(center_loss)
                coarse_losses.append(coarse_loss)
                refine_losses.append(refine_loss)
                
            except Exception as e:
                print(f"Error at level {i}: {str(e)}")
                continue
        
        # 合并损失并确保有梯度
        losses = {}
        if center_losses:
            losses['loss_center_heatmap'] = torch.stack(center_losses).mean()
        if coarse_losses:
            losses['loss_xywh_coarse'] = torch.stack(coarse_losses).mean()
        if refine_losses:
            losses['loss_xywh_refine'] = torch.stack(refine_losses).mean()
        
        # 如果没有有效损失，返回带梯度的默认值
        if not losses:
            default_tensor = torch.tensor(0.1, device=device, requires_grad=True)
            losses = {
                'loss_center_heatmap': default_tensor.clone(),
                'loss_xywh_coarse': default_tensor.clone(),
                'loss_xywh_refine': default_tensor.clone()
            }
        
        # 计算总损失
        losses['loss'] = sum(loss for loss in losses.values())
        
        return losses
    def get_targets(self, gt_bboxes, gt_labels, feat_shape, img_shape):
        """Compute regression and classification targets in multiple images.

        Args:
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box.
            feat_shape (list[int]): feature map shape with value [B, _, H, W]
            img_shape (list[int]): image shape in [h, w] format.

        Returns:
            tuple[dict,float]: The float value is mean avg_factor, the dict has
            components below:
            - center_heatmap_target (Tensor): targets of center heatmap, \
                shape (B, num_classes, H, W).
            - xywh_target (Tensor): targets of xywh predict, shape \
                (B, 4, H, W).
            - xywh_target_weight (Tensor): weights of wh and offset \
                predict, shape (B, 4, H, W).
        """
        img_h, img_w = img_shape[:2]
        bs, _, feat_h, feat_w = feat_shape

        width_ratio = float(feat_w / img_w)
        height_ratio = float(feat_h / img_h)

        # 修改 center_heatmap_target 的尺寸，使其与 feat_shape 匹配
        center_heatmap_target = gt_bboxes[0].new_zeros(
            [bs, self.num_classes, feat_h, feat_w])
        xywh_target = gt_bboxes[0].new_zeros([bs, feat_h, feat_w, 4])
        xywh_target_weight = gt_bboxes[0].new_zeros(
            [bs, feat_h, feat_w])
        xywh_l1target_weight = gt_bboxes[0].new_zeros(
            [bs, feat_h, feat_w, 4])

        for batch_id in range(bs):
            gt_bbox = gt_bboxes[batch_id]
            gt_label = gt_labels[batch_id]

            if gt_bbox.size(0) == 0:  # 如果没有 ground truth boxes
                continue

            center_x = (gt_bbox[:, [0]] + gt_bbox[:, [2]]) * width_ratio / 2
            center_y = (gt_bbox[:, [1]] + gt_bbox[:, [3]]) * height_ratio / 2
            gt_centers = torch.cat((center_x, center_y), dim=1)

            for j, ct in enumerate(gt_centers):
                ctx_int, cty_int = ct.int()
                ctx, cty = ct
                scale_box_h = (gt_bbox[j][3] - gt_bbox[j][1]) * height_ratio
                scale_box_w = (gt_bbox[j][2] - gt_bbox[j][0]) * width_ratio
                box_h = (gt_bbox[j][3] - gt_bbox[j][1])
                box_w = (gt_bbox[j][2] - gt_bbox[j][0])
                radius = gaussian_radius([box_h, box_w],
                                        min_overlap=0.3)
                radius = max(0, int(radius))
                ind = gt_label[j]

                # 确保坐标在特征图范围内
                ctx_int = torch.clamp(ctx_int, 0, feat_w - 1)
                cty_int = torch.clamp(cty_int, 0, feat_h - 1)

                # 生成高斯目标
                gen_gaussian_target(center_heatmap_target[batch_id, ind],
                                [ctx_int, cty_int], radius)

                if cty_int >= feat_h or ctx_int >= feat_w:
                    continue

                xywh_target[batch_id, cty_int, ctx_int, 0] = ctx
                xywh_target[batch_id, cty_int, ctx_int, 1] = cty
                xywh_target[batch_id, cty_int, ctx_int, 2] = scale_box_w/2
                xywh_target[batch_id, cty_int, ctx_int, 3] = scale_box_h/2

                xywh_target_weight[batch_id, cty_int, ctx_int] = 1
                xywh_l1target_weight[batch_id, cty_int, ctx_int, 0:2] = 1.0
                xywh_l1target_weight[batch_id, cty_int, ctx_int, 2:4] = 0.2

        avg_factor = max(1, center_heatmap_target.eq(1).sum())
        target_result = dict(
            center_heatmap_target=center_heatmap_target,
            xywh_target=xywh_target,
            xywh_target_weight=xywh_target_weight,
            xywh_l1target_weight=xywh_l1target_weight)
        return target_result, avg_factor

    def get_bboxes(self,
               center_heatmap_preds,
               xywh_preds,
               img_metas,
               rescale=False,
               with_nms=True):
        """Transform network output for a batch into bbox predictions.
        
        Returns:
            tuple[list[Tensor], list[Tensor]]: 每张图片的边界框和标签。
                - det_bboxes: list[Tensor], shape (n, 5), 最后一列是分数
                - det_labels: list[Tensor], shape (n,), 类别索引
        """
        # 处理输入是列表的情况
        if isinstance(center_heatmap_preds, list):
            center_heatmap_preds = center_heatmap_preds[-1]  # 使用最后一个预测
        if isinstance(xywh_preds, list):
            xywh_preds = xywh_preds[-1]  # 使用最后一个预测
            
        batch_size = center_heatmap_preds.shape[0]
        batch_det_bboxes = []
        batch_det_labels = []
        
        for img_id in range(batch_size):
            try:
                # 获取单张图片的预测结果
                det_bboxes, det_labels = self._get_bboxes_single(
                    center_heatmap_pred=center_heatmap_preds[img_id:img_id + 1],
                    xywh_pred=xywh_preds[img_id:img_id + 1],
                    img_meta=img_metas[img_id],
                    rescale=rescale,
                    with_nms=with_nms
                )
                
                # 确保结果是正确的类型和形状
                if not isinstance(det_bboxes, torch.Tensor):
                    det_bboxes = torch.from_numpy(det_bboxes).to(center_heatmap_preds.device)
                if not isinstance(det_labels, torch.Tensor):
                    det_labels = torch.from_numpy(det_labels).to(center_heatmap_preds.device)
                
                # 确保 det_bboxes 的形状是 (n, 5)
                if len(det_bboxes.shape) == 1:
                    det_bboxes = det_bboxes.unsqueeze(0)
                if det_bboxes.shape[1] == 4:
                    # 如果没有分数，添加一列零分数
                    scores = torch.zeros(det_bboxes.shape[0], 1, device=det_bboxes.device)
                    det_bboxes = torch.cat([det_bboxes, scores], dim=1)
                
                # 确保 det_labels 的形状是 (n,)
                if len(det_labels.shape) > 1:
                    det_labels = det_labels.squeeze()
                
                # 添加到批次结果中
                batch_det_bboxes.append(det_bboxes)
                batch_det_labels.append(det_labels)
                
            except Exception as e:
                print(f"Error processing image {img_id}: {str(e)}")
                # 发生错误时，添加空结果
                batch_det_bboxes.append(
                    torch.zeros((0, 5), device=center_heatmap_preds.device))
                batch_det_labels.append(
                    torch.zeros((0,), device=center_heatmap_preds.device, dtype=torch.long))
        
        # 打印调试信息
        for i, (bboxes, labels) in enumerate(zip(batch_det_bboxes, batch_det_labels)):
            print(f"Image {i}:")
            print(f"  bboxes shape: {bboxes.shape}, dtype: {bboxes.dtype}")
            print(f"  labels shape: {labels.shape}, dtype: {labels.dtype}")
        
        return batch_det_bboxes, batch_det_labels

    def _get_bboxes_single(self,
                       center_heatmap_pred,
                       xywh_pred,
                       img_meta,
                       rescale=False,
                       with_nms=True):
        """Transform outputs of a single image into bbox predictions."""
        # 打印输入的形状和值范围
        print("\nDebug _get_bboxes_single:")
        print(f"center_heatmap_pred: shape={center_heatmap_pred.shape}, "
            f"range=[{center_heatmap_pred.min():.4f}, {center_heatmap_pred.max():.4f}]")
        print(f"xywh_pred: shape={xywh_pred.shape}, "
            f"range=[{xywh_pred.min():.4f}, {xywh_pred.max():.4f}]")
        
        # 应用 sigmoid 到 xywh 预测
        xywh_pred = xywh_pred.sigmoid()
        print(f"After sigmoid - xywh_pred range: [{xywh_pred.min():.4f}, {xywh_pred.max():.4f}]")
        
        # 获取先验点
        points = self.prior_generator.grid_priors(
            [xywh_pred.shape[-2:]], 
            device=xywh_pred.device, 
            with_stride=False)[0]
        print(f"points: shape={points.shape}")
        
        try:
            # 解码热图和边界框预测
            batch_bboxes, batch_topk_labels = self.decode_heatmap(
                center_heatmap_pred,
                xywh_pred,
                img_meta['img_shape'],
                k=self.test_cfg.get('topk', 100)  # 只保留 k 参数
            )
            
            print(f"After decode - batch_bboxes: shape={batch_bboxes.shape}, "
                f"range=[{batch_bboxes[..., :4].min():.4f}, {batch_bboxes[..., :4].max():.4f}]")
            print(f"batch_topk_labels: shape={batch_topk_labels.shape}, "
                f"unique values={torch.unique(batch_topk_labels).tolist()}")
            
            # 如果需要，进行坐标缩放
            if rescale:
                scale_factor = batch_bboxes.new_tensor(img_meta['scale_factor'])
                batch_bboxes[..., :4] /= scale_factor.unsqueeze(0)
                print(f"After rescale - bbox range: [{batch_bboxes[..., :4].min():.4f}, "
                    f"{batch_bboxes[..., :4].max():.4f}]")
            
            # 应用 NMS
            if with_nms:
                det_bboxes, det_labels = self._bboxes_nms(
                    batch_bboxes,
                    batch_topk_labels,
                    self.test_cfg
                )
                print(f"After NMS - det_bboxes: shape={det_bboxes.shape}, "
                    f"score range=[{det_bboxes[:, -1].min():.4f} if len(det_bboxes) > 0 else 0, "
                    f"{det_bboxes[:, -1].max():.4f} if len(det_bboxes) > 0 else 0]")
                print(f"det_labels: shape={det_labels.shape}, "
                    f"unique values={torch.unique(det_labels).tolist()}")
            else:
                det_bboxes, det_labels = batch_bboxes, batch_topk_labels
            
        except Exception as e:
            print(f"Error in detection: {str(e)}")
            # 返回空的检测结果
            det_bboxes = center_heatmap_pred.new_zeros((0, 5))
            det_labels = center_heatmap_pred.new_zeros((0,), dtype=torch.long)
        
        # 确保返回的结果不为空且格式正确
        if det_bboxes.shape[0] == 0:
            print("Warning: No detections found!")
            # 返回一个空的检测结果
            det_bboxes = center_heatmap_pred.new_zeros((0, 5))
            det_labels = center_heatmap_pred.new_zeros((0,), dtype=torch.long)
        
        # 确保结果是 numpy 数组
        det_bboxes = det_bboxes.detach().cpu().numpy()
        det_labels = det_labels.detach().cpu().numpy()
        
        return det_bboxes, det_labels

    def decode_heatmap(self,
                       center_heatmap_pred,
                       xywh_pred,
                       img_shape,
                       k=100,
                       kernel=3):
        
        # 确保 img_shape 只包含高度和宽度
        if isinstance(img_shape, (list, tuple)) and len(img_shape) > 2:
            inp_h, inp_w = img_shape[:2]
        else:
            inp_h, inp_w = img_shape

        # center_heatmap_pred [bs, 1, H, W]
        height, width = xywh_pred.shape[2:]

        center_heatmap_pred = get_local_maximum(
            center_heatmap_pred, kernel=kernel)

        *batch_dets, topk_ys, topk_xs = self.get_topk_from_heatmap(
            center_heatmap_pred, k=k)
        batch_scores, batch_index, batch_topk_labels = batch_dets
        
        xywh = transpose_and_gather_feat(xywh_pred, batch_index)

        topk_xs = topk_xs + xywh[..., 0]
        topk_ys = topk_ys + xywh[..., 1]
        tl_x = (topk_xs - xywh[..., 2]) * (inp_w / width)
        tl_y = (topk_ys - xywh[..., 3]) * (inp_h / height)
        br_x = (topk_xs + xywh[..., 2]) * (inp_w / width)
        br_y = (topk_ys + xywh[..., 3]) * (inp_h / height)

        batch_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], dim=2)
        batch_bboxes = torch.cat((batch_bboxes, batch_scores[..., None]),
                                 dim=-1)
        return batch_bboxes, batch_topk_labels
    def _decode_heatmap(self,
                        center_heatmap_pred,
                        xywh_pred,
                        k=100,
                        kernel=3,
                        score_thr=0.1):
        """从热图中解码出边界框。"""
        batch, _, height, width = center_heatmap_pred.size()
        
        # 打印调试信息
        print(f"Heatmap max value: {center_heatmap_pred.max()}")
        print(f"Heatmap min value: {center_heatmap_pred.min()}")
        print(f"XYWH pred shape: {xywh_pred.shape}")
        
        # 获取热图的局部最大值
        heatmap = self._local_maximum(center_heatmap_pred, kernel)
        
        # 打印局部最大值的统计信息
        print(f"Number of local maxima: {(heatmap > score_thr).sum().item()}")
    def get_topk_from_heatmap(self, center_heatmap, k=20):
        """Get top k positions from heatmap.

        Args:
            scores (Tensor): Target heatmap with shape
                [batch, num_classes, height, width].
            k (int): Target number. Default: 20.

        Returns:
            tuple[torch.Tensor]: Scores, indexes, categories and coords of
                topk keypoint. Containing following Tensors:

            - topk_scores (Tensor): Max scores of each topk keypoint.
            - topk_inds (Tensor): Indexes of each topk keypoint.
            - topk_clses (Tensor): Categories of each topk keypoint.
            - topk_ys (Tensor): Y-coord of each topk keypoint.
            - topk_xs (Tensor): X-coord of each topk keypoint.
        """
        # center_heatmap values in list [0, 1]
        # shape [bs, 1, H, W]
        batch, _, height, width = center_heatmap.size()
        topk_scores, topk_inds = torch.topk(center_heatmap.view(batch, -1), k)
        topk_clses = topk_inds // (height * width)
        topk_inds = topk_inds % (height * width)
        topk_ys = topk_inds // width
        topk_xs = (topk_inds % width).int().float()
        topk_ys = (topk_ys / 4).int().float()
        topk_xs = (topk_xs / 4).int().float()
        topk_inds = width // 4 * topk_ys + topk_xs
        topk_inds = topk_inds.long()
        return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs


    def get_local_minimum(self, heat, kernel=3):
        """Extract local minimum pixel with given kernel.

        Args:
            heat (Tensor): Target heatmap.
            kernel (int): Kernel size of max pooling. Default: 3.

        Returns:
            heat (Tensor): A heatmap where local minimum pixels maintain its
                own value and other positions are 0.
        """
        pad = (kernel - 1) // 2
        heat = 1 - torch.div(heat, 10)
        hmax = F.max_pool2d(heat, kernel, stride=1, padding=pad)
        keep = (hmax == heat).float()
        return heat * keep

    def _bboxes_nms(self, bboxes, labels, cfg):
        """NMS for multi-class bboxes."""
        if bboxes.shape[0] == 0:
            return bboxes, labels
            
        # 确保所有输入都在有效范围内
        scores = bboxes[..., -1].contiguous()
        bboxes = bboxes[..., :4].contiguous()
        
        try:
            # 确保数据类型正确
            bboxes = bboxes.float()
            scores = scores.float()
            labels = labels.long()  # 确保标签是 long 类型
            
            # 应用 NMS
            nms_cfg = dict(type='nms', iou_threshold=0.5)
            if hasattr(cfg, 'nms'):
                nms_cfg = cfg.nms
                
            # 使用 mmcv 的 nms
            from mmcv.ops import nms
            keep = nms(
                boxes=bboxes,
                scores=scores,
                iou_threshold=nms_cfg.get('iou_threshold', 0.5)
            )
            
            # 限制每张图片的检测框数量
            max_num = getattr(cfg, 'max_per_img', 100)
            if max_num > 0 and len(keep) > max_num:
                keep = keep[:max_num]
            
            # 使用 keep 索引选择结果
            bboxes = bboxes[keep]
            scores = scores[keep]
            labels = labels[keep]
            
            # 组合结果
            det_bboxes = torch.cat([bboxes, scores.unsqueeze(-1)], dim=-1)
            det_labels = labels
            
        except Exception as e:
            print(f"NMS error: {e}")
            print(f"bboxes shape: {bboxes.shape}, dtype: {bboxes.dtype}")
            print(f"scores shape: {scores.shape}, dtype: {scores.dtype}")
            print(f"labels shape: {labels.shape}, dtype: {labels.dtype}")
            # 发生错误时返回空结果
            return torch.zeros((0, 5), device=bboxes.device), torch.zeros((0,), device=labels.device, dtype=torch.long)
        
        return det_bboxes, det_labels


    def simple_test(self, feats, img_metas, rescale=False):
        """Test function without test-time augmentation."""
        center_heatmap_preds, xywh_preds = self.forward(feats)
        results_list = self.get_bboxes(
            center_heatmap_preds,
            xywh_preds,
            img_metas,
            rescale=rescale)
        
        # 确保结果格式正确
        bbox_results = []
        for result in results_list:
            if isinstance(result, tuple):
                bbox_pred, scores = result
                # 将结果转换为正确的格式
                bbox_result = []
                for i in range(len(self.num_classes)):
                    bbox_result.append(
                        np.concatenate([
                            bbox_pred[scores[:, i] > self.test_cfg.score_thr],
                            scores[scores[:, i] > self.test_cfg.score_thr, i:i+1]
                        ], axis=1))
                bbox_results.append(bbox_result)
            else:
                bbox_results.append(result)
        
        return bbox_results

    def simple_test_bboxes(self, feats, img_metas, rescale=False, crop=False):
        """Test det bboxes without test-time augmentation."""
        outs = self.forward(feats)
        
        # 获取检测结果
        det_bboxes, det_labels = self.get_bboxes(
            center_heatmap_preds=outs[0],
            xywh_preds=outs[1],
            img_metas=img_metas,
            rescale=rescale
        )
        
        # 确保返回格式正确
        if not isinstance(det_bboxes, list):
            det_bboxes = [det_bboxes]
        if not isinstance(det_labels, list):
            det_labels = [det_labels]
        
        # 组合结果为正确的格式
        results_list = list(zip(det_bboxes, det_labels))
        
        if crop:
            lsm_result = self.LSM(outs[0], img_metas)
            return lsm_result, results_list
        else:
            return results_list
    def LSM(self, center_heatmap_preds, img_metas):
        '''
        Args:
            center_heatmap_preds (list[Tensor]):  (N, C, H, W)
        '''
        center_heatmap_pred = center_heatmap_preds[0]
        locmap = torch.max(center_heatmap_pred, dim=1, keepdim=True)[0].cpu().numpy()
        
        coord = self.findclusters(locmap, find_max=True, fname=["test"])

        '''for visualization'''
        border_pixs = [img_meta['border'] for img_meta in img_metas]
        # coord [x, y, w, h]
        coord[:, 0] = coord[:, 0] - border_pixs[0][2]
        coord[:, 1] = coord[:, 1] - border_pixs[0][0]
        return coord

    def findclusters(self, heatmap, find_max, fname):
        heatmap = 1 - heatmap
        heatmap = 255*heatmap / np.max(heatmap)
        heatmap = heatmap[0][0]

        gray = heatmap.astype(np.uint8)
        Thresh = 10.0/11.0 * 255.0
        ret, binary = cv2.threshold(gray, Thresh, 255, cv2.THRESH_BINARY_INV)

        '''
            16 : 10
        '''
        binmap = binary.copy()
        binmap[binmap==255] = 1
        density_map = np.zeros((16, 10))
        w_stride = binary.shape[1]//16
        h_stride = binary.shape[0]//10
        for i in range(16):
            for j in range(10):
                x1 = w_stride*i
                y1 = h_stride*j
                x2 = min(x1+w_stride, binary.shape[1])
                y2 = min(y1+h_stride, binary.shape[0])
                density_map[i][j] = binmap[y1:y2,x1:x2].sum()

        d = density_map.flatten()
        topk = 15
        idx = d.argsort()[-topk:][::-1]
        grid_idx = idx.copy()
        idx_x = idx // 10 * w_stride
        idx_x = idx_x.reshape((-1, 1))
        idx_y = idx % 10 * h_stride
        idx_y = idx_y.reshape((-1, 1))
        idx = np.concatenate((idx_x, idx_y), axis=1)
        idx_2 = idx.copy()
        idx_2[:,0] = np.clip(idx[:,0]+w_stride, 0, binary.shape[1])
        idx_2[:,1] = np.clip(idx[:,1]+h_stride, 0, binary.shape[0])

        grid = np.zeros((16, 10))
        for item in grid_idx:
            x1 = item // 10
            y1 = item % 10
            grid[x1, y1] = 255
        result = split_overlay_map(grid)
        result = np.array(result)
        result[:,0::2] = np.clip(result[:, 0::2]*w_stride, 0,  binary.shape[1])
        result[:,1::2] = np.clip(result[:, 1::2]*h_stride, 0,  binary.shape[0])
        
        for i in range(len(result)):
            cv2.rectangle(binary, (result[i, 0], result[i, 1]), (result[i, 2], result[i, 3]), (255, 0, 0), 2)

        cv2.imwrite("binary_heatmap_%s4.jpg" %(fname[0]), binary)

        result[:, 2] = result[:, 2] - result[:, 0]
        result[:, 3] = result[:, 3] - result[:, 1]
        return result


def split_overlay_map(grid):
    # This function is modified from https://github.com/Cli98/DMNet
    """
        Conduct eight-connected-component methods on grid to connnect all pixel within the similar region
        :param grid: desnity mask to connect
        :return: merged regions for cropping purpose
    """
    if grid is None or grid[0] is None:
        return 0
    # Assume overlap_map is a 2d feature map
    m, n = grid.shape
    visit = [[0 for _ in range(n)] for _ in range(m)]
    count, queue, result = 0, [], []
    for i in range(m):
        for j in range(n):
            if not visit[i][j]:
                if grid[i][j] == 0:
                    visit[i][j] = 1
                    continue
                queue.append([i, j])
                top, left = float("inf"), float("inf")
                bot, right = float("-inf"), float("-inf")
                while queue:
                    i_cp, j_cp = queue.pop(0)
                    if 0 <= i_cp < m and 0 <= j_cp < n and grid[i_cp][j_cp] == 255:
                        top = min(i_cp, top)
                        left = min(j_cp, left)
                        bot = max(i_cp, bot)
                        right = max(j_cp, right)
                    if 0 <= i_cp < m and 0 <= j_cp < n and not visit[i_cp][j_cp]:
                        visit[i_cp][j_cp] = 1
                        if grid[i_cp][j_cp] == 255:
                            queue.append([i_cp, j_cp + 1])
                            queue.append([i_cp + 1, j_cp])
                            queue.append([i_cp, j_cp - 1])
                            queue.append([i_cp - 1, j_cp])

                            queue.append([i_cp - 1, j_cp - 1])
                            queue.append([i_cp - 1, j_cp + 1])
                            queue.append([i_cp + 1, j_cp - 1])
                            queue.append([i_cp + 1, j_cp + 1])
                count += 1
                result.append([max(0, top), max(0, left), min(bot+1, m), min(right+1, n)])

    return result
