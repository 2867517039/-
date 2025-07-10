# Copyright (c) OpenMMLab. All rights reserved.
import torch
import numpy as np
import torch.nn as nn
from mmdet.core import bbox2result
from mmdet.models.builder import DETECTORS
from mmdet.core.utils import flip_tensor
from mmdet.models.detectors.single_stage import SingleStageDetector


@DETECTORS.register_module()
class YOLC(SingleStageDetector):
    """Implementation of YOLC

    <https://arxiv.org/abs/2404.06180>.
    """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(YOLC, self).__init__(backbone, neck, bbox_head, train_cfg,
                                        test_cfg, pretrained, init_cfg)
    def train(self, mode=True):
        """Convert the model into training mode."""
        super().train(mode)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.train(mode)
            # 确保所有参数都需要梯度
            for p in m.parameters():
                p.requires_grad_(True)
        return self

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation."""
        feat = self.extract_feat(img)
        results_list = self.bbox_head.simple_test_bboxes(feat, img_metas, rescale=rescale)
        
        # 处理返回结果
        if isinstance(results_list, tuple) and len(results_list) == 2:
            det_bboxes, det_labels = results_list
        else:
            det_bboxes, det_labels = results_list[0]
        
        # 确保 det_bboxes 和 det_labels 是列表
        if not isinstance(det_bboxes, list):
            det_bboxes = [det_bboxes]
        if not isinstance(det_labels, list):
            det_labels = [det_labels]
        
        # 处理每张图片的结果
        results = []
        for bboxes, labels in zip(det_bboxes, det_labels):
            # 确保数据类型正确
            bboxes = bboxes.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            
            # 转换为每个类别的结果
            result = []
            for i in range(self.bbox_head.num_classes):
                # 获取当前类别的掩码
                mask = labels == i
                # 获取当前类别的检测框和分数
                cls_bboxes = bboxes[mask]
                if len(cls_bboxes) > 0:
                    result.append(cls_bboxes)
                else:
                    result.append(np.zeros((0, 5), dtype=np.float32))
            
            # 将结果包装成字典格式
            results.append({
                'bbox': result  # 添加 bbox 键
            })
        
        # 打印调试信息
        print("Results format:")
        for i, result in enumerate(results):
            print(f"Image {i}:")
            print(f"  keys: {result.keys()}")
            print(f"  bbox length: {len(result['bbox'])}")
            for j, bbox in enumerate(result['bbox']):
                print(f"    class {j}: shape {bbox.shape}")
        
        return results

    def merge_aug_results(self, aug_results, with_nms):
        """Merge augmented detection bboxes and score.

        Args:
            aug_results (list[list[Tensor]]): Det_bboxes and det_labels of each
                image.
            with_nms (bool): If True, do nms before return boxes.

        Returns:
            tuple: (out_bboxes, out_labels)
        """
        recovered_bboxes, aug_labels = [], []
        for single_result in aug_results:
            recovered_bboxes.append(single_result[0][0])
            aug_labels.append(single_result[0][1])

        bboxes = torch.cat(recovered_bboxes, dim=0).contiguous()
        labels = torch.cat(aug_labels).contiguous()
        if with_nms:
            out_bboxes, out_labels = self.bbox_head._bboxes_nms(
                bboxes, labels, self.bbox_head.test_cfg)
        else:
            out_bboxes, out_labels = bboxes, labels

        return out_bboxes, out_labels

    def aug_test(self, imgs, img_metas, rescale=True):
        """Augment testing of CenterNet. Aug test must have flipped image pair,
        and unlike CornerNet, it will perform an averaging operation on the
        feature map instead of detecting bbox.

        Args:
            imgs (list[Tensor]): Augmented images.
            img_metas (list[list[dict]]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Default: True.

        Note:
            ``imgs`` must including flipped image pairs.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        img_inds = list(range(len(imgs)))
        assert img_metas[0][0]['flip'] + img_metas[1][0]['flip'], (
            'aug test must have flipped image pair')
        aug_results = []
        for ind, flip_ind in zip(img_inds[0::2], img_inds[1::2]):
            flip_direction = img_metas[flip_ind][0]['flip_direction']
            img_pair = torch.cat([imgs[ind], imgs[flip_ind]])
            x = self.extract_feat(img_pair)
            center_heatmap_preds, xywh_preds_coarse, xywh_preds_refine = self.bbox_head(x)
            assert len(center_heatmap_preds) == len(xywh_preds_coarse) == len(xywh_preds_refine)  == 1

            # Feature map averaging
            center_heatmap_preds[0] = (
                center_heatmap_preds[0][0:1] +
                flip_tensor(center_heatmap_preds[0][1:2], flip_direction)) / 2
            xywh_preds_refine[0] = xywh_preds_refine[0][0:1]

            bbox_list = self.bbox_head.get_bboxes(
                center_heatmap_preds,
                xywh_preds_coarse,
                xywh_preds_refine,
                img_metas[ind],
                rescale=rescale,
                with_nms=False)
            aug_results.append(bbox_list)

        nms_cfg = self.bbox_head.test_cfg.get('nms_cfg', None)
        if nms_cfg is None:
            with_nms = False
        else:
            with_nms = True
        bbox_list = [self.merge_aug_results(aug_results, with_nms)]
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results
    
