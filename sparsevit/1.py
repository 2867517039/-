import torch
from mmdet.apis import init_detector
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

def ensure_dir(path):
    """确保目录存在"""
    if not os.path.exists(path):
        os.makedirs(path)

def save_feature_map(feature, save_path, channel_idx=0):
    """保存特征图的辅助函数"""
    feature_vis = feature[0, channel_idx].cpu().numpy()
    feature_vis = (feature_vis - feature_vis.min()) / (feature_vis.max() - feature_vis.min()) * 255
    feature_vis = feature_vis.astype('uint8')
    cv2.imwrite(save_path, feature_vis)

def save_attention_map(attention, save_path, img_shape):
    """保存注意力图"""
    # 将注意力权重转换为热力图
    attention = attention.cpu().numpy()
    
    # 调整attention map的大小以匹配原始图像
    attention = cv2.resize(attention, (img_shape[1], img_shape[0]))
    
    # 归一化到0-1
    attention = (attention - attention.min()) / (attention.max() - attention.min())
    
    # 使用热力图颜色映射
    heatmap = cv2.applyColorMap(np.uint8(255 * attention), cv2.COLORMAP_JET)
    
    # 保存热力图
    cv2.imwrite(save_path, heatmap)
    return heatmap

def save_activation_distribution(activation, save_path):
    """保存激活值分布图"""
    plt.figure(figsize=(10, 6))
    activation_np = activation.cpu().numpy().flatten()
    
    # 绘制激活值分布直方图
    sns.histplot(activation_np, bins=50)
    plt.title('Activation Distribution')
    plt.xlabel('Activation Value')
    plt.ylabel('Count')
    
    plt.savefig(save_path)
    plt.close()

def visualize_layer_activations(feature_map, save_dir, layer_name):
    """可视化层的激活值"""
    # 获取特征图的统计信息
    mean_activation = torch.mean(feature_map, dim=[2, 3])
    max_activation = torch.max(torch.max(feature_map, dim=2)[0], dim=2)[0]
    
    # 保存激活值分布
    save_activation_distribution(
        feature_map.view(feature_map.size(0), -1), 
        os.path.join(save_dir, f'{layer_name}_activation_dist.png')
    )
    
    # 保存每个通道的最大激活值热力图
    for i in range(min(5, feature_map.size(1))):  # 只保存前5个通道
        channel_data = feature_map[0, i].cpu().numpy()
        plt.figure(figsize=(8, 6))
        sns.heatmap(channel_data, cmap='viridis')
        plt.title(f'Channel {i} Activation')
        plt.savefig(os.path.join(save_dir, f'{layer_name}_channel_{i}_heatmap.png'))
        plt.close()

def process_high_res_image():
    # 创建保存结果的目录
    result_dir = '/mnt/home/hks/平铺/sparsevit/results'
    ensure_dir(result_dir)
    
    # 1. 加载配置文件和权重
    config_file = '/mnt/home/hks/平铺/sparsevit/work_dirs/mask_rcnn_sparsevit_saa/mask_rcnn_sparsevit_saa.py'
    checkpoint_file = '/mnt/home/hks/平铺/sparsevit/work_dirs/mask_rcnn_sparsevit_saa/exp/epoch_12.pth'
    
    # 2. 初始化模型
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    print("模型加载完成")
    
    # 3. 读取并保存原始图片
    img_path = '/mnt/home/hks/平铺/sparsevit/demo/demo.jpg'
    original_img = cv2.imread(img_path)
    if original_img is None:
        print(f"无法读取图片: {img_path}")
        return
    print(f"原始图片尺寸: {original_img.shape}")
    cv2.imwrite(os.path.join(result_dir, '1_original.jpg'), original_img)

    # 4. 预处理图片
    img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(result_dir, '2_rgb.jpg'), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float()
    img_tensor = img_tensor.cuda()
    print(f"处理后的tensor尺寸: {img_tensor.shape}")

    # 注册hook来获取自注意力权重
    attention_maps = {}
    def hook_fn(module, input, output):
        if hasattr(module, 'attn_weights'):
            attention_maps[module.__class__.__name__] = module.attn_weights

    for name, module in model.named_modules():
        if 'self_attn' in name or 'attention' in name.lower():
            module.register_forward_hook(hook_fn)

    with torch.no_grad():
        # 5. Backbone特征提取
        features = model.backbone(img_tensor)
        print("Backbone特征提取完成")
        
        # 保存每个阶段的backbone特征和激活值
        for i, feat in enumerate(features):
            print(f"Backbone stage {i} 特征图尺寸: {feat.shape}")
            save_feature_map(feat, os.path.join(result_dir, f'3_backbone_stage_{i}.jpg'))
            visualize_layer_activations(feat, result_dir, f'backbone_stage_{i}')

        # 6. Neck特征提取
        if hasattr(model, 'neck') and model.neck is not None:
            neck_features = model.neck(features)
            print("Neck特征提取完成")
            
            for i, feat in enumerate(neck_features):
                print(f"Neck level {i} 特征图尺寸: {feat.shape}")
                save_feature_map(feat, os.path.join(result_dir, f'4_neck_level_{i}.jpg'))
                visualize_layer_activations(feat, result_dir, f'neck_level_{i}')

        # 7. 保存注意力图
        for name, attn_weights in attention_maps.items():
            if isinstance(attn_weights, torch.Tensor):
                # 取平均注意力权重
                avg_attention = attn_weights[0].mean(dim=0)
                attention_map = save_attention_map(
                    avg_attention,
                    os.path.join(result_dir, f'attention_map_{name}.jpg'),
                    original_img.shape
                )
                
                # 叠加注意力图到原图
                alpha = 0.5
                overlay = cv2.addWeighted(original_img, 1-alpha, attention_map, alpha, 0)
                cv2.imwrite(os.path.join(result_dir, f'attention_overlay_{name}.jpg'), overlay)

        # 8. RPN特征提取和处理
        if hasattr(model, 'rpn_head') and model.rpn_head is not None:
            rpn_outs = model.rpn_head(neck_features)
            print("RPN特征提取完成")
            
            for i, rpn_out in enumerate(rpn_outs):
                if isinstance(rpn_out, tuple):
                    cls_score, bbox_pred = rpn_out
                    print(f"RPN output {i} - cls: {cls_score.shape}, bbox: {bbox_pred.shape}")
                    save_feature_map(cls_score, os.path.join(result_dir, f'5_rpn_cls_{i}.jpg'))
                    save_feature_map(bbox_pred, os.path.join(result_dir, f'5_rpn_bbox_{i}.jpg'))
                    
                    # 可视化RPN的激活值
                    visualize_layer_activations(cls_score, result_dir, f'rpn_cls_{i}')
                    visualize_layer_activations(bbox_pred, result_dir, f'rpn_bbox_{i}')

            # 获取proposals
            try:
                proposals = model.rpn_head.get_proposals(
                    cls_scores=[o[0] for o in rpn_outs],
                    bbox_preds=[o[1] for o in rpn_outs],
                    img_metas=[{'img_shape': original_img.shape}],
                    cfg=model.test_cfg.rpn
                )
                
                # 在原图上画出proposals
                img_with_proposals = original_img.copy()
                for proposal in proposals[0]:
                    x1, y1, x2, y2 = map(int, proposal[:4])
                    cv2.rectangle(img_with_proposals, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.imwrite(os.path.join(result_dir, '6_proposals.jpg'), img_with_proposals)
            except Exception as e:
                print(f"获取proposals时出错: {e}")

        # 9. ROI特征提取和处理
        if hasattr(model, 'roi_head') and model.roi_head is not None:
            try:
                roi_feats = model.roi_head.extract_feat(neck_features)
                if isinstance(roi_feats, torch.Tensor):
                    print(f"ROI特征图尺寸: {roi_feats.shape}")
                    save_feature_map(roi_feats, os.path.join(result_dir, '7_roi_feats.jpg'))
                    visualize_layer_activations(roi_feats, result_dir, 'roi_features')
            except Exception as e:
                print(f"获取ROI特征时出错: {e}")

        # 10. 最终检测结果
        try:
            result = model.inference(img_tensor, [{'img_shape': original_img.shape}])
            
            # 在原图上画出最终检测结果
            final_img = original_img.copy()
            for bbox in result[0]:
                if len(bbox) >= 5:  # 确保bbox包含坐标和置信度
                    x1, y1, x2, y2, score = map(float, bbox[:5])
                    if score > 0.3:  # 设置置信度阈值
                        cv2.rectangle(final_img, 
                                    (int(x1), int(y1)), 
                                    (int(x2), int(y2)), 
                                    (0, 0, 255), 2)
                        cv2.putText(final_img, 
                                  f'{score:.2f}', 
                                  (int(x1), int(y1)-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.5, 
                                  (0, 0, 255), 
                                  2)
            cv2.imwrite(os.path.join(result_dir, '8_final_detection.jpg'), final_img)
        except Exception as e:
            print(f"生成最终检测结果时出错: {e}")

    print("所有处理步骤完成，包括注意力分布和激活值可视化")

if __name__ == '__main__':
    process_high_res_image()