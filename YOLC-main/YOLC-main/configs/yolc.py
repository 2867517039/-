dataset_type = 'VisDroneDataset'
data_root = '/mnt/home/hks/平铺/YOLC-main/YOLC-main/data/VisDrone2019/'
classes = ('pedestrian', "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor")
num_classes = 10
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, color_type='color'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='RandomCenterCropPad',
        crop_size=(1024, 640),
        ratios=(0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3),
        mean=[0, 0, 0],
        std=[1, 1, 1],
        to_rgb=True,
        test_pad_mode=None),
    dict(type='Resize', img_scale=(1024, 640), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(
        type='MultiScaleFlipAug',
        scale_factor=1.0,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(
                type='RandomCenterCropPad',
                ratios=None,
                border=None,
                mean=[0, 0, 0],
                std=[1, 1, 1],
                to_rgb=True,
                test_mode=True,
                test_pad_mode=['logical_or', 31],
                test_pad_add_pix=1),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                           'scale_factor', 'flip', 'flip_direction',
                           'img_norm_cfg', 'border'),
                keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset = dict(type=dataset_type,
            classes=classes,
            ann_file='/mnt/home/hks/平铺/YOLC-main/YOLC-main/data/VisDrone2019/VisDrone2019-DET_val_coco.json',
            img_prefix='/mnt/home/hks/平铺/YOLC-main/YOLC-main/data/VisDrone2019/VisDrone2019-DET-val/images',
            pipeline=[
                dict(
                    type='LoadImageFromFile',
                    to_float32=True,
                    color_type='color'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(
                    type='PhotoMetricDistortion',
                    brightness_delta=32,
                    contrast_range=(0.5, 1.5),
                    saturation_range=(0.5, 1.5),
                    hue_delta=18),
                dict(
                    type='RandomCenterCropPad',
                    crop_size=(1024, 640),
                    ratios=(0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3),
                    mean=[0, 0, 0],
                    std=[1, 1, 1],
                    to_rgb=True,
                    test_pad_mode=None),
                dict(type='Resize', img_scale=(1024, 640), keep_ratio=True),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
            ])),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file='/mnt/home/hks/平铺/YOLC-main/YOLC-main/data/VisDrone2019/VisDrone2019-DET_val_coco.json',
        img_prefix='/mnt/home/hks/平铺/YOLC-main/YOLC-main/data/VisDrone2019/VisDrone2019-DET-val/images',

        pipeline=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(
                type='MultiScaleFlipAug',
                scale_factor=1.0,
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(
                        type='RandomCenterCropPad',
                        ratios=None,
                        border=None,
                        mean=[0, 0, 0],
                        std=[1, 1, 1],
                        to_rgb=True,
                        test_mode=True,
                        test_pad_mode=['logical_or', 31],
                        test_pad_add_pix=1),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='DefaultFormatBundle'),
                    dict(
                        type='Collect',
                        meta_keys=('filename', 'ori_shape', 'img_shape',
                                   'pad_shape', 'scale_factor', 'flip',
                                   'flip_direction', 'img_norm_cfg', 'border'),
                        keys=['img'])
                ])
        ]),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file='/mnt/home/hks/平铺/YOLC-main/YOLC-main/data/VisDrone2019/VisDrone2019-DET_val_coco.json',
        img_prefix='/mnt/home/hks/平铺/YOLC-main/YOLC-main/data/VisDrone2019/VisDrone2019-DET-val/images',
        pipeline=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(
                type='MultiScaleFlipAug',
                scale_factor=1.0,
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(
                        type='RandomCenterCropPad',
                        ratios=None,
                        border=None,
                        mean=[0, 0, 0],
                        std=[1, 1, 1],
                        to_rgb=True,
                        test_mode=True,
                        test_pad_mode=['logical_or', 31],
                        test_pad_add_pix=1),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='DefaultFormatBundle'),
                    dict(
                        type='Collect',
                        meta_keys=('filename', 'ori_shape', 'img_shape',
                                   'pad_shape', 'scale_factor', 'flip',
                                   'flip_direction', 'img_norm_cfg', 'border'),
                        keys=['img'])
                ])
        ],
        gpu_collect=True,  # 添加这个
        workers_per_gpu=2   # 添加这个
        ))
# 修改优化器配置
# 使用更保守的优化器
optimizer = dict(
    type='SGD',  # 改用 SGD
    lr=1e-4,     # 适中的学习率
    momentum=0.9,
    weight_decay=0.0001
)

# 更严格的梯度裁剪
optimizer_config = dict(
    grad_clip=dict(max_norm=0.1, norm_type=2)  # 更小的梯度裁剪阈值
)

# 更温和的学习率调度
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[8, 11],
    gamma=0.1  # 添加学习率衰减因子
)

# 训练配置
# runner 配置
runner = dict(type='EpochBasedRunner', max_epochs=48)  # 确保与 total_epochs 一致

# 检查点配置
checkpoint_config = dict(interval=1)  # 每个 epoch 保存一次

# 日志配置
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ])

# 损失函数权重
model = dict(
    bbox_head=dict(
        loss_center_heatmap=dict(type='GaussianFocalLoss', loss_weight=0.01),
        loss_xywh_coarse=dict(type='L1Loss', loss_weight=0.001),
        loss_xywh_coarse_l1=dict(type='L1Loss', loss_weight=0.001),
        loss_xywh_refine=dict(type='L1Loss', loss_weight=0.001),
        loss_xywh_refine_l1=dict(type='L1Loss', loss_weight=0.001),
    )
)
# 评估配置
evaluation = dict(
    interval=1,  # 每个 epoch 评估一次
    metric=['bbox'],  # 评估指标
    classwise=True,  # 分类别评估
    proposal_nums=(100, 300, 1000),  # 评估不同数量的提议
    iou_thrs=None,  # 使用默认的 IoU 阈值
)
runner = dict(type='EpochBasedRunner', max_epochs=48)
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
model = dict(
    type='YOLC',
    backbone=dict(
        type='SparseViT',
        pretrain_img_size=224,  # 根据需要调整
        in_channels=3,  # 输入通道数
        embed_dims=96,  # 嵌入维度
        patch_size=4,  # patch大小
        window_size=7,  # 窗口大小
        mlp_ratio=4,  # MLP比率
        depths=(2, 2, 6, 2),  # 每个阶段的深度
        num_heads=(3, 6, 12, 24),  # 每个阶段的注意力头数
        strides=(4, 2, 2, 2),  # 每个阶段的步幅
        out_indices=(0, 1, 2, 3),  # 输出的阶段索引
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='LN'),
        with_cp=False,
        init_cfg=dict(type='Pretrained', checkpoint='/mnt/home/hks/平铺/Y+S/sparsevit/work_dirs/mask_rcnn_sparsevit_saa/exp/epoch_12.pth')
    ),
    neck=dict(
        type='HRFPN',
        in_channels=[96, 192, 384, 768],  # 根据SparseViT的输出通道数调整
        out_channels=256,  # HRFPN输出通道数
        num_outs=5  # 输出的特征层数
    ),
    bbox_head=dict(
        type='YOLCHead',
        num_classes=10,
        in_channel=256,  # 根据HRFPN的输出通道数调整
        feat_channel=96,
        loss_center_local=dict(type='GaussianFocalLoss', loss_weight=1.0),
        loss_xywh=dict(type='GWDLoss', loss_weight=2.0)),
    train_cfg=None,
    test_cfg=dict(
            nms=dict(type='nms', iou_threshold=0.5),  # 添加 NMS 配置
            score_thr=0.05,  # 分数阈值
            max_per_img=100  # 每张图片最多检测框数量
        ))
work_dir = './work_dir/yolc_sparsevit'
gpu_ids = range(0, 5)
