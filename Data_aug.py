# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 19:38:34 2025

@author: Kwaku Yeboah
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2
image_size =224

transform = A.Compose([
    # Simulate magnification change
    A.Resize(int(image_size * 1.2), int(image_size * 1.2)),
    #simulate zoom-in -multi-mag
    A.RandomCrop(image_size, image_size, p=1.0),

    # Geometric transforms
    A.Perspective(scale=(0.05, 0.1), keep_size=True, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),

    # Morphological noise
    A.ElasticTransform(alpha=50, sigma=5, alpha_affine=5, p=0.3),

    # Photometric variations
    A.CLAHE(clip_limit=2, tile_grid_size=(8, 8), p=0.2),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
    A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.5),

    # Low-resource artifacts
    A.GaussianBlur(blur_limit=(1, 3), p=0.3),
    A.MotionBlur(blur_limit=3, p=0.2),
    A.Downscale(scale_min=0.7, scale_max=0.9, interpolation=1, p=0.3),
    A.ImageCompression(quality_lower=30, quality_upper=70, p=0.3),
    A.ISONoise(color_shift=(0.01, 0.01), intensity=(0.1, 0.1), p=0.1),

    # Final resize to model input
    
    ToTensorV2()
])


