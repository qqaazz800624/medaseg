from manafaln.transforms import (
    NormalizeIntensityRanged, RandAdjustBrightnessAndContrastd,
    SimulateLowResolutiond, RandInverseIntensityGammad, RandFlipAxes3Dd
)
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, RandRotated,
    RandZoomd, SpatialPadd, RandCropByPosNegLabeld,
    RandGaussianNoised, RandGaussianSmoothd,
    RandAdjustContrastd,
    EnsureTyped
)
from custom.meta import SaveMeta

def get_pinkcc_transforms(stage, spacing, 
                          roi_size, 
                          intensity_min, 
                          intensity_max, 
                          intensity_mean, 
                          intensity_std, 
                          num_samples):
    """
    Return a list of MONAI transforms for the given stage.

    Args:
        stage (str): One of "training", "validation", "test", or "predict"
        spacing (list): Target voxel spacing
        roi_size (list): Target ROI size
        intensity_min, intensity_max, intensity_mean, intensity_std (float): Normalization parameters
        num_samples (int): Number of samples for RandCropByPosNegLabeld

    Returns:
        List[monai.transforms.Transform]: A list of MONAI transforms
    """
    common_pre_transforms = [
        LoadImaged(keys=["image", "label"] if stage != "predict" else ["image"], image_only=(stage != "predict")),
        EnsureChannelFirstd(keys=["image", "label"] if stage != "predict" else ["image"]),
        Orientationd(keys=["image", "label"] if stage != "predict" else ["image"], as_closest_canonical=True),
        Spacingd(keys=["image", "label"] if stage != "predict" else ["image"],
                 pixdim=spacing,
                 mode=("bilinear", "nearest") if stage != "predict" else "bilinear"),
    ]

    if stage == "training":
        return common_pre_transforms + [
            RandRotated(keys=["image", "label"], range_x=0.5236, range_y=0.5236, range_z=0.5236, 
                        prob=0.2, keep_size=False, mode=["bilinear", "nearest"]),
            RandZoomd(keys=["image", "label"], prob=0.2, min_zoom=0.7, max_zoom=1.4, 
                      mode=["trilinear", "nearest"], keep_size=False),
            NormalizeIntensityRanged(keys=["image"], a_min=intensity_min, a_max=intensity_max, 
                                     subtrahend=intensity_mean, divisor=intensity_std),
            SpatialPadd(keys=["image", "label"], spatial_size=roi_size),
            RandCropByPosNegLabeld(keys=["image", "label"], label_key="label", 
                                   spatial_size=roi_size, pos=2.0, neg=1.0, 
                                   num_samples=num_samples),
            RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.1),
            RandGaussianSmoothd(keys=["image"], sigma_x=[0.5, 1.5], sigma_y=[0.5, 1.5], 
                                sigma_z=[0.5, 1.5], prob=0.15),
            RandAdjustBrightnessAndContrastd(keys=["image"], probs=[0.15, 0.15], 
                                             brightness_range=[0.7, 1.3], contrast_range=[0.65, 1.5]),
            SimulateLowResolutiond(keys=["image"], prob=0.25, zoom_range=[0.5, 1.0]),
            RandAdjustContrastd(keys=["image"], prob=0.15, gamma=[0.8, 1.2]),
            RandInverseIntensityGammad(keys=["image"], prob=0.15, gamma=[0.8, 1.2]),
            RandFlipAxes3Dd(keys=["image", "label"], prob_x=0.5, prob_y=0.5, prob_z=0.5),
            EnsureTyped(keys=["image", "label"])
        ]
    elif stage in ("validation", "test"):
        return common_pre_transforms + [
            NormalizeIntensityRanged(keys=["image"], a_min=intensity_min, a_max=intensity_max, 
                                     subtrahend=intensity_mean, divisor=intensity_std),
            SaveMeta(keys=["image", "label"], meta_keys=["image_meta_dict", "label_meta_dict"]),
            EnsureTyped(keys=["image", "label"])
        ]
    elif stage == "predict":
        return common_pre_transforms + [
            NormalizeIntensityRanged(keys=["image"], a_min=intensity_min, a_max=intensity_max, 
                                     subtrahend=intensity_mean, divisor=intensity_std),
            SaveMeta(keys=["image"], meta_keys=["image_meta_dict", "label_meta_dict"]),
            EnsureTyped(keys=["image"])
        ]
    else:
        raise ValueError(f"Unknown stage: {stage}")
