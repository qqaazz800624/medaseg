#%%

from monai.networks.nets import UNet
from monailabel.tasks.infer.basic_infer import BasicInferTask
from monailabel.tasks.scoring.epistemic_v2 import EpistemicScoring
from monailabel.datastore.local import LocalDatastore
import torch
from custom.mednext import mednext_base
from monai.transforms import (
    Compose,
    LoadImaged,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    EnsureTyped,
    EnsureChannelFirstd,
    Invertd,
    AsDiscreted,
)
from manafaln.transforms.intensity.normalize import NormalizeIntensityRanged

class SpleenSegInfer(BasicInferTask):
    def __init__(self, ckpt_path, spatial_size=(128, 128, 128), device="cuda"):
        model = mednext_base(spatial_dims=3,
                             in_channels=1,
                             out_channels=2,
                             kernel_size=3,
                             filters=32,
                             deep_supervision=True,
                             use_grad_checkpoint=True)
        model.load_state_dict(torch.load(ckpt_path))
        model.eval()
        super().__init__(path=ckpt_path, model=model, dimension=3, spatial_size=spatial_size, device=device)

    def pre_transforms(self, request=None):
        return Compose([
            LoadImaged(keys="image", image_only=True),
            EnsureChannelFirstd(keys="image"),
            Orientationd(keys="image", as_closest_canonical=True),
            Spacingd(keys="image", pixdim=[1.0, 1.0, 1.0], mode="bilinear"),
            NormalizeIntensityRanged(keys="image", a_min=-61.0, a_max=161.0,
                                     subtrahend=73.45, divisor=39.99)
        ])

infer_task = SpleenSegInfer(ckpt_path="/home/u/qqaazz800624/manafaln/mednext/Active_Learning/version_0/checkpoints/best_model.ckpt",
                            spatial_size=(128, 128, 128),
                            device="cuda")

scoring = EpistemicScoring(
    infer_task=infer_task,
    max_samples=10,
    simulation_size=20,
    use_variance=False
)

datastore = LocalDatastore(
    path = "/data2/open_dataset/MSD/Task09_Spleen/imagesTr",
    extensions=[".nii.gz"]
)


#%%








#%%








#%%