from typing import Dict, Hashable, List, Mapping, Union

import torch
from monai.data import MetaTensor
from monai.config import KeysCollection
from monai.transforms import MapTransform

class DeepSupervisionSplitDimd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        output_list_postfix: str = "_ds",
        replace_preds: bool = True,
        allow_missing_keys: bool = False
    ):
        super().__init__(keys, allow_missing_keys)

        self.output_list_postfix = output_list_postfix
        self.replace_preds = replace_preds

    def __call__(
        self,
        data: Mapping[Hashable, torch.Tensor]
    ) -> Dict[Hashable, Union[torch.Tensor, List[torch.Tensor]]]:
        d = dict(data)
        for key in self.key_iterator(d):
            # Split the deep supervision outputs
            num_outputs = d[key].shape[1]
            preds = [d[key][:, i, ::] for i in range(num_outputs)]

            # Repalce original deep supervision preds with normal preds
            if self.replace_preds:
                if isinstance(d[key], MetaTensor):
                    d[key] = MetaTensor(preds[0], d[key].affine, d[key].meta)
                else:
                    d[key] = preds[0]
            # Save deep supervision outputs to list for loss calculation
            d[key + self.output_list_postfix] = preds
        return d
