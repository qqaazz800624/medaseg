# MONAILabel Adapter Example

## Setup

1. Dowload sample apps from MONAILabel repository:
```
git clone https://github.com/Project-MONAI/MONAILabel.git
```

2. Add manafaln segmentation app config to sample radiology app:
```
cp -r config MONAILabel/sample-apps/radiology
```

3. Patch MONAILabel to add ManafalnSegmentation model
```
cp manafaln_segmentation.py MONAILabel/sample-apps/radiology/lib/configs
```

4. Prepare your dataset

5. Prepare your pretrained model weight and save to `MONAILabel/sample-apps/radiology/model`

6. Run our MONAILabel docker image to start monailabel app
```
docker run -it --name monailabel --net=host --gpus=all \
  -v $PWD/MONAILabel/sample-apps:/workspace \
  -v <PATH_TO_YOUR_DATASET>:/data \
  -w /workspace/radiology \
  nanaha1003/monailabel:latest \
  monailabel start_server \
    --app /workspace/radiology \
    --studies /data \
    --conf models manafaln_segmentation \
    --conf mfn_config config/active_learning.yaml
```
