## Cross-Platform Video Person ReID: A New Benchmark Dataset and Adaptation Approach([PDF]())
### Installation

```
conda create -n vslaclip python=3.8
conda activate vslaclip
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
pip install yacs
pip install timm
pip install scikit-image
pip install tqdm
pip install ftfy
pip install regex
```

### Training

For example, if you want to run for the ls-vid, you need to modify the config file to

```
DATASETS:
   NAMES: ('lsvid')
   ROOT_DIR: ('your_dataset_dir')
OUTPUT_DIR: 'your_output_dir'
```
Then, if you want to use weight of [VIFI-CLIP](https://github.com/muzairkhattak/ViFi-CLIP) to initialize model, you need to down the weight form [link](https://github.com/muzairkhattak/ViFi-CLIP) and modify config file as:

```
MODEL:
  VIFI_WEIGHT : 'your_dataset_dir/vifi_weight.pth'
  USE_VIFI_WEIGHT : True
```
If you want to run FT-CLIP (fine tune image encoder):

```
CUDA_VISIBLE_DEVICES=0 python train_fine_tune.py --config_file configs/ft/vit_ft.yml
```

if you want to run VSLA-CLIP:

```
CUDA_VISIBLE_DEVICES=0 python train_reidadapter.py --config_file configs/adapter/vit_adapter.yml
```

### Evaluation

For example, if you want to test VSLA-CLIP for LS-VID

```
CUDA_VISIBLE_DEVICES=0 python test.py --config_file 'your_config_file' TEST.WEIGHT 'your_trained_checkpoints_path/ViT-B-16_120.pth'
```

### Weights
| Dataset    | LS-VID    | MARS | iLIDS                                                                                          | G2A   |
|------------|-----------|------|------------------------------------------------------------------------------------------------|-------|
| VSLA-CLIP‡ | [model](https://drive.google.com/drive/folders/1Wh4AJ9g59lZO_6trKEIloaLqKdU_j6ps?usp=sharing)     | [model](https://drive.google.com/drive/folders/1Wh4AJ9g59lZO_6trKEIloaLqKdU_j6ps?usp=sharing) | [model](https://drive.google.com/drive/folders/1Wh4AJ9g59lZO_6trKEIloaLqKdU_j6ps?usp=sharing)  | [model](https://drive.google.com/drive/folders/1Wh4AJ9g59lZO_6trKEIloaLqKdU_j6ps?usp=sharing) |

### Citation
```
@inproceedings{vsla-clip,
 author = {S. Zhang and W. Luo and D. Cheng and Q. Yang and L. Ran and Y. Xing and Y. Zhang},
 title  = {Cross-Platform Video Person ReID: A New Benchmark Dataset and Adaptation Approach},
 year   = {2024},
 booktitle = {ECCV}
}
```

### Acknowledgement

Codebase from [CLIP-ReID](https://github.com/Syliz517/CLIP-ReID), [TransReID](https://github.com/damo-cv/TransReID), [CLIP](https://github.com/openai/CLIP), and [CoOp](https://github.com/KaiyangZhou/CoOp).
