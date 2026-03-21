# CrossScore: Towards Multi-View Image Evaluation and Scoring

**[Project Page](https://crossscore.active.vision) |
[arXiv](https://arxiv.org/abs/2404.14409) |
[Data Processing Script](https://github.com/ziruiw-dev/CrossScore-3DGS-Preprocessing)**

[Zirui Wang](https://scholar.google.com/citations?user=zCBKqa8AAAAJ&hl=en), 
[Wenjing Bian](https://scholar.google.com/citations?user=IVfbqkgAAAAJ&hl=en), 
[Victor Adrian Prisacariu](http://www.robots.ox.ac.uk/~victor). 

[Active Vision Lab (AVL)](https://www.robots.ox.ac.uk/~lav), 
University of Oxford.


## Table of Content
- [Installation (pip)](#installation-pip)
- [Quick Start](#quick-start)
- [Environment (conda, legacy)](#environment-conda-legacy)
- [Data](#Data)
- [Training](#Training)
- [Inferencing](#Inferencing)

## Installation (pip)

**Step 1**: Install PyTorch with your preferred CUDA version (see [pytorch.org](https://pytorch.org/get-started/locally/)):
```bash
# Example: PyTorch with CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Example: PyTorch with CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Example: CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Step 2**: Install CrossScore:
```bash
pip install crossscore
```

Or install from source:
```bash
git clone https://github.com/ActiveVisionLab/CrossScore.git
cd CrossScore
pip install -e .
```

> **Why install PyTorch separately?** PyTorch distributions are coupled with specific CUDA versions. By letting you install PyTorch first, we avoid version conflicts with your system's CUDA setup. CrossScore works with PyTorch 2.0+ and any CUDA version it supports.

## Quick Start

### Python API
```python
import crossscore

# Score query images against reference images
results = crossscore.score(
    query_dir="path/to/query/images",
    reference_dir="path/to/reference/images",
)

# The model checkpoint is auto-downloaded on first use (~129MB)
# Score maps are written to disk and returned as tensors
for score_map in results["score_maps"]:
    print(score_map.shape)  # (batch_size, H, W)
```

### Command Line
```bash
crossscore --query-dir path/to/queries --reference-dir path/to/references

# With options
crossscore --query-dir renders/ --reference-dir gt/ --metric-type mae --batch-size 4
```

### Environment Variables
- `CROSSSCORE_CKPT_PATH`: Use a specific local checkpoint instead of auto-downloading
- `CROSSSCORE_CACHE_DIR`: Custom cache directory (default: `~/.cache/crossscore`)

## Environment (conda, legacy)
We also provide a `environment.yaml` file to set up a `conda` environment:
```bash
git clone https://github.com/ActiveVisionLab/CrossScore.git
cd CrossScore
conda env create -f environment.yaml
conda activate CrossScore
```

## Data
**TLDR**: download this 
[file](https://www.robots.ox.ac.uk/~ryan/CrossScore/MFR_subset_demo.tar.gz) (~3GB), 
put it in `datadir`:
```bash
mkdir datadir
cd datadir
wget https://www.robots.ox.ac.uk/~ryan/CrossScore/MFR_subset_demo.tar.gz
tar -xzvf MFR_subset_demo.tar.gz
rm MFR_subset_demo.tar.gz
cd ..
```

To demonstrate a minimum working example for training and inferencing steps shown below, 
we provide a small pre-processed subset.
The is a subset of
[Map-Free Relocalisation (MFR)](https://research.nianticlabs.com/mapfree-reloc-benchmark/dataset)
and is pre-processed using 
[3D Gaussian Splatting (3DGS)](https://github.com/graphdeco-inria/gaussian-splatting).
This small demo dataset is available at this
[link](https://www.robots.ox.ac.uk/~ryan/CrossScore/MFR_subset_demo.tar.gz) (~3GB). 
This is the file in TLDR.
We only use this demo subset to present the expected dataloading structure.

In our actual training, our model is trained using MFR that pre-processed by three NVS methods:
[3DGS](https://github.com/graphdeco-inria/gaussian-splatting), 
[TensoRF](https://docs.nerf.studio/nerfology/methods/tensorf.html), and 
[NeRFacto](https://docs.nerf.studio/nerfology/methods/nerfacto.html).
Due to the preprocessed file size (~2TB), it is challenging to directly share
this pre-processed data. One work around is to release a data pre-processing script
for MFR.
~~**We aim to release the pre-processing script in Dec 2024.**~~
We have released the data pre-processing script at [this repository](https://github.com/ziruiw-dev/CrossScore-3DGS-Preprocessing).

## Training
We train our model with two NVIDIA A5000 (24GB) GPUs for about two days. 
However, the model should perform reasonably well after 12 hours of training. 
It is also possible to train with a single GPU.
```bash
python task/train.py trainer.devices='[0,1]'  # 2 GPUs
# python task/train.py trainer.devices='[0]'  # 1 GPU
```

## Inferencing
We provide an example command to predict CrossScore for NVS rendered images 
by referencing real captured images. 
```bash
git lfs install && git lfs pull  # get our ckpt using git LFS
bash predict.sh
```
After running the script, our CrossScore score maps should be written to `predict` dir.
The output should be similar to our 
[demo video](https://crossscore.active.vision/assets/additional_results.mp4)
on our project page.

## Todo
- [ ] Create a HuggingFace demo page.
- [ ] Release ECCV quantitative results related scripts.
- [x] Release [data processing scripts](https://github.com/ziruiw-dev/CrossScore-3DGS-Preprocessing)
- [x] Release PyPI package.

## Acknowledgement
This research is supported by an 
[ARIA](https://facebookresearch.github.io/projectaria_tools/docs/intro) 
research gift grant from Meta Reality Lab. We gratefully thank 
[Shangzhe Wu](http://elliottwu.com), 
[Tengda Han](https://tengdahan.github.io/), 
[Zihang Lai](https://scholar.google.com/citations?user=31eXgMYAAAAJ&hl=en) for insightful discussions, and 
[Michael Hobley](https://portraits.keble.net/2022/michael-hobley) for proofreading.

## Citation
```bibtex
@inproceedings{wang2024crossscore,
  title={CrossScore: Towards Multi-View Image Evaluation and Scoring},
  author={Zirui Wang and Wenjing Bian and Victor Adrian Prisacariu},
  booktitle={ECCV},
  year={2024}
}
```
