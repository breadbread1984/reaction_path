# Introduction

this project is a pytorch reproduce of [this project](https://github.com/CederGroupHub/SynthesisSimilarity).

# Usage

## download dataset

download **rsc.zip** from [google drive](https://drive.google.com/uc?id=1ack7mcyHtUVMe99kRARvdDV8UhweElJ4). unzip rsc.zip under the root directory of this project.

## train models

```shell
python3 train.py --dataset rsc/data_split.npz --device (cpu|cuda)
```

