# PMNet: Coarse to Fine: Progressive and Multi-Task Learning for Salient Object Detection

Pytorch implementation of "Coarse to Fine: Progressive and Multi-Task Learning for Salient Object Detection"

<img src="Figure/overall_net.png">

### Qualitative Evaluation

<img src="Figure/qualitative.png">

<img src="Figure/inside.png">

<img src="Figure/multi_abation.png">

### Quantative Evaluation

<img src="Figure/table.png">

<img src="Figure/fm_pr.png">

## Getting Started
### Installation

- Clone this repository
```
git clone https://github.com/tiruss/PMNet.git
```

- You can install all the dependencies by 
```
pip install -r requirements.txt
```

### Download datasets

- Download training datasets [[DUTS-TR]](http://saliencydetection.net/duts/download/DUTS-TR.zip) from the link 

- Download [[HKU-IS]](https://sites.google.com/site/ligb86/hkuis) for test from the link 

- Other datasets can download from the link [[sal_eval_toolbox]](https://github.com/ArcherFMY/sal_eval_toolbox) Thank you for the awesome evaluation toolbox!

### Run experiments from pretrained weight

- Download pretrained weight from the link 

### Train from scratch

- DUTS-TR is our traning set for pair comparison

- Run train.py
