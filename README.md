# Collaborative Learning of Anomalies with Privacy (CLAP) for Unsupervised Video Anomaly Detection: A New Baseline [CVPR 2024]
[![Website](https://img.shields.io/badge/Project-Website-87CEEB)](https://anasemad11.github.io/CLAP/)
[![Dataset Splits](https://img.shields.io/badge/Dataset-Access-<COLOR>)](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/anas_al-lahham_mbzuai_ac_ae/Ek7OQNDf9tBLqk7AfH4CPAgBP9cvtjCZnIWbrfwGogXlsA?e=TwuRwr)
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2404.00847)


> [**Collaborative Learning of Anomalies with Privacy (CLAP) for Unsupervised Video Anomaly Detection: A New Baseline**](https://arxiv.org/abs/2404.00847)<br>
> [Anas Al-lahham](https://anasemad11.github.io/), [Muhammad Zaigham Zaheer](https://www.linkedin.com/in/zaighamzaheer/?originalSubdomain=kr), [Nubrek Tastan](https://tnurbek.github.io/), [Karthik Nandakumar](https://www.linkedin.com/in/karthik-nandakumar-5504465/)

Official implementation of the paper: "Collaborative Learning of Anomalies with Privacy (CLAP) for Unsupervised Video Anomaly Detection: A New Baseline" [CVPR 2024].

## Overview
![abstract figure](imgs/github_cvpr_mainfig.drawio.png)
> **<p align="justify"> Abstract:** Unsupervised (US) video anomaly detection (VAD) in surveillance applications is gaining more popularity recently due to its practical real-world applications. As surveillance videos are privacy sensitive and the availability of large-scale video data may enable better US-VAD systems, collaborative learning can be highly rewarding in this setting. However, due to the extremely challenging nature of the US-VAD task, where learning is carried out without any annotations, privacy-preserving collaborative learning of US-VAD systems has not been studied yet. In this paper, we propose a new baseline for anomaly detection capable of localizing anomalous events in complex surveillance videos in a fully unsupervised fashion without any labels on a privacy-preserving participant-based distributed training configuration. Additionally, we propose three new evaluation protocols to benchmark anomaly detection approaches on various scenarios of collaborations and data availability. Based on these protocols, we modify existing VAD datasets to extensively evaluate our approach as well as existing US SOTA methods on two large-scale datasets including UCF-Crime and XD-Violence.


## Requirements 
Follow these steps to set up a conda environment and ensure all necessary packages are installed:

```bash
git clone https://github.com/AnasEmad11/CLAP.git
cd CLAP

conda create -n FLAD python=3.7
conda activate FLAD

# The results are produced with PyTorch 1.12.1 and CUDA 11.3
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

pip install -r requirements.txt
```


### Dataset

To setup all the available datasets, plesae download the concatenated features from this [link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/anas_al-lahham_mbzuai_ac_ae/Ek7OQNDf9tBLqk7AfH4CPAgBP9cvtjCZnIWbrfwGogXlsA?e=TwuRwr)

All the required dataset splits are availble in the project webpage and can be downloaded from [Dataset Splits](https://github.com/AnasEmad11/CLAP/raw/webpage_v3/dataset_splits.zip)

### Dataset Distribution for each split
<div align="center">
    <img src="docs/static/images/data_dist.png"  alt=" Dataset  dist">
    <p><b>Distribution of UCF-Crime dataset videos based on the three training data organizations proposed in our paper to evaluate collaborative learning approaches for video Anomaly Detection.</b></p>
</div>



## CLAP Results

We evaluate CLAP on 3 different FL methods on the scene-based split.

|              | FedAvg | FedProx | SCAFFOLD |
|--------------|--------|---------|----------|
| **CLAP**     | 73.99% | 73.4%   | 73.7%    |

---

Comparison of CLAP with other SOTA Unsupervised and Weakly Supervised methods

|              | Method | UCF-Crime (US) | UCF-Crime (WS) | XD-Violence (US) | XD-Violence (WS) |
|--------------|--------|-----------------|----------------|------------------|------------------|
|  | GCL    | 71.04%          | 79.84%         | 73.62%           | 82.18%           |
|      **Centralized**        | C2FPL  | 80.65%          | 83.40%         | 80.09%           | 89.34%           |
|              | **CLAP** | **80.9%**       | **85.50%**     | **81.71%**       | **90.04%**       |
|      | GCL    | 56.63%          | 65.32%         | 58.11%           | 59.91%           |
|    **Local**           | C2FPL  | 61.33%          | 65.85%         | 60.05%           | 63.4%            |
|              | **CLAP** | **63.93%**      | **67.47%**     | **62.37%**       | **64.97%**       |
|  | GCL    | 67.12%       | 76.82%         | 68.19%           | 75.21%           |
|       **Collaborative**       | C2FPL  | 75.20%          | 77.60%         | 74.36%           | 76.98%           |
|              | **CLAP** | **78.02%**      | **83.23%**     | **77.65%**       | **85.67%**       |



<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@inproceedings{al2024collaborative,
  title={Collaborative Learning of Anomalies with Privacy (CLAP) for Unsupervised Video Anomaly Detection: A New Baseline},
  author={Al-Lahham, Anas and Zaheer, Muhammad Zaigham and Tastan, Nurbek and Nandakumar, Karthik},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={12416--12425},
  year={2024}
}
}</code></pre>
  </div>
</section>




