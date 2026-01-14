<p align="center">
  <img src="https://github.com/PedroRASB/Cerberus/blob/main/misc/LabelCriticTitle.png" alt="Project Logo" width="900"/>
</p>

<p align="center">
  <img src="https://github.com/PedroRASB/Cerberus/blob/main/misc/LabelCriticModel.png" alt="Project Logo" width="900"/>
</p>

Label Critic is an automated tool for reviewing AI-generated labels.
It helps users select better annotations when multiple label options exist,
and identify potentially incorrect labels when only a single annotation is available.

Label Critic uses pre-trained Large Vision-Language Models (LVLMs)
as label critics, comparing or assessing annotations without training new models.
In medical CT organ segmentation, it achieves 96.5% accuracy
in selecting higher-quality annotations per scan and class.


## Paper

<b>Label Critic: Design Data Before Models</b> <br/>
Pedro R. A. S. Bassi, Qilong Wu, Wenxuan Li, Sergio Decherchi, Andrea Cavalli, Alan Yuille, Zongwei Zhou <br/>
International Symposium on Biomedical Imaging (ISBI, 2025) <br/>
[Read More](https://arxiv.org/abs/2411.02753) <br/>

<br/>

[📄 View the ISBI Poster](https://github.com/PedroRASB/LabelCritic/raw/main/misc/ISBI_Label_critic_poster.pdf)

[![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://youtu.be/9D5-pFgtgDQ)




## Getting Started


### Installation

We recommend using Anaconda on Linux.


```bash
git clone https://github.com/PedroRASB/AnnotationVLM
cd AnnotationVLM
conda create -n vllm python=3.12 -y
conda activate vllm
conda install -y ipykernel
conda install -y pip
pip install vllm==0.6.1.post2
pip install git+https://github.com/huggingface/transformers@21fac7abba2a37fae86106f87fcf9974fd1e3830
pip install -r requirements.txt
mkdir HFCache
```



### Deploy Vision–Language Model Backend



```bash
export NCCL_P2P_DISABLE=1

TRANSFORMERS_CACHE=./HFCache \
HF_HOME=./HFCache \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
vllm serve "Qwen/Qwen2-VL-72B-Instruct-AWQ" \
  --dtype=half \
  --tensor-parallel-size 4 \
  --limit-mm-per-prompt image=3 \
  --gpu_memory_utilization 0.9 \
  --port 8000
```

We recommend using ≥ 4 A40 GPUs (48GB VRAM each) for stable deployment.
You can try different VL models, e.g.: Qwen/Qwen2-VL-2B-Instruct-AWQ.




## Usage

Label Critic supports multiple usage scenarios depending on the available
annotations and the desired level of analysis.
### Scenario 1: Compare Two Annotation Sets (Dataset-Level)


<details> <summary><strong>Dataset format (click to expand)</strong></summary>
<div style="margin-left: 25px;">

```
Dataset
├── BDMAP_A0000001
|    ├── ct.nii.gz
│    └── predictions1
│          ├── liver_tumor.nii.gz
│          ├── kidney_tumor.nii.gz
│          ├── pancreas_tumor.nii.gz
│          ├── aorta.nii.gz
│          ├── gall_bladder.nii.gz
│          ├── kidney_left.nii.gz
│          ├── kidney_right.nii.gz
│          ├── liver.nii.gz
│          ├── pancreas.nii.gz
│          └──...
│    └── predictions2
│          ├── liver_tumor.nii.gz
│          ├── kidney_tumor.nii.gz
│          ├── pancreas_tumor.nii.gz
│          ├── aorta.nii.gz
│          ├── gall_bladder.nii.gz
│          ├── kidney_left.nii.gz
│          ├── kidney_right.nii.gz
│          ├── liver.nii.gz
│          ├── pancreas.nii.gz
│          └──...
├── BDMAP_A0000002
|    ├── ct.nii.gz
│    └── predictions
│          ├── liver_tumor.nii.gz
│          ├── kidney_tumor.nii.gz
│          ├── pancreas_tumor.nii.gz
│          ├── aorta.nii.gz
│          ├── gall_bladder.nii.gz
│          ├── kidney_left.nii.gz
│          ├── kidney_right.nii.gz
│          ├── liver.nii.gz
│          ├── pancreas.nii.gz
│          └──...
...
```
</div>
</details>


Compare two individual labels:
```bash
python CompareOrgan.py \
  --ct Dataset/BDMAP_A0000001/ct.nii.gz \
  --mask1_subdir Dataset/BDMAP_A0000001/predictions1 \
  --mask2_subdir Dataset/BDMAP_A0000001/predictions2 \
  --organ pancreas \
  --port 8000 \
  --log_file ./comparison_summary.log

```
Edit the input paths directly inside CompareOrgan.py before running.



# Citation

```
@misc{bassi2024labelcriticdesigndata,
      title={Label Critic: Design Data Before Models}, 
      author={Pedro R. A. S. Bassi and Qilong Wu and Wenxuan Li and Sergio Decherchi and Andrea Cavalli and Alan Yuille and Zongwei Zhou},
      year={2024},
      eprint={2411.02753},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.02753}, 
}
```

<p align="center">
  <img src="https://github.com/PedroRASB/Cerberus/blob/main/misc/LabelCriticLogos.png" alt="Project Logo" width="900"/>
</p>

