<p align="center">
  <img src="https://github.com/PedroRASB/Cerberus/blob/main/misc/LabelCriticTitle.png" alt="Project Logo" width="900"/>
</p>

<p align="center">
  <img src="https://github.com/PedroRASB/Cerberus/blob/main/misc/LabelCriticModel.png" alt="Project Logo" width="900"/>
</p>

AI-generated annotations are increasingly used to build large datasets,
yet their quality is often assumed rather than verified.

Label Critic is an automated tool for reviewing AI-generated labels.
It helps users select better annotations when multiple label options exist,
and identify potentially incorrect labels when only a single annotation is available.

Label Critic uses pre-trained Large Vision-Language Models (LVLMs)
as label critics, comparing or assessing annotations without training new models.
In medical CT organ segmentation, it achieves 96.5% accuracy
in selecting higher-quality annotations per scan and class.

While this repository focuses on CT organ segmentation
(pancreas, liver, spleen, kidneys, aorta, and others),
the approach is data-centric and can be adapted to new classes with minimal effort.


## Paper

<b>Label Critic: Design Data Before Models</b> <br/>
Pedro R. A. S. Bassi, Qilong Wu, Wenxuan Li, Sergio Decherchi, Andrea Cavalli, Alan Yuille, Zongwei Zhou <br/>
International Symposium on Biomedical Imaging (ISBI, 2025) <br/>
[Read More](https://arxiv.org/abs/2411.02753) <br/>

<br/>

[📄 View the ISBI Poster](https://github.com/PedroRASB/LabelCritic/raw/main/misc/ISBI_Label_critic_poster.pdf)

[![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://youtu.be/9D5-pFgtgDQ)


## How Label Critic Works

Label Critic evaluates the quality of AI-generated annotations by:
1. Projecting 3D images and labels into informative 2D views
2. Using a pre-trained vision-language model to compare or assess annotations
3. Selecting better candidates or flagging potentially incorrect labels

No additional model training is required.


## Code

This repository provides an end-to-end pipeline for running Label Critic,
including dataset projection, LVLM-based comparison, and error detection.

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



### Deploy LLM API

Deploy API locally (tensor-parallel-size should be the number of GPUs, and it accepts only powers of 2).
```bash
export NCCL_P2P_DISABLE=1
TRANSFORMERS_CACHE=./HFCache HF_HOME=./HFCache CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve "Qwen/Qwen2-VL-72B-Instruct-AWQ" --dtype=half --tensor-parallel-size 4 --limit-mm-per-prompt image=3 --gpu_memory_utilization 0.9 --port 8000
```


### Label Critic: dataset projection
This code creates 2D projections of a CT dataset and its labels. The command is designed to project two datasets, which represents two set of labels you would like to compare. Both datasets should be in the same format and have matching folder and label names. You can compare your dataset labels (/path/to/Dataset1/) to alternative labels produced by a public AI model (/path/to/Dataset2/). For organ segmentation on CT, you can find many state-of-the-art public AI models in the [Touchstone Benchmark](https://github.com/mrgiovanni/touchstone)


<details>
<summary style="margin-left: 25px;">Dataset format: format your datasets with this structure.</summary>
<div style="margin-left: 25px;">

```
Dataset
├── BDMAP_A0000001
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


```bash
python3 ProjectDatasetFlex.py --good_folder /path/to/Dataset1/ --bad_folder /path/to/Dataset2/ --output_dir1 /path/to/projections/directory/ --num_processes 10
```

### Label Critic: Use LVLM for label comparisons
This command uses the LVLM to compare the two sets of labels, using the projections saved in the command above. See the end of the comparisons.log file for a detailed log of the result of each comparison.

```bash
python3 RunAPI.py --path /path/to/projections/directory/ > comparisons.log 2>&1
```

or, to compare two individual labels instead of full label sets:
```bash
python3 compare_organ.py
```
Edit the input paths directly inside compare_organ.py before running.

### Label Critic: Error Detection

In case you do not have two sets of labels to compare, Label Critic can be used to evaluate a single set of labels, and judge if each one is correct or not. The --examples argument controls the number of examples of good and bad labels given to the LVLM (in-context learning). To use examples, you may check the labels and select a few good and bad examples, and place them in the folders /path/to/good/label/examples/ and /path/to/bad/label/examples/. After running the command, check the log file for a detailed output.

```bash
python3 ProjectDatasetFlex.py --good_folder /path/to/Dataset1/ --bad_folder /path/to/Dataset1/ --output_dir1 /path/to/projections/directory/ --num_processes 10
python3 RunErrorDetection.py --path /path/to/projections/directory/ --port 8000 --organ [kidneys] --file_structure auto --examples 0 --good_examples_pth /path/to/good/label/examples/ --bad_examples_pth /path/to/bad/label/examples/ > organ.log 2>&1
```

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

