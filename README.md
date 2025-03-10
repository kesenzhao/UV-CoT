# UV-CoT


## Install

1. Clone this repository and navigate to RLAIF-V folder
```bash
git clone https://github.com/UV-CoT
cd UV-CoT
```

2. Install package
```bash
conda create -n uv-cot python=3.10 -y
conda activate uv-cot
pip install -e .
```
3. Install required spaCy model
```bash
wget https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.7.3/en_core_web_trf-3.7.3.tar.gz
pip install en_core_web_trf-3.7.3.tar.gz
```




## Data Generation
1. Environment Setup

We provide the OmniLMM 12B model for feedback generation. If you wish to use the MiniCPM-Llama3-V 2.5 for giving feedback, please configure its inference environment according to the instructions in the [MiniCPM-V GitHub repository](https://github.com/OpenBMB/MiniCPM-V).

Please download our fine-tuned Llama3 8B models: [split model](https://thunlp.oss-cn-qingdao.aliyuncs.com/rlaifv_llama3_split_model.tar.gz) and [question transformation model](https://thunlp.oss-cn-qingdao.aliyuncs.com/rlaifv_llama3_changeq_model.tar.gz), and store them in the `./models/llama3_split` folder and the `./models/llama3_changeq` folder respectively.

2. OmniLMM 12B Model Feedback

The following script demonstrates using the LLaVA-v1.5-7b model to generate candidate answers and the OmniLMM 12B model to provide feedback.

```bash
mkdir ./results
bash ./script/data_gen/run_data_pipeline_llava15_omni.sh
```




## Train

1. Prepare data

We also prepare the image file in this [link](https://huggingface.co/datasets/deepcs233/Visual-CoT/tree/main/cot_with_detailed_reasoning_steps), you need to merge these split archive files and then extract them.


- COCO: [train2017](http://images.cocodataset.org/zips/train2017.zip)

- GQA: [images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip)

- OCR-VQA: [download script](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing), **we save all files as `.jpg`**

- TextVQA: [train_val_images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip)

- VisualGenome: [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)

- Visual7W: [repo](https://github.com/yukezhu/visual7w-toolkit)

- Flickr30k: [homepage](https://shannon.cs.illinois.edu/DenotationGraph/)

- DocVQA: [homepage](https://www.docvqa.org/datasets/docvqa)

- InfographicsVQA: [homepage](https://www.docvqa.org/datasets/infographicvqa)

- VSR: [images](https://www.dropbox.com/s/0s3bj25s62crjh2/vsr_images.zip?dl=0)

- DUDE: [images](https://huggingface.co/datasets/jordyvl/DUDE_loader/blob/main/data/DUDE_train-val-test_binaries.tar.gz)

- SROIE: [homepage](https://rrc.cvc.uab.es/?ch=13&com=downloads)

  

After downloading all of them, organize the data as follows in `./playground/data`,

```
├── coco
│   └── train2017
│   └── train2014
├── gqa
│   └── images
├── ocr_vqa
│   └── images
├── textvqa
│   └── train_images
└── vg
│   ├── VG_100K
│   └── VG_100K_2
└── v7w
│   └── images
└── flickr30k
│   └── images
└── cot
│   └── flickr30k
│   └── docvqa
│   └── gqa
│   └── infographicsvqa
│   └── textvqa
│   └── vsr
│   └── dude
│   └── sroie
```

2. Training

Here, we provide a training script to train the model in **1 iteration**. The `max_step` parameter should be adjusted according to the amount of your data.

- **Fully Fine-tuning**
  

Run the following command to start fully fine-tuning.

```bash
bash ./script/train/llava15_train.sh
```



3. Iterative alignment

To reproduce the iterative training process in the paper, you need to do the following steps for 4 times:
- **S1. Data generation.**

  Follow the instructions in [data generation](https://github.com/RLHF-V/RLAIF-V?tab=readme-ov-file#data-generation) to generate preference pairs for the base model. Convert the generated jsonl file to huggingface parquet.
- **S2. Change training config.**

  In dataset code, replace `'openbmb/RLAIF-V-Dataset'` [here](muffin/data/datasets.py#L38) to your data path.

  In [training script](script/train/llava15_train.sh), replace `--data_dir` with a new directory, replace `--model_name_or_path` with the base model path, set `--max_step` to the number of steps for 4 epoch, set `--save_steps` to the number of steps for 1/4 epoch.
- **S3. Do DPO training.**

  Run the training script to train the base model.
- **S4. Choose base model for next iteration.**

  

## Evaluation

1. Single-GPU inference on both training datasets and zero-shot datasets, `UV-CoT` can be changed to other model names saved in the ./checkpoints/

```bash
bash scripts/v1_5/eval/cot_benchmark.sh UV-CoT
```

2. Obtain the score using GPT-4o, the API KEY need to be set in `llava/eval/eval_cot_score.py`

```bash
bash scripts/v1_5/eval/cot_benchmark_ablations.sh UV-CoT
```

3. Obtain the score using GPT-4o, the API KEY need to be set in `llava/eval/eval_cot_score.py`

```bash
bash scripts/v1_5/eval/cot_score.sh UV-CoT
```

