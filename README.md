<h1 align="center"> RE </h1>
<h3 align="center"> Exploration via Reasoning Estimator </h3>

<p align="center">
  <a href="xxx">📄arXiv</a> •
  <a href="xxx/">🌐Web</a> •
    <a href="xxx">𝕏 Blog</a>
    •
    <a href="xxx">🤗 HF</a> •
    <a href="xxx">🎧NotebookLM Audio</a>


  
</p>

[![Awesome](https://awesome.re/badge.svg)](https://github.com/HuzhouNLP/Exploration-via-Reasoning-Estimator) 
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
![](https://img.shields.io/github/last-commit/HuzhouNLP/Exploration-via-Reasoning-Estimator?color=green) 

## Table of Contents

- 🌻[Acknowledgement](#acknowledgement)
- 🌟[Overview](#overview)
- 🔧[Installation](#installation)
- 📚[Logical Element Extraction and Score Calculation](#logical-element-extraction-and-Score-Calculation)
- 📉[Model Training](#model-training)
- 🧐[Evaluation](#evaluation)
- 🚩[Citation](#citation)

---



## 🌻Acknowledgement

Our code for the training module and the inference module is implemented based on [TRL](https://github.com/huggingface/trl). The training and test datasets are sourced from [Reclor](https://github.com/yuweihao/reclor), [LogiQA](https://github.com/lgw863/LogiQA-dataset), [LogiQA2.0](https://github.com/csitfun/LogiQA2.0), and [LogicBench](https://github.com/Mihir3009/LogicBench). Thanks for their great contributions! 


![alt text](Framework.png)

## 🌟Overview

Optimizing the cognitive abilities of models from the perspective of training data has become a promising approach to improving the performance of large language models (LLMs) in complex reasoning tasks. Recently, a series of studies have attempted to utilize data engineering methods to reconstruct the cognitive representation of training data, aiming to enhance data expressiveness through knowledge distillation. However, the existing research faces two major challenges: (1) a lack of quantitative indicators to assess the complexity of data and (2) resource mismatch caused by the absence of a complexity-aware mechanism. Inspired by this, in this study, we introduce a quantitative theoretical framework based on data cognitive efficiency and design a training paradigm based on dynamic cognitive assessment to address these challenges. To solve the problem of insufficient quantification, we first define the elements of data complexity, extract and quantify them into complexity scores, making this practical quantitative method applicable to various real-world reasoning tasks. To solve the problem of resource mismatch, we propose a new complexity-aware cognitive learning method, adopting differentiated weights during model training to achieve adaptive matching of training resources. Through extensive experiments on 4 models and 4 tasks, this study validates the effectiveness and rationality of the proposed framework and paradigm. Additionally, the study explores the evaluative role of complexity scores when LLMs simulate human cognition to answer questions. We hope that this work can help the academic community gain a comprehensive understanding of data cognitive efficiency and provide strategic references for optimizing the reasoning capabilities of LLMs.


## 🔧Installation

```bash
git https://github.com/HuzhouNLP/Exploration-via-Reasoning-Estimator
cd Exploration-via-Reasoning-Estimator
pip install -r requirements.txt
```

TRL install

```bash
pip install trl
```

## 📚Logical Element Extraction and Score Calculation

To process data
```sh
python ./score_calculate/data_process/1.logiQA.py
python ./score_calculate/data_process/2.logiQA2.0(MCQA).py
python ./score_calculate/data_process/3.reclor.py
python ./score_calculate/data_process/4.logicBench(Aug).py
python ./score_calculate/data_process/4.logicBench(Eval).py
```

To extract the logical element
```sh
python ./score_calculate/context_extract.py
python ./score_calculate/options_extract.py
```

To calculate the score
```sh
python ./score_calculate/final_score_all_data_calculate_LayerNorm_Sigmoid
python ./score_calculate/score_all.py
python ./score_calculate/extraction_final_chart.py
```
这里还没改!!!!!!!!!!!!!!!

Our training data has been uploaded to [huggingface](https://huggingface.co/datasets/zjunlp/WKM-train-data).

If you wish to extract data based on the score ratio
```sh
python ./score_calculate/filter_test_data_every_interval.py
```

If you want to calculate the error rate
```sh
python ./score_calculate/interveal_error_data.py
```

## 📉Model Training

Using TRL framework to train model
```sh
# If you want to train directly
python ./Our_way/train/direct_trainingDirect_training.py

# If you want to use curriculum learning method for training
python ./Our_way/train/curriculum_learning.py

# If you want to use the bin-based progressive learning method for training
python ./Our_way/train/bin-based_progressive_learning.py

# If you want to use our method for training
python ./Our_way/train/our_way.py
```

## 🧐Evaluation


To evaluate the task, you need to place your model and lora parameters in a designated location, which can be pointed to by adjusting the position in the file
```sh
# If you only have one original test set file to evaluate
python ./Our_way/eval/our_way.py

# If you only have one balanced test set file to evaluate
python ./Our_way/eval/Balanced_test.py

# If you have multiple original test set files that need to be evaluated
python ./Our_way/eval/TRL-submit_all.sh

# If you have multiple balanced test set files that need to be evaluated
python ./Our_way/eval/TRL-sig-submit_all.sh
```

After obtaining the results, in order to calculate the accuracy
```sh
python ./Our_way/eval/comprehensive_processing_accuracy_compute.py

# You can also create it as an Excel spreadsheet
python ./Our_way/eval/json_to_excel.py
```

To calculate the error rate
```sh
# If you only have one file
python ./Our_way/eval/error_rate_calculation_one.py

# If you have multiple files
python ./Our_way/eval/error_rate_calculation_multi.py
```

## 🚩Citation

Please cite our repository if you use WKM in your work. Thanks!

```bibtex

```



## 🎉Contributors



We will offer long-term maintenance to fix bugs and solve issues. So if you have any problems, please put issues to us.
