# Experimental Instructions: Small Models + DeepSeek Integration

## Overview
This experiment evaluates the performance of different small models (SVM, AdaBoost, Bagging, BERT, CodeBERT) integrated with DeepSeek.
- **Data Split**: 80% training set, 20% validation set
- **Dataset**: TacticData

## Environment Requirements
- **Python Version**: >= 3.8
- **Key Dependencies**:
  - scikit-learn
  - transformers
  - torch
  - pandas
  - numpy

Install dependencies:

```bash
pip install -r requirements.txt
```

## Experimental Procedure

### 1. Configure API Key
Replace the placeholder with your DeepSeek API Key in `call_deepseek.py`:

### 2. Prepare Data
Ensure the datasets are located at the following paths:
- **Training Set**: `data/TacticData_train.tsv`
- **Validation Set**: `data/TacticData_val.tsv`

### 3. Run Experiment

```bash
python main.py
```

## Experimental Results

Result files location:
- **SVM**: `llm_result/SVM_deepseek/TacticData/results.csv`
- **AdaBoost**: `llm_result/AdaBoost_deepseek/TacticData/results.csv`
- **Bagging**: `llm_result/Bagging_deepseek/TacticData/results.csv`
- **BERT**: `llm_result/BERT_deepseek/TacticData/results.csv`
- **CodeBERT**: `llm_result/CodeBERT_deepseek/TacticData/results.csv`