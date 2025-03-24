# Project Dicoding 1 | Sentiment Analysis

Name : Muhammad Hafizh Dzaki

Email : 111202113370@mhs.dinus.ac.id

### Introduction

Predict an emotion / nuance from a poem. Poem data was scraped from [this link](https://www.poemhunter.com).

### Environment Used

- Windows Subsystem Linux 2 (WSL2) environment
- GPU CUDA Acceleration Enabled (When labelling dataset and training model. Recommended)
- Visual Studio Code that connects to WSL2

### Scenario that Was Done

Base variation used in this project:

1. Labelling Method : HuggingFace model from [J-Hartmann](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base) [[1](#hartmann)] and [Bhadresh-Savani](https://huggingface.co/bhadresh-savani/distilbert-base-uncased-emotion) [[2](#salvani)]

2. Train-Test Ratio : 80:20 and 90:10

3. Model Architecture : LSTM and GRU

Combination three points above results **Eight Different Scenarios**

### Understanding Folder 

1. dataset : Folder contains Dataset, mainly for training model
    - labelled_poem_xxx.csv : Dataset that has been labelled using respective HuggingFace model
    - poem_dataset.csv : Dataset consist only poem data, no label yet (created after merging files inside poem folder)
2. model : Folder that stores best model from every scenario used
    - Format : best_model_{labelling-method}\_{test-ratio}_{model}.keras
    - Example : best_model_hartmann_0.1_gru.keras
        - hartmann : HuggingFace model [emotion-english-distilroberta-base](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)
        - 0.1 : Train-Test ratio is 90:10
        - gru : Gated Recurrent Unit arhitecture was used
3. poem : Folder that stores raw poem data after being scraped
4. tokenizer : Folder contains tokenizer model for each scenario
    - Format : tokenizer_{labelling-method}\_{test-ratio}_{model}.pkl (self explanatory)

### Understanding Files

1. scraping.ipynb : Contains code for scraping poem data from [this website](https://www.poemhunter.com) 
2. merge_dataset.ipynb : Contains code for merging 150 parts file of poem data inside folder *poem* into one file named `poem_dataset.csv` inside folder *dataset*
3. labelling.ipynb : Contains code for labelling `poem_dataset.csv` using 2 different HuggingFace models
4. modelling.ipynb : Contains main code for cleaning, preprocessing, modelling, and testing
5. inference.ipynb : Contains code for doing inference the saved model
6. chromedriver.exe : File that helps for scraping poem data from desired website
7. custom_function.py : Contains main function used in `modelling.ipynb`
8. custom_vocab.txt : Contains string that not necessary. Helps cleaning process better
9. poem_link.json : Contains base poem link. Mainly used in `scraping.ipynb`
10. requirements.txt : Self explanatory

### References

<div id='hartmann'>[1] Jochen Hartmann, "Emotion English DistilRoBERTa-base". https://huggingface.co/j-hartmann/emotion-english-distilroberta-base/, 2022.
</div>

<br>

<div id='salvani'>[2] Bhadresh Savani, "Distilbert-base-Uncased-Emotion". https://huggingface.co/bhadresh-savani/distilbert-base-uncased-emotion/, 2022
</div>