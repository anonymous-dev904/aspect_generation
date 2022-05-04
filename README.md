# Silent Vulnerable Dependency Alert Prediction with Vulnerability Key Aspect Explanation

This repository shows the labels and codes of paper **Silent Vulnerable Dependency Alert Prediction with Vulnerability Key Aspect Explanation**.
We only show part of training data here as the full data is too big, please contact authors to get the full data.

## Contents

| File | Description |
|-|-|
| [code/clasifier](https://github.com/anonymous-dev904/aspect_generation/tree/main/code/classifier) | Codes for classifier of BERT-CodeBERT model with cross model self-attention layer |
| [code/generator](https://github.com/anonymous-dev904/aspect_generation/tree/main/code/generator) | Codes for generator of BART-CodeBERT model with cross model self-attention layer |
| [data/clasifier](https://github.com/anonymous-dev904/aspect_generation/tree/main/data/classifier) | Labels for training classifier |
| [data/generator](https://github.com/anonymous-dev904/aspect_generation/tree/main/data/generator) | Labels for training generator |


## BART-CodeBERT Model Structure
<img src="https://github.com/anonymous-dev904/aspect_generation/blob/main/images/network2.png" width=50% height=50%>

We combine the pre-trained BART model with pre-trained CodeBERT model to test the capability of model containing both natural language and code information. One drawback of the pre-trained Transformer model is that the dimensions of its input and parameters are fixed, so we cannot simply concatenate the two encoders’ outputs, which is incompatible with the dimensions of the pre-trained decoder’s input. To solve the problem, we use cross-model self-attention layer, using query states (Q) of BART encoder with key (K) and value states (V) of CodeBERT encoder to calculate importance of each CodeBERT output token to the BART encoder’s output tokens. The input of BART decoder is output of residual structure between BART encoder and cross-model self-attention layer.

## Note
The code is based on [CodeBERT](https://github.com/microsoft/CodeBERT) project. The path of all files need to be changed.

##  Quality of Silent Dependency Alert Classification 
According to paper Section 4.1, the classification results are shown below:

###  Results of Silent Dependency Alert Classification by Different Models
||CodeBERT|BERT|BERT-CodeBert|Transformer|LSTM|
|:---:|:---:|:---:|:---:|:---:|:---:|
|AUC |**0.91**|0.89|0.57|0.80|0.71|

CodeBERT has the best result (0.91 AUC), followed by BERT (0.89 AUC). Both of them have much better performance than non-pre-trained models (Transformer and LSTM). 

###  Results of Silent Dependency Alert Classification with Different Types of Inputs
||Commit Message|Added & Deleted Code Segments|All Code Segments|Commit Message & Added & Deleted Code Segments|Commit Message & All Code Segments|
|:---:|:---:|:---:|:---:|:---:|:---:|
|AUC |0.55|0.67|0.62|0.80|**0.91**|

The input results with both commit messages and code segments are better than the results without message or code segment (0.80-0.91 AUC versus 0.55-0.67 AUC), indicating that both commit messages and code segments are useful for the silent dependency alert detection.

##  Quality of Explainable Silent Dependency Alert Generation
According to paper Section 4.2, the generation results are shown below:

###  Results of Different Models of Aspect Generator
![alt text](https://github.com/anonymous-dev904/aspect_generation/blob/main/images/generator_model.png "Results of Different Models of Aspect Generator")

The results of pre-trained model based models (BART, CodeBERT and BART-CodeBERT) are much better than non-pre-trained Transformer model and the LSTM in all four key aspects. This demonstrates the advantages of model pre-training.

###  Results of CodeBERT-based Aspect Generator for Different Types of Inputs
![alt text](https://github.com/anonymous-dev904/aspect_generation/blob/main/images/generator_inputs.png "Results of CodeBERT-based Aspect Generator for Different Types of Inputs")

Inputs with both commit messages and code contents have much better results than the ones using only commit message or code contents. This indicates both commit messages and code contents are important to the generation task.
