# Silent Vulnerable Dependency Alert Prediction with Vulnerability Key Aspect Explanation

This repository shows the labels and codes of paper **Silent Vulnerable Dependency Alert Prediction with Vulnerability Key Aspect Explanation**.
We only show part of training data here as the full data is too big, please contact [u5871153@anu.edu.au](mailto:u5871153@anu.edu.au) to get the full data.

## Contents

| File | Description |
|-|-|
| [code/clasifier](https://github.com/anonymous-dev904/aspect_generation/tree/main/code/classifier) | Codes for classifier of fusion model with cross model self-attention layer |
| [code/generator](https://github.com/anonymous-dev904/aspect_generation/tree/main/code/generator) | Codes for generator of fusion model with cross model self-attention layer |
| [data/clasifier](https://github.com/anonymous-dev904/aspect_generation/tree/main/data/classifier) | Labels for training classifier |
| [data/generator](https://github.com/anonymous-dev904/aspect_generation/tree/main/data/generator) | Labels for training generator |


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


