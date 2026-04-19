## Supplementary Material
This supplementary material for the paper "Enhancing the Accuracy and Comprehensibility in Architectural Tactics Detection via Small Model-Augmented Prompt Engineering" is organized as follows:

### Project Structure

```text
Prmt4TD/
├── TacticData dataset.zip    # Dataset
├── Hadoop case.zip    #Case
├── Experiments/
│   ├── RQ1/    
│   ├── RQ2/    
│   ├── RQ3/    
│   ├── RQ4/    
├── Additional experiments/
│   ├── Result tables/    # Data files
│   ├── Prmt4TD_Additional experiments.pdf    # Experimental results
└── README.md
```

> - <mark>TacticData dataset.zip</mark>: The training dataset used by Prmt4TD includes preprocessed data and raw Java files.
> - <mark>Hadoop case.zip</mark>: The test dataset used by Prmt4TD includes preprocessed data and raw Java files.

> **Note:** We obtained the Hadoop case data from the papers "Keim J, Kaplan A, Koziolek A, et al. Does BERT understand code?–An exploratory study on the detection of architectural tactics in code[C]//European Conference on Software Architecture. Cham: Springer International Publishing, 2020: 220-228." and "Mirakhorli M, Cleland-Huang J. Detecting, tracing, and monitoring architectural tactics in code[J]. IEEE Transactions on Software Engineering, 2016, 42(3): 205-220.", but they did not provide the raw Java files. All the raw code files in the Hadoop case were collected by us based on the file paths.
