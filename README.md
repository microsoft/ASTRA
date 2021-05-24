# Self-Training with Weak Supervision

This repo holds the code for our weak supervision framework, ASTRA, described in our NAACL 2021 paper: "[Self-Training with Weak Supervision](https://www.microsoft.com/en-us/research/publication/leaving-no-valuable-knowledge-behind-weak-supervision-with-self-training-and-domain-specific-rules/)" 


## Overview of ASTRA

ASTRA is a weak supervision framework for training deep neural networks by automatically generating weakly-labeled data. Our framework can be used for tasks where it is expensive to manually collect large-scale labeled training data. 

ASTRA leverages domain-specific **rules**, a large amount of **unlabeled data**, and a small amount of **labeled data**  through a **teacher-student** architecture:

![alt text](https://github.com/microsoft/ASTRA/blob/main/astra.jpg?raw=true)

Main components:
* **Weak Rules**: domain-specific rules, expressed as Python labeling functions. Weak supervision usually considers multiple rules that rely on heuristics (e.g., regular expressions) for annotating text instances with weak labels.
*  **Student**: a base model (e.g., a BERT-based classifier) that provides pseudo-labels as in standard self-training. In contrast to heuristic rules that cover a subset of the instances, the student can predict pseudo-labels for all instances.
* **RAN Teacher**: our Rule Attention Teacher Network that aggregates the predictions of multiple weak sources (rules and student) with instance-specific weights to compute a single pseudo-label for each instance. 

The following table reports classification results over 6 benchmark datasets averaged over multiple runs.

Method | TREC | SMS | YouTube | CENSUS | MIT-R | Spouse 
--- | --- | --- | --- |--- |--- |--- 
Majority Voting | 60.9 | 48.4 | 82.2 | 80.1 | 40.9 | 44.2
Snorkel | 65.3 | 94.7 | 93.5 | 79.1 | 75.6 | 49.2
Classic Self-training | 71.1 | 95.1 | 92.5 | 78.6 | 72.3 | 51.4
**ASTRA** | **80.3** | **95.3** | **95.3** | **83.1** | **76.1** | **62.3**

Our [NAACL'21 paper](https://www.microsoft.com/en-us/research/publication/leaving-no-valuable-knowledge-behind-weak-supervision-with-self-training-and-domain-specific-rules/) describes our ASTRA framework and more experimental results in detail. 

## Installation

First, create a conda environment running Python 3.6: 
```
conda create --name astra python=3.6
conda activate astra
```

Then, install the required dependencies:
```
pip install -r requirements.txt
```

## Download Data
For reproducibility, you can directly download our pre-processed data files (split into multiple unlabeled/train/dev sets): 

```
cd data
bash prepare_data.sh
```

The original datasets are available [here](https://github.com/awasthiabhijeet/Learning-From-Rules).


## Running ASTRA 


To replicate our NAACL '21 experiments, you can directly run our bash script:
```
cd scripts
bash run_experiments.sh
```
The above script will run ASTRA and report results under a new "experiments" folder. 

You can alternatively run ASTRA with custom arguments as: 
```
cd astra
python main.py --dataset <DATASET> --student_name <STUDENT> --teacher_name <TEACHER>
```

Supported STUDENT models: 
1. **logreg**: Bag-of-words Logistic Regression classifier
2. **elmo**: ELMO-based classifier
3. **bert**: BERT-based classifier

Supported TEACHER models: 
1. **ran**: our Rule Attention Network (RAN)

We will soon add instructions for supporting custom datasets as well as student and teacher components. 




## Citation 

```
@InProceedings{karamanolakis2021self-training,
author = {Karamanolakis, Giannis and Mukherjee, Subhabrata (Subho) and Zheng, Guoqing and Awadallah, Ahmed H.},
title = {Self-training with Weak Supervision},
booktitle = {NAACL 2021},
year = {2021},
month = {May},
publisher = {NAACL 2021},
url = {https://www.microsoft.com/en-us/research/publication/self-training-weak-supervision-astra/},
}
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
