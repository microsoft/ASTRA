# Self-Training with Weak Supervision

This repo holds the code for our weak supervision framework, ASTRA, described in our NAACL 2021 paper: "[Self-Training with Weak Supervision](https://www.microsoft.com/en-us/research/publication/leaving-no-valuable-knowledge-behind-weak-supervision-with-self-training-and-domain-specific-rules/)"


## Installation

Install Requirements (Python 3.6):
```
pip install -r requirements.txt
```

## Download Data

We will soon add detailed instructions for downloading datasets and domain-specific rules as well as supporting custom datasets. 


## ASTRA Framework

ASTRA leverages domain-specific rules, a large amount of unlabeled data, and a small amount of labeled data via iterative self-training.

You can run ASTRA as: 
```
cd astra
python main.py --dataset <DATASET> --student_name <STUDENT_MODEL> --teacher_name <TEACHER_MODEL>
```

Supported <STUDENT_MODEL> arguments: 
1. **logreg**: Bag-of-words Logistic Regression classifier
2. **elmo**: ELMO-based classifier
3. **berttf**: BERT-based classifier

Supported <TEACHER_MODEL> arguments: 
1. **ran**: our Rule Attention Network (RAN)

We will soon add instructions for supporting custom student and teacher components. 

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
