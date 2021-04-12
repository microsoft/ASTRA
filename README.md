# Self-Training with Weak Supervision

This repo holds the code for our NAACL 2021 paper: "[Self-Training with Weak Supervision](https://www.microsoft.com/en-us/research/publication/leaving-no-valuable-knowledge-behind-weak-supervision-with-self-training-and-domain-specific-rules/)"

# Installation

## Install Requirements (in a Python 3.6 environment)
```
pip install -r requirements.txt
```

## OR Install conda environment 
``` 
conda env create -f env/environment.yml
```

## Download Data
```
cd data
bash download_data.sh
```

# ASTRA Framework

## Run ASTRA
```
cd astra
python main.py --dataset youtube --datapath ../data 
```

Supported classifiers: logreg (Logistic Regression), berttf (Bert-based classifier), elmo (ELMO-based classifier)
Supported datasets (lowercase): TREC, SMS, YOUTUBE, CENSUS, MITR, SPOUSE

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
