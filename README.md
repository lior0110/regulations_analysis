# Regulations Analysis

* [**NIS800-53 analysis**](https://github.com/lior0110/regulations_analysis/blob/main/NIS800-53%20analysis.ipynb) - notebook for analysis of the [NIS800-53 regulations](https://csrc.nist.gov/pubs/sp/800/53/r5/upd1/final) to see the connections between the regulations
    * This notebook is a trail to analyze the NIS800-53 families
    * This notebook mainly uses graph analysis approaches
* [**NIS800-53 analysis NLP**](https://github.com/lior0110/regulations_analysis/blob/main/NIS800-53%20analysis%20NLP.ipynb) - notebook for finding new relationships between [NIS800-53 regulations](https://csrc.nist.gov/pubs/sp/800/53/r5/upd1/final) in a human-independent way
    * This notebook is using three different NLP algorithms BM25, Sentence Transformers Embedding and SaaS NLP embedding
    * This notebook is the first step in the human-independent pipeline
* [**helping_functions**](https://github.com/lior0110/regulations_analysis/blob/main/helping_functions.py) - a Python file that contains the functions of the human-independent pipeline
    * contains all the Python functions needed for the analysis pipeline, similar to the manual pipeline from [NIS800-53 analysis](https://github.com/lior0110/regulations_analysis/blob/main/NIS800-53%20analysis.ipynb), just more automatic and human-independent
* [**NIS800-53 analysis 2**](https://github.com/lior0110/regulations_analysis/blob/main/NIS800-53%20analysis%202.ipynb) - notebook for running the second stage of the human-independent analysis pipeline
    * uses the output from [NIS800-53 analysis NLP](https://github.com/lior0110/regulations_analysis/blob/main/NIS800-53%20analysis%20NLP.ipynb) and the functions from [helping_functions](https://github.com/lior0110/regulations_analysis/blob/main/helping_functions.py)
    * makes the second stage of the automated human-independent analysis pipeline for the analysis of the regulations