# Regulations Analysis

This repository is a trail of analyzing a regulation framework for families/groups of regulations inside the framework using graph analysis and NLP.

## Sections

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

## Presentations

[Short Presentation](https://docs.google.com/presentation/d/1lkssY2LE_9qsnAbTPCxjyROY9zEah3oH9rK0dg_Kxfo/edit?usp=sharing) - a short form presentation made to show the work done in this project

[Long Presentation](https://docs.google.com/presentation/d/1Kak5krK2rAR6CS-FmMnTM3k_9eCGJE30MAZkxlJEhyA/edit?usp=sharing) - a long form presentation made to show the work done in this project

## Publications

[*Analysis of the NIS800–53 regulations, Are the NIS families real?*](https://medium.com/@lior0110/analysis-of-the-nis800-53-regulations-are-the-nis-families-real-03148755da3a?source=friends_link&sk=e6b351d9df599508763b44bf6ccea677) - a medium article that covers the first stage of the work from [**NIS800-53 analysis**](https://github.com/lior0110/regulations_analysis/blob/main/NIS800-53%20analysis.ipynb) of evaluating the **NIS800–53 families** based on the **Related Controls** they give themself.

[*Human-Independent analysis of the NIS800–53 regulations*](https://medium.com/@lior0110/human-independent-analysis-of-the-nis800-53-regulations-3db09ed0df9c?source=friends_link&sk=e67925c08e48286a676d6ef643be40b3) - a medium article on the second stage of the work covering the automation pipeline and the making of the NLP discovered **Related Controls**.
