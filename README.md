# Sentiment lexicons for suicide risk assessment

This repository contains files related to a study of 6 sentiment lexicons used for suicide risk assessment. This includes all code used to extract sentiment words from the eHOST-IT case-control cohort of CRIS clinical notes. The 6 lexicons (not provided) must be downloaded separately. These are:
* [AFINN](https://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=6010)
* [Linguistic Inquiry and Word Count (LIWC) 2015 lexicon](http://liwc.wpengine.com/)
* [NRC Word-Emotion Association Lexicon (aka EmoLex)](https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm)
* [Pattern lexicon](https://www.clips.uantwerpen.be/pages/pattern-en)
* [Opinion lexicon](https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html#lexicon)
* [SentiWordNet 3.0](https://github.com/aesuli/sentiwordnet)

The scripts are as follows:
* **emotions.py:** code to prepare data and lexicons for experiments and extract sentiment words.
* **emotions_afinn.py:** code to extract sentiment words using AFINN.
* **emotions_emolex.py:** code to extract sentiment words using EmoLex.
* **emotions_pattern.py:** code (Python 2.7) to extract words using the Pattern lexicon.
* **emotions_pattern_p36.py:** code (Python 3.6) to extract sentiment words from previously tokenised text using Pattern.
* **emotions_swn.py:** code to extract sentiment words using the NLTK interface for SentiWordNet 3.0.
* **sentiment_extraction.py:** code to calculate frequency statistics and test cross-corpus statistical significance (Mann-Whitney U Test) of frequency differences.
