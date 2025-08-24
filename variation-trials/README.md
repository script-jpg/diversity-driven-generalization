# This folder contains some data for variations I did on the metrics
Here is a summary of the results:
1. Using the entire dataset for the embedding isn't significantly more informative but is massively more costly to do. See for the first 3 embeddings. Took almost 1.5 hours on a A100 GPU
2. Using the embedding model's tokenizer keeps the embeddings the same size but loses a lot of information that we would have from the model's own tokenizer (despite different embedding sizes). This can be seen in the updated fit_model_ssr.ipynb file in this dir where the edc variation is called edc_fixed size
