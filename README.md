# NanoLlama
Home dummy project of transformer decoder models of various sizes trained on news articles on Google's free T4 GPU. The training process involved three main stages:

1. Pre-training on 1.87B tokens of cleaned data for two epochs. 
2. Supervised fine-tuning on one epoch on selected articles with length between 128 and 256 tokens. 
3. Direct Preference Optimization on generated articles that were judged by ChatGPT.

