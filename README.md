# NanoLlama
Home dummy project of transformer decoder models of various sizes trained on news articles on Google's free T4 GPU. The training process involved three main stages:

1. Pre-training on 1.87B tokens of cleaned data for two epochs. 
2. Supervised fine-tuning for one epoch on selected articles with length between 128 and 256 tokens. 
3. Direct Preference Optimization on generated articles that were judged by ChatGPT (However model was too small to be further fine-tuned so I only implemented training logic.)

Below are the pre-training parameters and loss for the 29M-parameter model. Training took approximately 30 hours in fp16 precision. The model was trained with roughly an order of magnitude of 10^17 FLOPs. Scaling laws from the Chinchilla paper do not apply for at least two reasons: 1) The dataset was only from one domain and very small compared to regular pre-training settings. 2) Instead of training for one epoch on all of the data, I trained the model for two epochs, which likely resulted in a lower loss than expected.

| Parameter         | Value         |
|:------------------|:--------------|
| model_size        | tiny          |
| vocab_size        | 20000         |
| max_num_steps     | 230000        |
| context_size      | 256           |
| tokens_per_batch  | 16384         |
| max_lr            | 6e-04         |
| min_lr            | 6e-05         |
| warmup_steps      | 2000          |
| weight_decay      | 0.1           |
| betas             | [0.9, 0.999]  |
| clip_grad         | 1.0           |
| grad_acc          | 1             |
| dropout           | 0.0           |
| dim               | 512           |
| num_heads         | 8             |
| num_layers        | 6             |
| batch_size        | 64            |


![trainlosss](./loss/trainloss.PNG)