Title: Coding an LLM Architecture
Date: 2024-09-01
Category: LLMs
Tags: LLMs From Scratch, Architecture
Status: Draft


This blog post is a Part II of building an LLM from scratch. This is yet again inspired by a coding workshop - Building LLMs from the Ground Up: A 3-hour Coding Workshop by Sebastian Raschka. 

As in the below diagram of contructing an LLM ground up, let's assume that we've a black box 'Attention Mechanism' ready with us, and we are jumping straight ahead to Part 3, LLM Architecture. I'd revisit Part 2 Attention Mechanism in a different blog post. This way of learning helps me, as I get to understand the high level architecture before diving deep into any of the phases! We will begin with a top-down view of the model architecture in the next section before covering the individual components in more detail.

![LLM-Pipeline](images/llm-architecture/pipeline.png)

[TOC]

### Coding a GPT-like large language model (LLM) that can be trained to generate human-like text
LLMs, such as GPT (which stands for Generative Pretrained Transformer), are large deep
neural network architectures designed to generate new text one word (or token) at a time.

![LLM-Pipeline](images/llm-architecture/raw-llm-archi.png)

The configuration of the small GPT-2 model via the following Python dictionary,
which we will use in the code examples later.


```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.0,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}
```

### Normalizing layer activations to stabilize neural network training

### Adding shortcut connections in deep neural networks to train models more effectively

### Implementing transformer blocks to create GPT models of various sizes

### Computing the number of parameters and storage requirements of GPT models