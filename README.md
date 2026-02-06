# MeKi
Homepage for paper "MeKi: Memory-based Expert Knowledge Injection for Efficient LLM Scaling" ([Arxiv](https://www.arxiv.org/pdf/2602.03359))

# Implementation
We provide a quick start to demonstrate the model logic and data flow of our method. The demo can be found at [meki_modeling_demo.py](meki_modeling_demo.py).
```
pip install torch easydict
python3 meki_modeling_demo.py
```

# Introduction
Scaling Large Language Models (LLMs) typically relies on increasing the number of parameters or test-time computations to boost performance. However, these strategies are impractical for edge device deployment due to limited RAM and NPU resources. Despite hardware constraints, deploying performant LLM on edge devices such as smartphone remains crucial for user experience. To address this, we propose MeKi (Memory-based Expert Knowledge Injection), a novel system that scales LLM capacity via storage space rather than FLOPs. MeKi equips each Transformer layer with token-level memory experts that injects pre-stored semantic knowledge into the generation process. To bridge the gap between training capacity and inference efficiency, we employ a re-parameterization strategy to fold parameter matrices used during training into a compact static lookup table. By offloading the knowledge to ROM, MeKi decouples model capacity from computational cost, introducing zero inference latency overhead. Extensive experiments demonstrate that MeKi significantly outperforms dense LLM baselines with identical inference speed, validating the effectiveness of memory-based scaling paradigm for on-device LLMs.
