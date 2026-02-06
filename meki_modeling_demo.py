"""
================================================================================
[MeKi Architecture Demo]

This implementation is a demo version to show the model logic and data flow of the proposed MeKi.

This implementation requires further optimizations for practical production purpose. 

`pip install easydict` before running.
================================================================================
"""

import torch
from torch import nn
from easydict import EasyDict
F = nn.functional


class MeKi(nn.Module):
    def __init__( self, config ):
        super().__init__()

        self.mem_dim = config.mem_dim
        self.vocab_size = config.vocab_size

        self.embeddings = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.mem_dim,
        )
        self.gate_proj = nn.Linear(
            config.hidden_size,
            config.mem_dim,
            bias=False
        )
        self.out_proj = nn.Linear(
            config.mem_dim,
            config.hidden_size,
            bias=False
        )
        self.beta_scale = nn.Parameter(torch.tensor(1.))
        self.alpha_scale = nn.Parameter(torch.tensor(1.))
        
        kwargs = {
            "hidden_size": config.hidden_size,
            "intermediate_size": config.hidden_size // 2,
            "output_size": config.mem_dim
        }
        self.word_emb_projection = MeKiMLP(config, kwargs)

        self.mix_norm = nn.RMSNorm(config.mem_dim, eps=config.rms_norm_eps)
        self.post_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.flag_reparam = False

    def forward(self, hidden_states, input_ids, all_word_emb: nn.Embedding):
        if self.training:
            word_emb = all_word_emb(input_ids)
            return self.train_forward(hidden_states, input_ids, word_emb)
        
        else:
            if not self.flag_reparam:
                assert all_word_emb.weight.shape[0] == self.vocab_size
                self.reparameterization(all_word_emb.weight)
                self.flag_reparam = True
                print("Re-param done.")

            return self.eval_forward(hidden_states, input_ids)

    def train_forward(self, hidden_states, input_ids, word_emb):
        print("This is train fwd")
        static_embedding = self.embeddings(input_ids)
        dynamic_embedding = self.word_emb_projection(word_emb)
        meki_embedding = self.mix_norm(
            static_embedding + dynamic_embedding * self.beta_scale
        )  * self.alpha_scale

        hidden_states = self.gate_proj(hidden_states)
        hidden_states = F.sigmoid(hidden_states) + meki_embedding
        hidden_states = self.out_proj(hidden_states)

        return self.post_norm(hidden_states)

    @torch.inference_mode()
    def reparameterization(self, all_word_emb):
        static_embedding = self.embeddings.weight
        dynamic_embedding = self.word_emb_projection(all_word_emb)

        meki_embedding = self.mix_norm(
            static_embedding + dynamic_embedding * self.beta_scale
        )  * self.alpha_scale

        self.embeddings.weight.data.copy_(meki_embedding)

    @torch.inference_mode()
    def eval_forward(self, hidden_states, input_ids):
        print("This is eval fwd")
        meki_embedding = self.embeddings(input_ids)

        hidden_states = self.gate_proj(hidden_states)
        hidden_states = F.sigmoid(hidden_states) + meki_embedding
        hidden_states = self.out_proj(hidden_states)

        return self.post_norm(hidden_states)


class MeKiMLP(nn.Module):
    def __init__(self, config, kwargs={}):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.act_fn = nn.SiLU()

        if len(kwargs.keys()):
            self.gate_proj = nn.Linear(kwargs["hidden_size"], kwargs["intermediate_size"], bias=False)
            self.up_proj = nn.Linear(kwargs["hidden_size"], kwargs["intermediate_size"], bias=False)
            self.down_proj = nn.Linear(kwargs["intermediate_size"], kwargs["output_size"], bias=False)
        else:
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        activations = self.act_fn(self.gate_proj(hidden_states))
        up_proj = self.up_proj(hidden_states)
        down_proj = self.down_proj(activations * up_proj)
        return down_proj


class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pre_attn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = lambda x:x
        self.pre_ffn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn  = MeKiMLP(config)
        self.meki_module = MeKi(config)
    
    def forward(self, hidden_states, input_ids, all_word_emb):
        hidden_states = self.pre_attn_norm(self.attn(hidden_states)) + hidden_states

        residual = hidden_states
        hidden_states = self.pre_ffn_norm(hidden_states)
        meki_output = self.meki_module(hidden_states, input_ids, all_word_emb)
        hidden_states = self.ffn(hidden_states) + meki_output + residual
        return hidden_states


class LLM(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.word_emb = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [TransformerLayer(config) for layer_id in range(config.num_layers)]
        )
        self.head = nn.Sequential(
            nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps),
            nn.Linear(config.hidden_size, config.vocab_size)
        )

    def forward(self, input_ids):
        hidden_states = self.word_emb(input_ids)

        for _, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, input_ids, self.word_emb)

        output = self.head(hidden_states)
        return output


if __name__ == '__main__':

    config = EasyDict({
        "vocab_size": 150000,
        "mem_dim": 256,
        "hidden_size": 2048,
        "intermediate_size": 4096,
        "num_layers": 3,
        "rms_norm_eps": 1e-6
    })

    bsz, seq_len = 2, 4096
    input_ids = torch.randint(low=0, high=config.vocab_size, size=(bsz, seq_len))

    llm = LLM(config)

    llm.train()
    logits = llm(input_ids)
    print("Train: logits.shape =", logits.shape)

    llm.eval()
    
    bsz, seq_len = 2, 4096
    input_ids = torch.randint(low=0, high=config.vocab_size, size=(bsz, seq_len))
    logits = llm(input_ids)
    print("Eval: logits.shape =", logits.shape)

    bsz, seq_len = 8, 1024
    input_ids = torch.randint(low=0, high=config.vocab_size, size=(bsz, seq_len))
    logits = llm(input_ids)
    print("Eval: logits.shape =", logits.shape)
