from torch.utils.data import Dataset, DataLoader
import numpy as np
import tiktoken
import torch

"""  
chapter 2
"""
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        """ 
        max_length: 最大长度
        stride: 滑动窗口步长 
        """
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []

        # 对全部文本进行分词
        token_ids = tokenizer.encode(txt, allowed_special={'<|endoftext|>'})

        # 使用滑动窗口将图书分块为最大长度的重叠序列
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True):

    # 分词器初始化
    tokenizer = tiktoken.get_encoding("gpt2")

    # 创建数据集
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # 创建加载器
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

    return dataloader


"""  
chapter 3
"""
from typing import Tuple

class MultiHeadAttention(torch.nn.Module):
    def __init__(
            self,
            d_in: int,
            d_out: int,
            max_num_tokens: int,
            num_heads: int,
            dropout_rate: float,
            with_bias: bool = False,
            with_mask: bool = False,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.max_num_tokens = max_num_tokens
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.with_bias = with_bias
        self.with_mask = with_mask

        if d_out % num_heads != 0:
            raise ValueError(f"d_out必须可以被num_heads整除")
        self.d_head = d_out // num_heads

        self.wq = None
        self.wk = None
        self.wv = None
        # self.mask = None
        self.dropout = None
        self.out_proj = None
        self._init_parameters()

    def _init_parameters(self):
        d_in, d_out, with_bias = self.d_in, self.d_out, self.with_bias
        self.wq = torch.nn.Linear(in_features=d_in, out_features=d_out, bias=with_bias)
        self.wk = torch.nn.Linear(in_features=d_in, out_features=d_out, bias=with_bias)
        self.wv = torch.nn.Linear(in_features=d_in, out_features=d_out, bias=with_bias)

        block_size = self.max_num_tokens
        if self.with_mask:
            mask = torch.triu(torch.ones(block_size, block_size), diagonal=1).bool()
            self.register_buffer(name="mask", tensor=mask)  # 保存模型时，也会同时保存掩码

        self.dropout = torch.nn.Dropout(p=self.dropout_rate)

        self.out_proj = torch.nn.Linear(d_out, d_out)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, d_in = X.shape  # 这里需要获取实际输入数据集的上下文长度
        assert num_tokens <= self.max_num_tokens, f"输入序列长度 {num_tokens} 超过最大允许长度 {self.max_num_tokens}"
        assert d_in == self.d_in, "输入维度不正确"

        Q, K, V = self._compute_qkv(X)
        Q, K, V = self._reshape_qkv(Q, K, V, num_tokens)
        Q, K, V = self._transpose_qkv(Q, K, V)

        attention_scores = self._compute_attention_scores(Q, K)
        if self.with_mask:
            attention_scores = self._mask_attention_scores(attention_scores, num_tokens)

        attention_weights = self._compute_attention_weights(attention_scores)
        attention_weights = self.dropout(attention_weights)

        contexts = self._compute_contexts(attention_weights, V)
        contexts = self._transpose_contexts(contexts)
        contexts = self._reshape_contexts(contexts, num_tokens)
        
        Y = self.out_proj(contexts)
        return Y

    def _compute_qkv(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ 
        X.shape:        (batch_size, num_tokens, d_in)
        Q, K, V.shape:  (batch_size, num_tokens, d_out)
        """
        Q, K, V = self.wq(X), self.wk(X), self.wv(X)
        return Q, K, V
    
    def _reshape_qkv(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, num_tokens: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ 
        这里的num_tokens与self.num_tokens是不相同的，
        这里的num_tokens指的是输入数据的token数量，
        而self.num_tokens是初始化时设定的最大token数。
        """
        batch_size = Q.shape[0]
        """
        Q, K, V.shape:  (batch_size, num_tokens, d_out) -> (batch_size, num_tokens, num_heads, d_head)
        """
        Q = Q.reshape(batch_size, num_tokens, self.num_heads, self.d_head)
        K = K.reshape(batch_size, num_tokens, self.num_heads, self.d_head)
        V = V.reshape(batch_size, num_tokens, self.num_heads, self.d_head)
        return Q, K, V

    def _transpose_qkv(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ 
        Q, K, V.shape:  (batch_size, num_tokens, num_heads, d_head) -> (batch_size, num_heads, num_tokens, d_head)
        """
        Q = Q.transpose(1, 2).contiguous()
        K = K.transpose(1, 2).contiguous()
        V = V.transpose(1, 2).contiguous()
        return Q, K, V
    
    def _compute_attention_scores(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """  
        attention_scores.shape: (batch_size, num_heads, num_tokens, num_tokens)
        """
        attention_scores = Q @ K.transpose(2, 3)
        return attention_scores
    
    def _mask_attention_scores(self, attention_scores: torch.Tensor, num_tokens: int) -> torch.Tensor:
        mask = self.mask[:num_tokens, :num_tokens]
        mask = mask.reshape(1, 1, num_tokens, num_tokens)  # 实际输入数据的token数量未必就是初始化时的token数
        attention_scores.masked_fill_(mask, -torch.inf)
        return attention_scores

    def _compute_attention_weights(self, attention_scores: torch.Tensor) -> torch.Tensor:
        """  
        attention_weights.shape: (batch_size, num_heads, num_tokens, num_tokens)
        """
        attention_weights = torch.softmax(attention_scores / self.d_head ** 0.5, dim=-1)
        return attention_weights
    
    def _compute_contexts(self, attention_weights: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """  
        contexts.shape: (batch_size, num_heads, num_tokens, d_head)
        """
        contexts = attention_weights @ V
        return contexts
    
    def _transpose_contexts(self, contexts: torch.Tensor) -> torch.Tensor:
        """
        contexts.shape: (batch_size, num_heads, num_tokens, d_head) -> (batch_size, num_tokens, num_heads, d_head)
        """
        contexts = contexts.transpose(1, 2).contiguous()
        return contexts
    
    def _reshape_contexts(self, contexts: torch.Tensor, num_tokens: int) -> torch.Tensor:
        """
        contexts.shape: (batch_size, num_tokens, num_heads, d_head) -> (batch_size, num_tokens, d_out)
        """
        batch_size = contexts.shape[0]
        contexts = contexts.reshape(batch_size, num_tokens, self.d_out)
        return contexts


"""  
chapter 4
"""
class LayerNorm(torch.nn.Module):
    def __init__(self, d_emb: int):
        """ 
        Args:
            d_emb (int): 词嵌入的维度大小
        """
        super().__init__()
        self.delta = 1e-5
        self.scale = torch.nn.Parameter(torch.ones(d_emb))
        self.shift = torch.nn.Parameter(torch.zeros(d_emb))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        mean = X.mean(dim=-1, keepdim=True)  # 默认X张量的最后一个维度为词嵌入的维度
        var = X.var(dim=-1, keepdim=True, unbiased=False)  # 使用有偏方差
        X_normed = (X - mean) / torch.sqrt(var + self.delta)
        return X_normed * self.scale + self.shift


class GELU(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        Y = 0.5 * X * (1 + torch.tanh((2 / torch.pi) ** 0.5 * (X + 0.044715 * torch.pow(X, 3))))
        return Y


class FeedForward(torch.nn.Module):
    def __init__(self, emb_dim: int, dropout_rate: float):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 4 * emb_dim),
            GELU(),
            torch.nn.Linear(4 * emb_dim, emb_dim),
            torch.nn.Dropout(dropout_rate)
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        Y = self.layers(X)
        return Y
    

# 实现我们的transformer decoder-only模块
class TransformerDecoderOnly(torch.nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        max_num_tokens: int,
        num_heads: int,
        dropout_rate: float,
        with_bias: bool = False,
        with_mask: bool = False
    ):
        # 这里的d_in和d_out都是相同的
        assert d_in == d_out, "d_in and d_out must be equal in decoder-only."
        super().__init__()
        # Dropout是公用的：
        self.drop = torch.nn.Dropout(dropout_rate)
        # LN + MHA + Dropout：
        self.norm1 = LayerNorm(d_in)
        self.mha = MultiHeadAttention(
            d_in=d_in,
            d_out=d_out,
            max_num_tokens=max_num_tokens,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            with_bias=with_bias,
            with_mask=with_mask
        )
        # LN + FFN + Dropout：
        self.norm2 = LayerNorm(d_out)
        self.ffn = FeedForward(emb_dim=d_out, dropout_rate=dropout_rate)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        shortcut = X
        X = self.norm1(X)
        X = self.mha(X)
        X = self.drop(X)
        X += shortcut

        shortcut = X
        X = self.norm2(X)
        X = self.ffn(X)
        X = self.drop(X)
        X += shortcut
        return X
    

class GPT2Small(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        ctx_len: int,
        emb_dim: int,
        n_heads: int,
        n_layers: int,
        dropout_rate: float,
        with_bias: bool = False,
        with_mask: bool = True
    ):
        super().__init__()

        self.tok_emb = torch.nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = torch.nn.Embedding(ctx_len, emb_dim)
        self.decoder = torch.nn.Sequential(
            *[
                TransformerDecoderOnly(
                    d_in=emb_dim,
                    d_out=emb_dim,
                    max_num_tokens=ctx_len,
                    num_heads=n_heads,
                    dropout_rate=dropout_rate,
                    with_bias=with_bias,
                    with_mask=with_mask
                ) for _ in range(n_layers)
            ]
        )
        self.norm = LayerNorm(emb_dim)
        self.out = torch.nn.Linear(emb_dim, vocab_size, bias=with_bias)

    def forward(self, input_indices: torch.Tensor) -> torch.Tensor:
        # X是输入的token索引，形状为[batch_size, seq_len]
        batch_size, seq_len = input_indices.shape
        token_embeds = self.tok_emb(input_indices)
        position_embeds = self.pos_emb(torch.arange(seq_len, device=input_indices.device))
        X = token_embeds + position_embeds
        X = self.decoder(X)
        X = self.norm(X)
        Y = self.out(X)
        return Y


def generate_text_simple(model: torch.nn.Module, indices: torch.Tensor, max_new_tokens: int, context_size: int):
    """ 
    Args:
        model: GPT2 Small模型
        indices: 初始的上下文索引数组，形状为(B, T)
        max_new_tokens: 生成的最大新token数量
        context_size: 当前的上下文长度
    """
    for _ in range(max_new_tokens):
        # 如果当前上下文超过了支持的长度，就对当前上下文进行截断
        # 例如，如果LLM只支持5个token，而上下文长度为10，
        # 那么只有最后5个token会被用作上下文
        idx_cond = indices[:, -context_size:]
        
        # 获取预测结果
        with torch.no_grad():
            logits = model(idx_cond)
        
        # 只关注最后一个时间步
        # (batch, n_token, vocab_size)变为(batch, vocab_size)
        logits = logits[:, -1, :]  

        # 通过softmax函数获得对应的概率
        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

        # 获取概率值最高的单词索引
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)

        # 将采样到的索引添加到当前运行的上下文索引序列中
        indices = torch.cat((indices, idx_next), dim=1)  # (batch, n_tokens+1)

    return indices


""" 
chapter 5
"""
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt2, params):
    # Weight tying
    gpt2.pos_emb.weight = assign(gpt2.pos_emb.weight, params['wpe'])
    gpt2.tok_emb.weight = assign(gpt2.tok_emb.weight, params['wte'])
    
    for b in range(len(params["blocks"])):
        wq, wk, wv = np.split((params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt2.decoder[b].mha.wq.weight = assign(gpt2.decoder[b].mha.wq.weight, wq.T)
        gpt2.decoder[b].mha.wk.weight = assign(gpt2.decoder[b].mha.wk.weight, wk.T)
        gpt2.decoder[b].mha.wv.weight = assign(gpt2.decoder[b].mha.wv.weight, wv.T)
    
        q_b, k_b, v_b = np.split((params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt2.decoder[b].mha.wq.bias = assign(gpt2.decoder[b].mha.wq.bias, q_b)
        gpt2.decoder[b].mha.wk.bias = assign(gpt2.decoder[b].mha.wk.bias, k_b)
        gpt2.decoder[b].mha.wv.bias = assign(gpt2.decoder[b].mha.wv.bias, v_b)
    
        gpt2.decoder[b].mha.out_proj.weight = assign(gpt2.decoder[b].mha.out_proj.weight, params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt2.decoder[b].mha.out_proj.bias = assign(gpt2.decoder[b].mha.out_proj.bias, params["blocks"][b]["attn"]["c_proj"]["b"])
    
        gpt2.decoder[b].ffn.layers[0].weight = assign(gpt2.decoder[b].ffn.layers[0].weight, params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt2.decoder[b].ffn.layers[0].bias = assign(gpt2.decoder[b].ffn.layers[0].bias, params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt2.decoder[b].ffn.layers[2].weight = assign(gpt2.decoder[b].ffn.layers[2].weight, params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt2.decoder[b].ffn.layers[2].bias = assign(gpt2.decoder[b].ffn.layers[2].bias, params["blocks"][b]["mlp"]["c_proj"]["b"])
    
        gpt2.decoder[b].norm1.scale = assign(gpt2.decoder[b].norm1.scale, params["blocks"][b]["ln_1"]["g"])
        gpt2.decoder[b].norm1.shift = assign(gpt2.decoder[b].norm1.shift, params["blocks"][b]["ln_1"]["b"])
        gpt2.decoder[b].norm2.scale = assign(gpt2.decoder[b].norm2.scale, params["blocks"][b]["ln_2"]["g"])
        gpt2.decoder[b].norm2.shift = assign(gpt2.decoder[b].norm2.shift, params["blocks"][b]["ln_2"]["b"])
    
        gpt2.norm.scale = assign(gpt2.norm.scale, params["g"])
        gpt2.norm.shift = assign(gpt2.norm.shift, params["b"])
        gpt2.out.weight = assign(gpt2.out.weight, params["wte"])


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())