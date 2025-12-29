#!/usr/bin/env python3
"""
inference.py

Unified inference script for the three notebooks:
- rnn_nmt.ipynb  -> checkpoint like: rnn_nmt.pt
- transformer_nmt.ipynb -> checkpoint like: tf_nmt.pt
- t5_nmt.ipynb (fine-tuning Helsinki-NLP/opus-mt-zh-en) -> load from HF name or local dir

Examples
--------
# RNN checkpoint (beam)
python inference.py --backend rnn --ckpt rnn_nmt.pt --text "今天天气很好。"

# Transformer checkpoint (greedy or beam)
python inference.py --backend transformer --ckpt tf_nmt.pt --decode beam --beam_size 5 --text "我喜欢机器学习。"

# Pretrained/fine-tuned OPUS model (HF or local)
python inference.py --backend opus --model_name Helsinki-NLP/opus-mt-zh-en --text "我们去吃饭吧。"
python inference.py --backend opus --model_dir ./opus_zh_en_ckpt/best --text "我们去吃饭吧。"

# Batch from a file (one zh sentence per line)
python inference.py --backend opus --model_dir ./opus_zh_en_ckpt/best --input zh.txt --output pred.txt
"""

from __future__ import annotations
import argparse
import math
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Tokenization (matches notebooks)
# -------------------------
try:
    import jieba  # type: ignore
except Exception:
    jieba = None

SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]
PAD, BOS, EOS, UNK = SPECIALS

def tokenize_zh(s: str) -> List[str]:
    s = s.strip()
    if not s:
        return []
    if jieba is None:
        # fallback: char-level
        return list(s)
    return list(jieba.cut(s, cut_all=False))

def tokenize_en(s: str) -> List[str]:
    return s.strip().split()


# -------------------------
# Vocab (checkpoint carries itos/stoi)
# -------------------------
class Vocab:
    def __init__(self, itos: List[str], stoi: Dict[str, int]):
        self.itos = list(itos)
        self.stoi = dict(stoi)

    @classmethod
    def from_ckpt(cls, obj: Dict) -> "Vocab":
        return cls(obj["itos"], obj["stoi"])

    def encode(self, tokens: List[str], add_bos_eos: bool = True) -> List[int]:
        ids = []
        if add_bos_eos:
            ids.append(self.bos_id)
        for t in tokens:
            ids.append(self.stoi.get(t, self.unk_id))
        if add_bos_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: List[int], stop_at_eos: bool = True, remove_special: bool = True) -> List[str]:
        out = []
        for i in ids:
            if i < 0 or i >= len(self.itos):
                tok = UNK
            else:
                tok = self.itos[i]
            if stop_at_eos and tok == EOS:
                break
            if remove_special and tok in {PAD, BOS, EOS}:
                continue
            out.append(tok)
        return out

    @property
    def pad_id(self) -> int: return self.stoi[PAD]
    @property
    def bos_id(self) -> int: return self.stoi[BOS]
    @property
    def eos_id(self) -> int: return self.stoi[EOS]
    @property
    def unk_id(self) -> int: return self.stoi[UNK]
    def __len__(self) -> int: return len(self.itos)


# ============================================================
# RNN Seq2Seq with Attention (matches saved state_dict structure)
# ============================================================
class Attention(nn.Module):
    """
    alignment:
      - dot: score = h^T s
      - general: score = h^T W s
      - additive: score = v^T tanh(W_h h + W_s s)
    """
    def __init__(self, alignment: str, hidden_size: int):
        super().__init__()
        assert alignment in ["dot", "general", "additive"]
        self.alignment = alignment
        self.hidden_size = hidden_size
        if alignment == "general":
            self.W = nn.Linear(hidden_size, hidden_size, bias=False)
        elif alignment == "additive":
            self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
            self.W_s = nn.Linear(hidden_size, hidden_size, bias=False)
            self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, query: torch.Tensor, keys: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # query: [B,H], keys: [B,S,H], mask: [B,S] (True for valid)
        if self.alignment == "dot":
            scores = torch.bmm(keys, query.unsqueeze(2)).squeeze(2)  # [B,S]
        elif self.alignment == "general":
            proj = self.W(keys)  # [B,S,H]
            scores = torch.bmm(proj, query.unsqueeze(2)).squeeze(2)
        else:
            e = torch.tanh(self.W_s(keys) + self.W_h(query).unsqueeze(1))  # [B,S,H]
            scores = self.v(e).squeeze(2)  # [B,S]
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = F.softmax(scores, dim=1)  # [B,S]
        context = torch.bmm(attn.unsqueeze(1), keys).squeeze(1)  # [B,H]
        return context, attn


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int, hidden_size: int, num_layers: int, rnn_type: str, dropout: float):
        super().__init__()
        assert rnn_type in ["gru", "lstm"]
        self.rnn_type = rnn_type
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        rnn_cls = nn.GRU if rnn_type == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=emb_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, src_ids: torch.Tensor, src_lens: torch.Tensor):
        # src_ids: [B,S]
        emb = self.embedding(src_ids)  # [B,S,E]
        # pack for speed/stability
        packed = nn.utils.rnn.pack_padded_sequence(emb, src_lens.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, state = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)  # [B,S,H]
        return out, state


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int, hidden_size: int, num_layers: int, rnn_type: str, dropout: float, attn: Attention):
        super().__init__()
        assert rnn_type in ["gru", "lstm"]
        self.rnn_type = rnn_type
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        rnn_cls = nn.GRU if rnn_type == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=emb_size + hidden_size,  # concat [emb, ctx]
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn = attn
        self.out = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward_step(self, y_prev: torch.Tensor, state, enc_out: torch.Tensor, enc_mask: torch.Tensor):
        # y_prev: [B]
        emb = self.dropout(self.embedding(y_prev)).unsqueeze(1)  # [B,1,E]

        # query = top layer hidden from previous state
        if self.rnn_type == "gru":
            query = state[-1]  # [B,H]
        else:
            h, c = state
            query = h[-1]

        ctx, attn = self.attn(query, enc_out, enc_mask)  # [B,H]
        ctx = ctx.unsqueeze(1)  # [B,1,H]

        rnn_inp = torch.cat([emb, ctx], dim=2)  # [B,1,E+H]
        rnn_out, new_state = self.rnn(rnn_inp, state)  # rnn_out: [B,1,H]
        rnn_out = rnn_out.squeeze(1)  # [B,H]
        logits = self.out(rnn_out)  # [B,V]
        return logits, new_state, attn


class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_pad_id: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_id = src_pad_id

    @torch.no_grad()
    def translate_greedy(self, src_ids: torch.Tensor, src_lens: torch.Tensor, bos_id: int, eos_id: int, max_len: int = 120):
        self.eval()
        enc_out, enc_state = self.encoder(src_ids, src_lens)
        B, S, H = enc_out.shape
        enc_mask = (src_ids != self.src_pad_id)

        dec_state = enc_state
        y_prev = torch.full((B,), bos_id, device=src_ids.device, dtype=torch.long)

        outputs: List[List[int]] = [[] for _ in range(B)]
        for step in range(max_len):
            logits, dec_state, _ = self.decoder.forward_step(y_prev, dec_state, enc_out, enc_mask)
            if step == 0:
                logits[:, eos_id] = -1e9
            y_prev = torch.argmax(logits, dim=1)
            for i in range(B):
                outputs[i].append(int(y_prev[i].item()))
            if all((len(o) > 0 and o[-1] == eos_id) for o in outputs):
                break
        return outputs

    @torch.no_grad()
    def translate_beam(
        self,
        src_ids: torch.Tensor,
        src_lens: torch.Tensor,
        bos_id: int,
        eos_id: int,
        beam_size: int = 5,
        max_len: int = 120,
        len_norm_alpha: float = 0.6,
    ):
        """
        Beam search (batch size = 1), matching notebook intent.
        """
        self.eval()
        assert src_ids.size(0) == 1, "Beam search here assumes batch_size=1."
        device = src_ids.device
        enc_out, enc_state = self.encoder(src_ids, src_lens)
        enc_mask = (src_ids != self.src_pad_id)

        # beams: (tokens, state, logp, ended)
        beams = [([bos_id], enc_state, 0.0, False)]

        for step in range(max_len):
            new_beams = []
            all_ended = True
            for tokens, state, logp, ended in beams:
                if ended:
                    new_beams.append((tokens, state, logp, True))
                    continue
                all_ended = False
                y_prev = torch.tensor([tokens[-1]], device=device, dtype=torch.long)
                logits, new_state, _ = self.decoder.forward_step(y_prev, state, enc_out, enc_mask)
                if step == 0:
                    logits[:, eos_id] = -1e9
                lprobs = F.log_softmax(logits, dim=1).squeeze(0)  # [V]
                topk = torch.topk(lprobs, k=beam_size)
                for next_id, lp in zip(topk.indices.tolist(), topk.values.tolist()):
                    ntok = tokens + [int(next_id)]
                    nlogp = logp + float(lp)
                    nend = (next_id == eos_id)
                    new_beams.append((ntok, new_state, nlogp, nend))

            if all_ended:
                break

            def norm_score(b):
                tokens, _, lp, _ = b
                L = max(1, len(tokens) - 1)  # exclude BOS
                return lp / ((5 + L) ** len_norm_alpha / (5 ** len_norm_alpha))

            new_beams.sort(key=norm_score, reverse=True)
            beams = new_beams[:beam_size]

        best = max(beams, key=lambda b: b[2])
        return [best[0][1:]]  # remove BOS


def load_rnn_ckpt(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt["args"]
    src_vocab = Vocab.from_ckpt(ckpt["src_vocab"])
    tgt_vocab = Vocab.from_ckpt(ckpt["tgt_vocab"])

    enc = Encoder(
        vocab_size=len(src_vocab),
        emb_size=int(args["emb"]),
        hidden_size=int(args["hid"]),
        num_layers=2,
        rnn_type=str(args.get("rnn", "gru")),
        dropout=float(args.get("dropout", 0.1)),
    )
    attn = Attention(str(args.get("attn", "general")), hidden_size=int(args["hid"]))
    dec = Decoder(
        vocab_size=len(tgt_vocab),
        emb_size=int(args["emb"]),
        hidden_size=int(args["hid"]),
        num_layers=2,
        rnn_type=str(args.get("rnn", "gru")),
        dropout=float(args.get("dropout", 0.1)),
        attn=attn,
    )
    model = Seq2Seq(enc, dec, src_pad_id=src_vocab.pad_id).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, src_vocab, tgt_vocab, args


# ============================================================
# Transformer NMT (torch.nn.Transformer) + greedy/beam decoding
# ============================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerNMT(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        pad_id: int,
        d_model: int = 256,
        nhead: int = 4,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_len: int = 5000,
        pos_type: str = "sin",
    ):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.src_emb = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_id)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_id)
        self.pos_type = pos_type
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.pos = PositionalEncoding(d_model, dropout=dropout, max_len=max_len)

        self.tf = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.proj = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src_ids: torch.Tensor, tgt_inp_ids: torch.Tensor) -> torch.Tensor:
        # src_ids: [B,S], tgt_inp_ids: [B,T]
        if self.pos_type == "sin":
            src = self.pos(self.src_emb(src_ids) * math.sqrt(self.d_model))
            tgt = self.pos(self.tgt_emb(tgt_inp_ids) * math.sqrt(self.d_model))
        else:
            # learned positional embeddings
            src = self.src_emb(src_ids) * math.sqrt(self.d_model)
            tgt = self.tgt_emb(tgt_inp_ids) * math.sqrt(self.d_model)
            pos_s = torch.arange(src.size(1), device=src.device).unsqueeze(0).expand(src.size(0), -1)
            pos_t = torch.arange(tgt.size(1), device=tgt.device).unsqueeze(0).expand(tgt.size(0), -1)
            src = src + self.pos_emb(pos_s)
            tgt = tgt + self.pos_emb(pos_t)
            src = F.dropout(src, p=self.tf.dropout, training=self.training)
            tgt = F.dropout(tgt, p=self.tf.dropout, training=self.training)

        # masks
        src_key_padding_mask = (src_ids == self.pad_id)  # True = pad
        tgt_key_padding_mask = (tgt_inp_ids == self.pad_id)
        T = tgt_inp_ids.size(1)
        tgt_mask = torch.triu(torch.ones(T, T, device=src.device), diagonal=1).bool()

        out = self.tf(
            src=src,
            tgt=tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        logits = self.proj(out)  # [B,T,V]
        return logits

    @torch.no_grad()
    def translate_greedy(self, src_ids: torch.Tensor, bos_id: int, eos_id: int, max_len: int = 120) -> List[int]:
        self.eval()
        device = src_ids.device
        ys = torch.tensor([[bos_id]], device=device, dtype=torch.long)
        for _ in range(max_len):
            logits = self.forward(src_ids, ys)  # [1,t,V]
            next_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
            ys = torch.cat([ys, torch.tensor([[next_id]], device=device, dtype=torch.long)], dim=1)
            if next_id == eos_id:
                break
        return ys.squeeze(0).tolist()[1:]  # remove BOS

    @torch.no_grad()
    def translate_beam(
        self,
        src_ids: torch.Tensor,
        bos_id: int,
        eos_id: int,
        beam_size: int = 5,
        max_len: int = 120,
        len_norm_alpha: float = 0.6,
    ) -> List[int]:
        self.eval()
        device = src_ids.device
        beams = [([bos_id], 0.0, False)]  # tokens, logp, ended

        for _ in range(max_len):
            new_beams = []
            all_ended = True
            for tokens, logp, ended in beams:
                if ended:
                    new_beams.append((tokens, logp, True))
                    continue
                all_ended = False
                ys = torch.tensor([tokens], device=device, dtype=torch.long)
                logits = self.forward(src_ids, ys)  # [1,t,V]
                lprobs = F.log_softmax(logits[:, -1, :], dim=-1).squeeze(0)
                topk = torch.topk(lprobs, k=beam_size)
                for next_id, lp in zip(topk.indices.tolist(), topk.values.tolist()):
                    ntok = tokens + [int(next_id)]
                    nlogp = logp + float(lp)
                    nend = (next_id == eos_id)
                    new_beams.append((ntok, nlogp, nend))
            if all_ended:
                break

            def norm_score(b):
                tokens, lp, _ = b
                L = max(1, len(tokens) - 1)
                return lp / ((5 + L) ** len_norm_alpha / (5 ** len_norm_alpha))

            new_beams.sort(key=norm_score, reverse=True)
            beams = new_beams[:beam_size]

        best = max(beams, key=lambda b: b[1])
        return best[0][1:]  # remove BOS


def load_transformer_ckpt(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt["args"]
    src_vocab = Vocab.from_ckpt(ckpt["src_vocab"])
    tgt_vocab = Vocab.from_ckpt(ckpt["tgt_vocab"])

    model = TransformerNMT(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        pad_id=src_vocab.pad_id,
        d_model=int(args.get("d_model", 256)),
        nhead=int(args.get("nhead", 4)),
        num_encoder_layers=int(args.get("enc_layers", args.get("num_encoder_layers", 4))),
        num_decoder_layers=int(args.get("dec_layers", args.get("num_decoder_layers", 4))),
        dim_feedforward=int(args.get("ffn", args.get("dim_feedforward", 1024))),
        dropout=float(args.get("dropout", 0.1)),
        max_len=int(args.get("max_len", 5000)),
        pos_type=str(args.get("pos_type", "sin")),
    ).to(device)

    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, src_vocab, tgt_vocab, args


# ============================================================
# OPUS / HuggingFace Seq2Seq (pretrained or fine-tuned)
# ============================================================
def load_opus(model_name: Optional[str], model_dir: Optional[str], device: str):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  # lazy import
    if model_dir:
        tok = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    else:
        if not model_name:
            raise ValueError("For --backend opus, provide --model_name or --model_dir")
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model, tok


# -------------------------
# IO helpers
# -------------------------
def read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]

def write_lines(path: str, lines: List[str]):
    with open(path, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")


# -------------------------
# Main
# -------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["rnn", "transformer", "opus"], required=True,
                   help="Which model family to use.")
    p.add_argument("--ckpt", type=str, default=None,
                   help="Path to .pt checkpoint for rnn/transformer backends.")
    p.add_argument("--model_name", type=str, default=None,
                   help="HuggingFace model name (opus backend).")
    p.add_argument("--model_dir", type=str, default=None,
                   help="Local directory of a fine-tuned HF model (opus backend).")
    p.add_argument("--device", type=str, default=None,
                   help="cpu/cuda. Default: auto.")
    p.add_argument("--decode", choices=["greedy", "beam"], default="greedy",
                   help="Decoding strategy for rnn/transformer checkpoints.")
    p.add_argument("--beam_size", type=int, default=5)
    p.add_argument("--max_len", type=int, default=120)
    p.add_argument("--text", type=str, default=None,
                   help="Single input sentence (Chinese).")
    p.add_argument("--input", type=str, default=None,
                   help="Path to a text file, one Chinese sentence per line.")
    p.add_argument("--output", type=str, default=None,
                   help="Where to write translations (only when --input is used).")
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Collect inputs
    if args.text is not None:
        inputs = [args.text]
    elif args.input is not None:
        inputs = read_lines(args.input)
    else:
        # stdin lines
        inputs = [ln.rstrip("\n") for ln in sys.stdin if ln.strip()]

    if args.backend in ["rnn", "transformer"] and not args.ckpt:
        raise SystemExit("--ckpt is required for rnn/transformer backends")

    outputs: List[str] = []

    if args.backend == "rnn":
        model, src_vocab, tgt_vocab, _ = load_rnn_ckpt(args.ckpt, device)
        for s in inputs:
            src_tokens = tokenize_zh(s)
            src_ids = torch.tensor([src_vocab.encode(src_tokens, add_bos_eos=True)], device=device, dtype=torch.long)
            src_lens = torch.tensor([src_ids.size(1)], device=device, dtype=torch.long)
            if args.decode == "beam":
                pred_ids = model.translate_beam(src_ids, src_lens, src_vocab.bos_id, src_vocab.eos_id,
                                                beam_size=args.beam_size, max_len=args.max_len)[0]
            else:
                pred_ids = model.translate_greedy(src_ids, src_lens, src_vocab.bos_id, src_vocab.eos_id,
                                                  max_len=args.max_len)[0]
            pred_toks = tgt_vocab.decode(pred_ids, stop_at_eos=True, remove_special=True)
            outputs.append(" ".join(pred_toks))

    elif args.backend == "transformer":
        model, src_vocab, tgt_vocab, _ = load_transformer_ckpt(args.ckpt, device)
        for s in inputs:
            src_tokens = tokenize_zh(s)
            src_ids = torch.tensor([src_vocab.encode(src_tokens, add_bos_eos=True)], device=device, dtype=torch.long)
            if args.decode == "beam":
                pred_ids = model.translate_beam(src_ids, src_vocab.bos_id, src_vocab.eos_id,
                                                beam_size=args.beam_size, max_len=args.max_len)
            else:
                pred_ids = model.translate_greedy(src_ids, src_vocab.bos_id, src_vocab.eos_id, max_len=args.max_len)
            pred_toks = tgt_vocab.decode(pred_ids, stop_at_eos=True, remove_special=True)
            outputs.append(" ".join(pred_toks))

    else:
        model, tok = load_opus(args.model_name, args.model_dir, device)
        from transformers import GenerationConfig  # lazy import
        gen_cfg = GenerationConfig(
            max_new_tokens=args.max_len,
            num_beams=args.beam_size if args.decode == "beam" else 1,
            do_sample=False,
        )
        for s in inputs:
            batch = tok([s], return_tensors="pt", padding=True, truncation=True).to(device)
            out_ids = model.generate(**batch, generation_config=gen_cfg)
            pred = tok.batch_decode(out_ids, skip_special_tokens=True)[0]
            outputs.append(pred)

    if args.output:
        write_lines(args.output, outputs)
    else:
        for o in outputs:
            print(o)


if __name__ == "__main__":
    main()
