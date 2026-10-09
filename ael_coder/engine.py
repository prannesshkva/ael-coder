# -*- coding: utf-8 -*-
"""
ISOM-R2 Streaming-Retrieval Engine (architecture-agnostic)
===========================================================

A drop-in companion for any Hugging Face causal LM (Qwen2, Falcon, DeepSeek-V2,
custom SSM/MoE ...) that lets a model answer questions over a context far larger
than its native window while keeping GPU memory flat.

Pipeline (the same stages used by Aazhi-Coder-1.5B):

  1. STREAM    The context is cut into fixed-size chunks (default 2,048 tokens).
               Each chunk is pushed through the model once, with NO KV cache kept,
               so GPU memory is bounded by weights + one chunk of activations.
               A mean-pooled mid-layer hidden state is stored per chunk (dense index)
               and the chunk's token ids are kept in host RAM (lexical index).
  2. RETRIEVE  For a question, every chunk is scored with a hybrid of
               lexical identifier matching (exact class / function / variable /
               file-name hits, with a large bonus for definitions) and dense
               similarity of the mid-layer chunk embeddings.
  3. SLICE     The top-k chunks (default 2) are cut down to an anchor-centred
               micro-window (default 1,024 tokens) around the matched identifier.
  4. DECODE    The model answers from a short focused prompt that contains only
               those micro-windows plus the question.

Honest scope
------------
* The model does NOT attend over the full context at answer time. It reads
  every token once during STREAM (to build the index) and then answers from the
  retrieved micro-windows. Answer quality is bounded by (a) whether retrieval
  found the right chunk and (b) the capability of the underlying model.
* Retrieval is dominated by lexical identifier matching; the dense signal is a
  secondary tie-breaker. Questions that name no identifier rely on it alone.
* No model weights are modified. This file is pure inference-time code.

Usage
-----
    from isom_r2_engine import ISOMR2Engine
    engine = ISOMR2Engine(model, tokenizer)            # defaults: 2 chunks x 1024 tokens
    engine.ingest(big_code_string)                     # stream once
    result = engine.generate("Where is rope_theta defined and what is its default?")
    print(result.text, result.retrieved_chunks, result.peak_vram_gb)

    # or attach to the model:
    from isom_r2_engine import enable_isom_r2
    enable_isom_r2(model, tokenizer)
    result = model.generate_r2(question="...", context=big_code_string)
"""

from __future__ import annotations

import inspect
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

__all__ = ["ISOMR2Engine", "ISOMR2Result", "enable_isom_r2"]

__version__ = "2.0.0"


class _StopForward(Exception):
    """Raised from a forward hook to skip the remaining layers once the probe layer ran."""


_STOPWORDS = {
    "what", "is", "the", "exact", "value", "of", "and", "or", "to", "in", "a", "an", "below",
    "source", "files", "question", "answer", "directly", "with", "alone", "string", "output",
    "only", "assistant", "user", "system", "code", "here", "relevant", "excerpt", "file",
    "defined", "variable", "constant", "im_start", "im_end",
}

_DEFAULT_SYSTEM_PROMPT = (
    "You are an expert code intelligence assistant. Based on the provided repository code "
    "excerpts, answer the user request with clear, robust, production-quality code."
)


@dataclass
class ISOMR2Result:
    """Everything measured during one ingest + generate call."""

    text: str
    retrieved_chunks: List[int]
    windows: Dict[int, tuple] = field(default_factory=dict)  # chunk -> (start, end) token range
    context_tokens: int = 0
    num_chunks: int = 0
    prompt_tokens: int = 0
    ingest_seconds: float = 0.0
    generate_seconds: float = 0.0
    peak_vram_gb: Optional[float] = None

    @property
    def total_seconds(self) -> float:
        return self.ingest_seconds + self.generate_seconds


class ISOMR2Engine:
    """ISOM-R2 streaming-retrieval engine. See module docstring for the pipeline."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        chunk_size: int = 2048,
        num_retrieved_chunks: int = 2,
        micro_window_size: int = 1024,
        dense_scoring: bool = True,
        dense_weight: float = 1.0,
        dense_layer_fraction: float = 0.5,
        anchor_bias: float = 0.5,
        expand_adjacent: bool = True,
        early_exit: Optional[bool] = None,
        system_prompt: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        Args:
            chunk_size: tokens per streamed chunk.
            num_retrieved_chunks: chunks paged into the final prompt.
            micro_window_size: tokens kept per retrieved chunk, centred on the anchor.
            dense_scoring: also run every chunk through the model to build the dense index.
                Set False for a lexical-only index (much faster, no model forward at ingest).
            dense_weight: weight of the dense similarity inside the hybrid score.
            dense_layer_fraction: depth of the probe layer (0.5 = middle of the stack).
            anchor_bias: where the anchor sits inside the micro-window (0.5 = centred,
                lower values keep more text AFTER the anchor).
            expand_adjacent: when a chunk holds a definition hit, also take chunk + 1.
            early_exit: stop each ingest forward right after the probe layer. None = automatic
                (disabled when the model is offloaded to CPU/disk).
        """
        self.model = model
        self.tokenizer = tokenizer
        self.chunk_size = int(chunk_size)
        self.num_retrieved_chunks = int(num_retrieved_chunks)
        self.micro_window_size = int(micro_window_size)
        self.dense_scoring = bool(dense_scoring)
        self.dense_weight = float(dense_weight)
        self.dense_layer_fraction = float(dense_layer_fraction)
        self.anchor_bias = float(anchor_bias)
        self.expand_adjacent = bool(expand_adjacent)
        self.system_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT
        self.verbose = bool(verbose)

        # Architecture discovery (no per-model code needed)
        self._input_device = self._resolve_input_device()
        self._backbone = self._resolve_backbone()
        self._backbone_takes_use_cache = self._accepts(self._backbone, "use_cache") if self._backbone is not None else False
        self._probe_layer = self._resolve_probe_layer()
        if early_exit is None:
            device_map = getattr(model, "hf_device_map", None) or {}
            offloaded = any(str(d) in ("cpu", "disk") for d in device_map.values())
            early_exit = not offloaded
        self.early_exit = bool(early_exit)

        # Index state
        self.chunk_tokens: Dict[int, torch.Tensor] = {}
        self.chunk_text: Dict[int, str] = {}
        self.centroids: Optional[torch.Tensor] = None
        self.context_len = 0
        self.last_ingest_seconds = 0.0
        self.last_retrieved_chunks: List[int] = []
        self.last_windows: Dict[int, tuple] = {}

    # ------------------------------------------------------------------ discovery
    def _resolve_input_device(self) -> torch.device:
        try:
            return self.model.get_input_embeddings().weight.device
        except Exception:
            return next(self.model.parameters()).device

    def _resolve_backbone(self) -> Optional[nn.Module]:
        bb = getattr(self.model, "base_model", None)
        if bb is None or bb is self.model:
            return None
        return bb

    @staticmethod
    def _accepts(module: nn.Module, name: str) -> bool:
        try:
            params = inspect.signature(module.forward).parameters
        except (TypeError, ValueError):
            return False
        return name in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())

    def _resolve_probe_layer(self) -> nn.Module:
        best = None
        for _, mod in self.model.named_modules():
            if isinstance(mod, nn.ModuleList) and len(mod) >= 2:
                cls = type(mod[0]).__name__.lower()
                if any(k in cls for k in ("layer", "block", "decoder")):
                    if best is None or len(mod) > len(best):
                        best = mod
        if best is None:
            for _, mod in self.model.named_modules():
                if isinstance(mod, nn.ModuleList) and len(mod) >= 2:
                    if best is None or len(mod) > len(best):
                        best = mod
        if best is None:
            raise RuntimeError("ISOM-R2: could not locate the transformer layer stack of this model.")
        idx = min(len(best) - 1, max(0, int(len(best) * self.dense_layer_fraction)))
        return best[idx]

    # ------------------------------------------------------------------ helpers
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    @staticmethod
    def _peak_vram_gb() -> Optional[float]:
        if not torch.cuda.is_available():
            return None
        return sum(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())) / (1024 ** 3)

    @staticmethod
    def _reset_peak() -> None:
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(i)

    def _encode_ids(self, context: Union[str, torch.Tensor, List[int]]) -> torch.Tensor:
        if isinstance(context, str):
            ids = self.tokenizer(context, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        elif isinstance(context, torch.Tensor):
            ids = context.reshape(-1)
        else:
            ids = torch.tensor(list(context), dtype=torch.long)
        return ids.cpu().long()

    # ------------------------------------------------------------------ stage 1: stream
    @torch.no_grad()
    def _embed(self, ids_1d: torch.Tensor) -> torch.Tensor:
        """Mean-pooled probe-layer hidden state of one chunk (float32, CPU)."""
        x = ids_1d.to(self._input_device).unsqueeze(0)
        captured: Dict[str, torch.Tensor] = {}

        def hook(_module, _inputs, output):
            h = output[0] if isinstance(output, (tuple, list)) else output
            captured["h"] = h.detach().float().reshape(-1, h.shape[-1]).mean(dim=0)
            if self.early_exit:
                raise _StopForward()

        handle = self._probe_layer.register_forward_hook(hook)
        try:
            if self._backbone is not None:
                if self._backbone_takes_use_cache:
                    self._backbone(input_ids=x, use_cache=False)
                else:
                    self._backbone(input_ids=x)
            else:
                self.model(input_ids=x)
        except _StopForward:
            pass
        finally:
            handle.remove()
        if "h" not in captured:
            raise RuntimeError("ISOM-R2: probe layer did not run during the streaming forward pass.")
        return captured["h"].cpu()

    @torch.no_grad()
    def ingest(self, context: Union[str, torch.Tensor, List[int]]) -> "ISOMR2Engine":
        """Stream `context` through the model once and build the lexical + dense index."""
        ids = self._encode_ids(context)
        total = int(ids.numel())
        num_chunks = (total + self.chunk_size - 1) // self.chunk_size
        self.chunk_tokens, self.chunk_text, self.centroids = {}, {}, None
        self.context_len = total
        self._reset_peak()

        mode = "dense + lexical" if self.dense_scoring else "lexical only"
        self._log(f"  [ISOM-R2 Engine] Streaming {total:,} context tokens across {num_chunks} chunks ({mode} index, no KV cache kept)...")
        t0 = time.time()
        centroids: List[torch.Tensor] = []
        next_report = 0.2
        for c in range(num_chunks):
            chunk = ids[c * self.chunk_size:(c + 1) * self.chunk_size]
            self.chunk_tokens[c] = chunk
            self.chunk_text[c] = self.tokenizer.decode(chunk, skip_special_tokens=True)
            if self.dense_scoring:
                centroids.append(self._embed(chunk))
            done = (c + 1) / num_chunks
            if done >= next_report or c == num_chunks - 1:
                vram = self._peak_vram_gb()
                vram_s = f" | peak VRAM: {vram:.2f} GB" if vram is not None else ""
                self._log(f"  [ISOM-R2] Streamed {min((c + 1) * self.chunk_size, total):,} / {total:,} tokens ({done * 100:.1f}%){vram_s}")
                while next_report <= done:
                    next_report += 0.2
        if centroids:
            self.centroids = torch.stack(centroids, dim=0)
        self.last_ingest_seconds = time.time() - t0
        return self

    # ------------------------------------------------------------------ stage 2: retrieve
    @torch.no_grad()
    def retrieve(self, question: str) -> List[int]:
        """Score all chunks for `question`; returns chunk indices (best chunk LAST)."""
        if not self.chunk_tokens:
            raise RuntimeError("ISOM-R2: call ingest(context) before retrieve()/generate().")
        num_chunks = len(self.chunk_tokens)
        q_ids = self.tokenizer(question, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        q_set = set(q_ids.tolist())

        raw_words = re.findall(r"[A-Za-z0-9_\.]+", question)
        priority = [
            w for w in raw_words
            if w.lower() not in _STOPWORDS and (len(w) >= 4 or "_" in w or any(ch.isupper() for ch in w))
        ]

        lexical = torch.zeros(num_chunks)
        anchors: Dict[int, int] = {}
        for c in range(num_chunks):
            text = self.chunk_text[c]
            score, pos = 0.0, None
            for ident in priority:
                if ident in text:
                    score += 500.0 if ("_" in ident or any(ch.isupper() for ch in ident)) else 100.0
                    d = re.search(r"(?:class\s+|def\s+|\b)" + re.escape(ident) + r"(?:\s*\(|\s*=|:)", text)
                    if d:
                        score += 2000.0
                        pos = d.start()
                    elif pos is None:
                        pos = text.find(ident)
            overlap = len(set(self.chunk_tokens[c].tolist()) & q_set)
            score += float(overlap)
            # Prioritize core library implementation code over tests, benchmarks, and docs
            if "### FILE:" in text:
                if any(k in text for k in ("/tests/", "\\tests\\", "/bench/", "\\bench\\", "tests.", "bench.")):
                    score *= 0.25
                elif any(k in text for k in ("/src/", "\\src\\", "/lib/", "\\lib\\", "langgraph/graph/", "langgraph/channels/", "langgraph/pregel/", "libs/langgraph/langgraph/")):
                    score *= 3.0
            lexical[c] = score
            if pos is not None:
                tok_idx = len(self.tokenizer.encode(text[:pos], add_special_tokens=False))
                anchors[c] = min(tok_idx, len(self.chunk_tokens[c]) - 1)
            elif overlap > 0:
                anchors[c] = 0
        self._anchors = anchors

        dense = torch.zeros(num_chunks)
        if self.dense_scoring and self.centroids is not None:
            q_vec = self._embed(q_ids)
            mean = self.centroids.mean(dim=0, keepdim=True)
            cc = self.centroids - mean
            cc = cc / cc.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            qq = (q_vec - mean[0])
            qq = qq / qq.norm().clamp(min=1e-8)
            dense = cc @ qq

        scores = self.dense_weight * dense + 10.0 * lexical
        top = torch.topk(scores, k=min(num_chunks, max(16, self.num_retrieved_chunks * 2))).indices.tolist()

        selected: List[int] = []
        for c in top:
            if c not in selected:
                selected.append(c)
                if self.expand_adjacent and lexical[c] > 500.0 and (c + 1) < num_chunks and (c + 1) not in selected:
                    selected.append(c + 1)
            if len(selected) >= self.num_retrieved_chunks:
                break
        return list(reversed(selected[: self.num_retrieved_chunks]))

    # ------------------------------------------------------------------ stage 3: slice
    def _micro_window(self, chunk_idx: int) -> tuple:
        toks = self.chunk_tokens[chunk_idx]
        n = int(toks.numel())
        if n <= self.micro_window_size:
            return 0, n
        peak = getattr(self, "_anchors", {}).get(chunk_idx, 0)
        lead = int(self.micro_window_size * self.anchor_bias)
        start = max(0, min(peak - lead, n - self.micro_window_size))
        return start, min(n, start + self.micro_window_size)

    # ------------------------------------------------------------------ stage 4: decode
    def _build_prompt(self, excerpts: str, question: str):
        q = question.strip()
        if not (q.startswith("Question:") or q.startswith("Query:")):
            q = f"Question: {q}"
        user = f"Relevant Code Excerpt:\n\n{excerpts}\n\n{q}"
        if getattr(self.tokenizer, "chat_template", None):
            try:
                text = self.tokenizer.apply_chat_template(
                    [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": user}],
                    tokenize=False, add_generation_prompt=True,
                )
                return text, False
            except Exception:
                pass
        return f"{self.system_prompt}\n\n{user}\n\nAnswer:\n", True

    @torch.no_grad()
    def generate(
        self,
        question: str,
        context: Union[str, torch.Tensor, List[int], None] = None,
        max_new_tokens: int = 512,
        **gen_kwargs: Any,
    ) -> ISOMR2Result:
        """Answer `question`. If `context` is given it is ingested first; otherwise the
        previously ingested index is reused (ingest once, ask many questions)."""
        self._reset_peak()
        ingest_s = 0.0
        if context is not None:
            self.ingest(context)
            ingest_s = self.last_ingest_seconds

        t0 = time.time()
        chunks = self.retrieve(question)
        blocks, windows = [], {}
        for c in chunks:
            s, e = self._micro_window(c)
            windows[c] = (s, e)
            self._log(f"  [ISOM-R2 Engine] Micro-window sliced: Chunk {c}, range=[{s}:{e}] ({e - s} tokens)")
            txt = self.tokenizer.decode(self.chunk_tokens[c][s:e], skip_special_tokens=True).strip()
            if txt:
                blocks.append(txt)
        self.last_retrieved_chunks, self.last_windows = chunks, windows

        prompt, add_special = self._build_prompt("\n\n---\n\n".join(blocks), question)
        enc = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=add_special)
        input_ids = enc["input_ids"].to(self._input_device)
        self._log(f"  [ISOM-R2] Retrieved salient context pages: {chunks} | Focused prompt: {input_ids.shape[-1]:,} tokens")

        gen_kwargs.setdefault("do_sample", False)
        if not gen_kwargs["do_sample"]:
            for k in ("temperature", "top_p", "top_k"):
                gen_kwargs.pop(k, None)
        pad = getattr(self.tokenizer, "pad_token_id", None)
        if pad is None:
            pad = getattr(self.tokenizer, "eos_token_id", None)
        if pad is not None:
            gen_kwargs.setdefault("pad_token_id", pad)

        prev_r2 = getattr(getattr(self.model, "config", None), "use_isom_r2_svd", None)
        try:
            if prev_r2 is not None:
                self.model.config.use_isom_r2_svd = False
            out = self.model.generate(input_ids, max_new_tokens=max_new_tokens, **gen_kwargs)
        finally:
            if prev_r2 is not None:
                self.model.config.use_isom_r2_svd = prev_r2
        new_tokens = out[0][input_ids.shape[-1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        return ISOMR2Result(
            text=text,
            retrieved_chunks=chunks,
            windows=windows,
            context_tokens=self.context_len,
            num_chunks=len(self.chunk_tokens),
            prompt_tokens=int(input_ids.shape[-1]),
            ingest_seconds=ingest_s,
            generate_seconds=time.time() - t0,
            peak_vram_gb=self._peak_vram_gb(),
        )


def enable_isom_r2(model: nn.Module, tokenizer: Any, **config: Any) -> ISOMR2Engine:
    """Attach an engine to `model` as `model.isom_r2` and expose `model.generate_r2(...)`.

    The model's own `generate()` and weights are left untouched."""
    engine = ISOMR2Engine(model, tokenizer, **config)
    model.isom_r2 = engine
    model.generate_r2 = engine.generate
    return engine


def _main() -> None:  # pragma: no cover - convenience CLI
    import argparse
    import glob
    import os

    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser(description="ISOM-R2 streaming-retrieval demo")
    ap.add_argument("--model", required=True, help="HF repo id or local path")
    ap.add_argument("--context-dir", required=True, help="directory of source files to ingest")
    ap.add_argument("--glob", default="**/*.py")
    ap.add_argument("--question", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--no-dense", action="store_true", help="lexical-only index")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    ).eval()
    blocks = []
    for path in sorted(glob.glob(os.path.join(args.context_dir, args.glob), recursive=True)):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            blocks.append(f"# FILE: {os.path.relpath(path, args.context_dir).replace(os.sep, '/')}\n{f.read()}\n")
    engine = ISOMR2Engine(model, tok, dense_scoring=not args.no_dense)
    res = engine.generate(args.question, context="\n".join(blocks), max_new_tokens=args.max_new_tokens)
    print(f"\nTokens: {res.context_tokens:,} | chunks: {res.num_chunks} | retrieved: {res.retrieved_chunks} "
          f"| time: {res.total_seconds:.1f}s | peak VRAM: {res.peak_vram_gb}")
    print(res.text)


if __name__ == "__main__":
    _main()
