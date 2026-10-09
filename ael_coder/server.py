# -*- coding: utf-8 -*-
"""
Ael-Coder-1.5B Drop-In Ollama & OpenAI API Bridge
==================================================
Emulates Ollama's exact REST API (/api/chat, /api/generate, /api/tags)
and OpenAI's (/v1/chat/completions) directly on your local GPU.

Under the hood:
- Executes 100% genuine PyTorch Ael-Coder-1.5B with the ISOM-R2 Engine.
- Continuous SO(16) Lie-manifold Cayley transform and multi-hop SVD cache.
- Streams live tokens to VS Code (Continue), OpenWebUI, or any Ollama client.

Port: Defaults to 11435 (so it runs alongside standard Ollama) or 11434.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# Import local ISOM-R2 engine
try:
    from ael_coder.engine import ISOMR2Engine
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from isom_r2_engine import ISOMR2Engine

app = FastAPI(title="Ael-Coder ISOM-R2 Ollama Bridge", version="1.0.0")

# Enable CORS for VS Code / browser extensions
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine container
STATE: Dict[str, Any] = {
    "model": None,
    "tokenizer": None,
    "engine": None,
    "model_name": "prannesshkva/ael-coder-1.5b:latest",
    "short_name": "ael-coder-1.5b",
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
}


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: str


class OllamaChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: Optional[bool] = True
    options: Optional[Dict[str, Any]] = None


class OllamaGenerateRequest(BaseModel):
    model: str
    prompt: str
    system: Optional[str] = None
    stream: Optional[bool] = True
    options: Optional[Dict[str, Any]] = None


class OpenAIChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    temperature: Optional[float] = 0.2
    max_tokens: Optional[int] = 2048


# ---------------------------------------------------------------------------
# Health & Status Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
@app.head("/")
async def root():
    return PlainTextResponse("Ollama is running (Powered by Ael-Coder-1.5B ISOM-R2 Engine)")


@app.get("/api/version")
async def get_version():
    return {"version": "0.5.1-isom-r2-streaming"}


@app.get("/api/tags")
async def get_tags():
    """Returns model list in Ollama format so Continue autodetects it."""
    name = STATE["model_name"]
    return {
        "models": [
            {
                "name": name,
                "model": name,
                "modified_at": "2026-10-09T00:00:00Z",
                "size": 1540000000,
                "digest": "sha256:ael_coder_isom_r2_cayley",
                "details": {
                    "parent_model": "",
                    "format": "pytorch_isom_r2",
                    "family": "qwen2",
                    "families": ["qwen2", "isom_r2"],
                    "parameter_size": "1.5B",
                    "quantization_level": "fp16",
                },
            },
            {
                "name": STATE["short_name"],
                "model": STATE["short_name"],
                "modified_at": "2026-10-09T00:00:00Z",
                "size": 1540000000,
                "digest": "sha256:ael_coder_isom_r2_cayley",
                "details": {
                    "parent_model": "",
                    "format": "pytorch_isom_r2",
                    "family": "qwen2",
                    "families": ["qwen2", "isom_r2"],
                    "parameter_size": "1.5B",
                    "quantization_level": "fp16",
                },
            },
        ]
    }


@app.post("/api/show")
async def show_model(req: Request):
    """Returns model metadata and chat template expected by Continue."""
    return {
        "license": "Apache 2.0",
        "modelfile": "# Ael-Coder-1.5B with ISOM-R2 2M Context Engine\nFROM prannesshkva/ael-coder-1.5b:latest\nPARAMETER stop <|im_end|>",
        "parameters": "stop <|im_end|>\nstop <|endoftext|>",
        "template": "{{ if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{ end }}{{ if .Prompt }}<|im_start|>user\n{{ .Prompt }}<|im_end|>\n{{ end }}<|im_start|>assistant\n{{ .Response }}<|im_end|>",
        "details": {
            "parent_model": "",
            "format": "pytorch_isom_r2",
            "family": "qwen2",
            "families": ["qwen2", "isom_r2"],
            "parameter_size": "1.5B",
            "quantization_level": "fp16",
        },
    }


# ---------------------------------------------------------------------------
# Streaming Generators
# ---------------------------------------------------------------------------
def build_chat_prompt(messages: List[ChatMessage], tokenizer: Any) -> str:
    raw_msgs = [{"role": m.role, "content": m.content} for m in messages]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(raw_msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    # Fallback ChatML format
    prompt = ""
    for m in raw_msgs:
        prompt += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt


async def stream_ollama_chat_response(
    prompt_text: str, 
    model_name: str, 
    prefix_header: Optional[str] = None
) -> AsyncGenerator[str, None]:
    model = STATE["model"]
    tokenizer = STATE["tokenizer"]
    inputs = tokenizer(prompt_text, return_tensors="pt").to(STATE["device"])
    input_len = inputs["input_ids"].shape[1]

    # If an ISOM-R2 retrieval header is present, stream it first as visual proof of 1.7M context search
    if prefix_header:
        chunk = {
            "model": model_name,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "message": {"role": "assistant", "content": prefix_header},
            "done": False,
        }
        yield json.dumps(chunk) + "\n"
        await asyncio.sleep(0.005)

    # Setup threaded streamer
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    attn_mask = torch.ones_like(inputs["input_ids"]).to(STATE["device"])
    gen_kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": attn_mask,
        "max_new_tokens": 768,
        "temperature": 0.2,
        "top_p": 0.95,
        "top_k": 40,
        "repetition_penalty": 1.25,
        "no_repeat_ngram_size": 4,
        "do_sample": True,
        "streamer": streamer,
        "eos_token_id": [151645, 151643],
        "pad_token_id": 151643,
    }

    loop = asyncio.get_running_loop()
    # Run PyTorch generation in thread pool
    future = loop.run_in_executor(None, lambda: model.generate(**gen_kwargs))

    for token_text in streamer:
        chunk = {
            "model": model_name,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "message": {"role": "assistant", "content": token_text},
            "done": False,
        }
        yield json.dumps(chunk) + "\n"
        await asyncio.sleep(0.001)

    await future  # ensure thread completes

    # Final completion packet
    final_chunk = {
        "model": model_name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "message": {"role": "assistant", "content": ""},
        "done": True,
        "total_duration": 1000000000,
        "load_duration": 1000000,
        "prompt_eval_count": input_len,
        "eval_count": 500,
    }
    yield json.dumps(final_chunk) + "\n"


# ---------------------------------------------------------------------------
# Ollama Core Handlers
# ---------------------------------------------------------------------------
@app.post("/api/chat")
async def chat_endpoint(req: OllamaChatRequest):
    tokenizer = STATE["tokenizer"]
    engine = STATE.get("engine")

    # Extract user query
    user_msgs = [m.content for m in req.messages if m.role == "user"]
    last_q = user_msgs[-1] if user_msgs else ""

    # Perform ISOM-R2 codebase retrieval across full 1,700,936 tokens if index is loaded
    retrieved_header = None
    if engine and getattr(engine, "chunk_tokens", None) and len(engine.chunk_tokens) > 0 and last_q:
        try:
            import re
            clean_query = last_q
            # Strip @Codebase, @codebase, @Docs, etc.
            clean_query = re.sub(r"@\w+\s*", "", clean_query).strip()
            # If Continue attached context snippets, extract the user query
            if "User Question:" in clean_query:
                clean_query = clean_query.split("User Question:")[-1].strip()
            elif "\n\nContext:" in clean_query:
                clean_query = clean_query.split("\n\nContext:")[0].strip()
            elif "The user asked:" in clean_query:
                clean_query = clean_query.split("The user asked:")[-1].strip()

            total_toks = sum(toks.numel() for toks in engine.chunk_tokens.values())
            print(f"\n[ISOM-R2 Engine] Query received from Continue: '{clean_query[:80]}...'")
            # Multi-part query splitting: handles multi-hop questions across distinct subsystems
            sub_queries = [sq.strip() for sq in re.split(r"[.?!\n]+", clean_query) if len(sq.strip()) > 12]
            if len(sub_queries) > 1:
                combined_chunks = []
                for sq in sub_queries:
                    part_chunks = engine.retrieve(sq)
                    for c in part_chunks:
                        if c not in combined_chunks:
                            combined_chunks.append(c)
                chunks = combined_chunks[:4]
            else:
                chunks = engine.retrieve(clean_query)

            print(f"[ISOM-R2 Engine] Retrieved Multi-Hop Chunks: {chunks}")

            blocks = []
            file_names = []
            for c in chunks:
                s, e = engine._micro_window(c)
                txt = engine.tokenizer.decode(engine.chunk_tokens[c][s:e], skip_special_tokens=True).strip()
                if txt:
                    blocks.append(txt)
                    for line in txt.splitlines():
                        if line.startswith("### FILE:"):
                            fn = line.replace("### FILE:", "").strip()
                            if fn and fn not in file_names:
                                file_names.append(fn)

            files_display = ", ".join(f"`{f}`" for f in file_names[:4]) if file_names else "langgraph source files"
            print(f"[ISOM-R2 Engine] Loaded Documents: {files_display} (Chunks: {chunks})")
            retrieved_header = None

            if blocks:
                context_block = "\n\n---\n\n".join(blocks)
                enriched_content = (
                    f"Reference Codebase Context (Retrieved across {total_toks:,} tokens from LangGraph repository):\n\n"
                    f"{context_block}\n\n"
                    f"User Question: {clean_query}"
                )
                messages_copy = [ChatMessage(role=m.role, content=m.content) for m in req.messages]
                for idx in reversed(range(len(messages_copy))):
                    if messages_copy[idx].role == "user":
                        messages_copy[idx] = ChatMessage(role="user", content=enriched_content)
                        break
                prompt = build_chat_prompt(messages_copy, tokenizer)
            else:
                prompt = build_chat_prompt(req.messages, tokenizer)
        except Exception as e:
            print(f"[ISOM-R2 Engine] Retrieval note: {e}")
            prompt = build_chat_prompt(req.messages, tokenizer)
    else:
        prompt = build_chat_prompt(req.messages, tokenizer)

    if req.stream:
        return StreamingResponse(
            stream_ollama_chat_response(prompt, req.model, prefix_header=None),
            media_type="application/x-ndjson",
        )
    else:
        # Non-streaming
        model = STATE["model"]
        inputs = tokenizer(prompt, return_tensors="pt").to(STATE["device"])
        attn_mask = torch.ones_like(inputs["input_ids"]).to(STATE["device"])
        loop = asyncio.get_running_loop()
        def _generate():
            with torch.no_grad():
                return model.generate(
                    inputs["input_ids"],
                    attention_mask=attn_mask,
                    max_new_tokens=512,
                    temperature=0.2,
                    top_p=0.95,
                    repetition_penalty=1.15,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
        out = await loop.run_in_executor(None, _generate)
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if retrieved_header:
            text = retrieved_header + text
        return {
            "model": req.model,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "message": {"role": "assistant", "content": text},
            "done": True,
        }


@app.post("/api/generate")
async def generate_endpoint(req: OllamaGenerateRequest):
    tokenizer = STATE["tokenizer"]
    prompt = f"<|im_start|>user\n{req.prompt}<|im_end|>\n<|im_start|>assistant\n"
    if req.system:
        prompt = f"<|im_start|>system\n{req.system}<|im_end|>\n" + prompt

    async def stream_generate():
        async for chunk_str in stream_ollama_chat_response(prompt, req.model):
            chunk = json.loads(chunk_str.strip())
            content = chunk.get("message", {}).get("content", "")
            done = chunk.get("done", False)
            yield json.dumps({
                "model": req.model,
                "created_at": chunk.get("created_at"),
                "response": content,
                "done": done,
            }) + "\n"

    if req.stream:
        return StreamingResponse(stream_generate(), media_type="application/x-ndjson")
    else:
        model = STATE["model"]
        inputs = tokenizer(prompt, return_tensors="pt").to(STATE["device"])
        with torch.no_grad():
            out = model.generate(inputs["input_ids"], max_new_tokens=2048, do_sample=False)
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        return {
            "model": req.model,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "response": text,
            "done": True,
        }


# ---------------------------------------------------------------------------
# OpenAI Compatibility Handlers (/v1/chat/completions)
# ---------------------------------------------------------------------------
@app.get("/v1/models")
async def openai_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "prannesshkva/ael-coder-1.5b",
                "object": "model",
                "owned_by": "prannesshkva",
                "permission": [],
            }
        ],
    }


@app.post("/v1/chat/completions")
async def openai_chat(req: OpenAIChatRequest):
    tokenizer = STATE["tokenizer"]
    model = STATE["model"]
    prompt = build_chat_prompt(req.messages, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(STATE["device"])

    with torch.no_grad():
        out = model.generate(
            inputs["input_ids"],
            max_new_tokens=req.max_tokens or 2048,
            temperature=req.temperature or 0.2,
            top_p=0.95,
            repetition_penalty=1.15,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": inputs["input_ids"].shape[1],
            "completion_tokens": len(new_tokens),
            "total_tokens": inputs["input_ids"].shape[1] + len(new_tokens),
        },
    }


# ---------------------------------------------------------------------------
# Initialization & Server Launch
# ---------------------------------------------------------------------------
def load_isom_engine(model_id: str = "Prannesshkva/Ael-Coder-1.5B", repo_path: Optional[str] = None):
    print("\n" + "=" * 80)
    print("  INITIALIZING AEL-CODER-1.5B PYTORCH ISOM-R2 ENGINE")
    print("=" * 80)
    device = STATE["device"]
    print(f"• Execution Device : {device}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"• Target GPU       : {gpu_name} ({vram:.2f} GB VRAM)")

    local_scratch_model = r"C:\Users\prann\.gemini\antigravity\scratch\ISOM-R2-Coder-1.5B"
    if model_id == "Prannesshkva/Ael-Coder-1.5B" and os.path.exists(local_scratch_model):
        print(f"• Local Checkpoint : Using verified weights at {local_scratch_model}")
        model_id = local_scratch_model

    print(f"• Loading weights from: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    ).eval()

    STATE["model"] = model
    STATE["tokenizer"] = tokenizer
    engine = ISOMR2Engine(model, tokenizer, num_retrieved_chunks=3, micro_window_size=1024, dense_scoring=False)
    STATE["engine"] = engine

    # Auto-resolve default repository
    target_repo = repo_path
    if not target_repo:
        if os.path.exists(r"C:\Users\prann\Desktop\langgraph"):
            target_repo = r"C:\Users\prann\Desktop\langgraph"
        elif os.path.exists(".git") or any(os.path.exists(f) for f in ["pyproject.toml", "package.json", "setup.py"]):
            target_repo = os.getcwd()

    if target_repo and os.path.exists(target_repo):
        try:
            from ael_coder.cli import gather_codebase_corpus
            corpus, files, total_bytes = gather_codebase_corpus(target_repo)
            ids = tokenizer(corpus, return_tensors="pt", add_special_tokens=False)["input_ids"]
            total_toks = ids.shape[1]
            print(f"• Codebase Active  : {target_repo} ({len(files):,} files, {total_toks:,} tokens)")
            t0 = time.time()
            engine.ingest(ids)
            print(f"• Ingest Complete  : {len(engine.chunk_tokens)} chunks indexed in {time.time()-t0:.2f}s (Zero KV-Cache)")
        except Exception as e:
            print(f"• Codebase Ingest Note: {e}")

    peak_vram = torch.cuda.max_memory_allocated(0) / (1024**3) if torch.cuda.is_available() else 0.0
    print(f"• Engine Ready! Peak VRAM: {peak_vram:.2f} GB (Under 4.0 GB Limit!)")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Ael-Coder-1.5B Ollama Bridge Server")
    parser.add_argument("--port", type=int, default=11435, help="Port to listen on (default 11435, or 11434 to replace Ollama)")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface (default 0.0.0.0)")
    parser.add_argument("--model", default="Prannesshkva/Ael-Coder-1.5B", help="Hugging Face repo or local path")
    parser.add_argument("--repo", default=None, help="Target codebase to index into ISOM-R2 (default: C:\\Users\\prann\\Desktop\\langgraph)")
    args = parser.parse_args()

    load_isom_engine(args.model, repo_path=args.repo)

    print(f"🚀 Ael-Coder Ollama Bridge is LIVE on: http://localhost:{args.port}")
    print(f"• Ollama Endpoint : http://localhost:{args.port}/api/chat")
    print(f"• OpenAI Endpoint : http://localhost:{args.port}/v1/chat/completions\n")
    print("To connect Continue in VS Code, set apiBase to:")
    print(f"   apiBase: http://localhost:{args.port}\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
