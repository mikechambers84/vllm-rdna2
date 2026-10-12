<!-- markdownlint-disable MD001 MD041 -->
# vLLM for AMD gfx1030 (Radeon Pro V620 / RDNA2)

This is a fork of [vLLM](https://github.com/vllm-project/vllm) that runs well on
AMD **gfx1030** GPUs: Radeon Pro V620 and other Navi 21 cards (RX 6800 / 6800 XT /
6900 XT). Upstream vLLM's ROCm support targets CDNA (MI-series) and RDNA3/4. On
gfx1030 it misdetects the GPU, depends on instructions the chip lacks (MFMA/WMMA,
FP8, bf16 dot products), and runs most hot paths on slow fallbacks.

This fork fixes the platform layer and adds hand-written HIP kernels built on
what gfx1030 does have: `v_dot2_f32_f16`, `v_dot4_i32_i8`, wave32 and 64 KB of
LDS. It is based on upstream commit `6e517b15c1`. Everything was developed and
tested on a host with 8x Radeon Pro V620 (32 GB each, PCIe only, no XGMI).

### What has been added and optimized

#### Platform and runtime

- gfx1030 is detected correctly, including on mixed AMD + NVIDIA hosts.
  Capability checks no longer mistake its compute capability (10.x) for NVIDIA
  Blackwell.
- fp16 is the native compute type. `--dtype auto` runs bf16 checkpoints in fp16,
  and `--dtype bfloat16` also works, through the fp16 kernels (within 1-8% of fp16).
  Triton's bf16 multiply lowering is rewritten for gfx10.
- Full and piecewise HIP graphs, pipeline parallelism across PCIe cards, and
  speculative decoding (MTP) on the last pipeline stage.
- The AOT compile cache is keyed on tensor layouts, so weight layouts chosen at
  load time never reuse stale compiled artifacts.

#### Dense GEMM / GEMV kernels (`csrc/rocm/*_rdna2.cu`)

- A skinny fp16 GEMV for decode and the `lm_head`, ported from
  [yiminyuan/vllm](https://github.com/yiminyuan/vllm) (Yimin Yuan). It reaches
  about 500 GB/s, against about 100 GB/s for rocBLAS.
- 4-bit GPTQ/AWQ (W4A16) GEMM on Exllama's tensors, up to 512 rows, with an
  fp32-accumulated prefill path.
- `gemm_w8_rdna2`: one K-major kernel for FP8 (weight-only, exact e4m3 to fp16
  widening), W8A16, W8A8 int8 (`v_dot4`, bit-exact with `triton_scaled_mm`) and,
  opt-in, unquantized fp16 weights. It supports per-tensor, per-channel and
  block scales.
- W8A8 int8 GEMV and GEMM, an opt-in W4A8 prefill path for 4-bit layers, and a
  shipped table of tuned rocBLAS solutions for fp16 GEMMs.

#### Mixture of experts

- Fused-MoE decode kernels for int4 (group 32 / 128, with or without zero
  points) and int8 experts, with gate/up, SiLU and the top-k sum fused, and
  grouped top-k routing in a single kernel.
- Grouped W4A16 / W8A8 prefill GEMMs, skinny grouped GEMMs for a few rows per
  expert, a K-split GEMM for large experts (16-row tiles that skip padding
  rows), and a W4A8 MoE prefill (opt-in).
- An FP8 MoE backend, and MoE tile sizes chosen by tokens per expert (no
  per-model config files).

#### Attention and linear attention

- HIP paged prefill attention (head 64 / 128 / 256, GQA packing, sliding window,
  softcap, sinks) at 2.5-4.6x Triton's speed.
- Split-KV decode and speculative-verify attention, and fp8 KV cache support.
- A HIP chunked gated-delta (GDN) pipeline for Qwen3.x linear-attention layers.
- Model-specific kernels:
    - Qwen3.8-Flash-Next: QSA sparse attention, PLE short conv and the
      hyper-connection gate.
    - GLM-5.3-Flash: mHC/Sinkhorn, rope-free sparse MLA attention, sparse
      indexer logits for prefill and decode, and a batched GEMV for MLA weight
      absorption.

#### Speculative decoding and memory

- MTP drafters share the target's embedding and `lm_head` instead of holding
  vocabulary-sized copies.
- Opt-in: limit the draft logits to a prefix of the vocabulary, and keep input
  embeddings in host memory (UVA).

#### GLM-5.3-Flash fixes

- Long prompts: the sparse-attention indexer read and wrote the wrong cache
  pages whenever the hybrid model's KV block was larger than the indexer's
  page (prompts past 2048 tokens attended to the wrong keys, and could fault).
  On gfx1030 the decode-side logits kernel also read the cache layout wrongly.
  Both are fixed: 24K-token text NLL 0.84 -> 0.61, and a code hidden in 32K
  tokens is retrieved exactly.
- MTP under pipeline parallelism: the drafter's input embedding was never
  loaded (acceptance 24% -> 90%).
- Vision under pipeline parallelism: the vision tower is built only on the
  first stage, which runs it, instead of on every stage (about 1.1 GiB of KV
  cache per stage).

### Tested models

The figures below were measured on V620s with 120-140 W power caps. Prefill is
one 8192-token prompt with one output token (time to first token). Blank cells
were not measured.

| Model | GPUs | Decode, batch 1 | 8K-token prefill | Other |
| --- | --- | --- | --- | --- |
| Qwen3.8-27B W4A16 | 1 | 38.8 ms/token | 23.2 s, 353 tok/s; 11.8 s, 694 tok/s with the W4A8 opt-in | throughput 321 tok/s, MTP chat 42 tok/s; 3.8-5.1x stock vLLM |
| Qwen3.8-27B W8A8 | 1 | | 11.7 s, 700 tok/s | throughput 454 tok/s; 1.8-2.1x stock vLLM |
| Qwen3.6-35B-A3B W4A16 | 1 | 14.3 ms/token | 3.0 s, 2730 tok/s; 2.5 s, 3270 tok/s with the W4A8 opt-in | throughput 1738 tok/s, 1970 with the K-major opt-in; MTP chat 120 tok/s |
| Qwen3.6-35B-A3B W8A8 | 2 (PP=2) | | 8K prompt + 256 output tokens: 6.7 s | throughput 2462 tok/s |
| Qwen3.8-Flash-Next INT4 (fp8 KV, int8 / W4A8 opt-ins) | 4 (PP=4) | 21.5 ms/token; 12.8 ms/token with MTP k=2 | 2.74 s, 2990 tok/s | batch-8 decode 34.2 ms/step |
| GLM-5.3-Flash W4A16 | 8 (PP=8) | 30.9 ms/token, 32 tok/s with MTP k=2 | 4.95 s, 1650 tok/s | 32K / 64K prefill 17.8 / 35.4 s |

### Building

The build uses the ROCm 7.14 SDK from AMD's Python wheels, not a system ROCm
install. You need Python 3.12, [uv](https://docs.astral.sh/uv/) and the
`amdgpu` kernel driver.

```bash
git clone https://github.com/mikechambers84/vllm-rdna2.git
cd vllm-rdna2
uv venv --python 3.12
source .venv/bin/activate

# torch 2.12 + ROCm 7.14 for gfx1030, Triton, and the ROCm SDK used to compile.
echo "triton==3.7.1+git0263a6a6.rocm7.14.0" > /tmp/vllm-overrides.txt
uv pip install --index-strategy unsafe-best-match \
    --extra-index-url https://repo.amd.com/rocm/whl-multi-arch/ \
    --override /tmp/vllm-overrides.txt \
    -r requirements/build/rocm.txt -r requirements/rocm.txt \
    "amd-torch-device-gfx1030==2.12.0+rocm7.14.0" "rocm[devel]==7.14.0"

# Point the build at the SDK inside the venv. Set these in every shell that
# builds or runs vLLM.
SP=$PWD/.venv/lib/python3.12/site-packages
export ROCM_PATH=$SP/_rocm_sdk_devel ROCM_HOME=$SP/_rocm_sdk_devel
export HIP_DEVICE_LIB_PATH=$SP/_rocm_sdk_core/lib/llvm/amdgcn/bitcode
export PATH=$ROCM_PATH/bin:$ROCM_PATH/llvm/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$SP/_rocm_sdk_core/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export CMAKE_PREFIX_PATH=$ROCM_PATH/lib/cmake:$SP/torch/share/cmake
export PYTORCH_ROCM_ARCH=gfx1030 VLLM_TARGET_DEVICE=rocm

# Without build isolation, so the build uses the torch installed above.
uv pip install -e . --no-build-isolation --no-deps
```

After you change C++/HIP code, rebuild incrementally with CMake, as described in
[docs/contributing/incremental_build.md](docs/contributing/incremental_build.md).
`cmake --build <build dir> --target install` recompiles only the changed kernels.

### Running

Use vLLM as usual. Pass `--dtype float16`, and use pipeline parallelism for
models larger than one card. For example, GLM-5.3-Flash on eight cards with
MTP, tool calling, reasoning, images and video (the GLM-5.3-Flash
figures above use these settings):

```bash
VLLM_ROCM_KMAJOR_UNQUANTIZED=1 VLLM_PP_LAYER_PARTITION=7,6,5,5,6,6,6,4 \
vllm serve /path/to/GLM-5.3-Flash-W4A16 \
    --dtype float16 --pipeline-parallel-size 8 \
    --max-model-len 262144 --max-num-batched-tokens 384 --max-num-seqs 32 \
    --gpu-memory-utilization 0.95 \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}' \
    --tool-call-parser glm47 --enable-auto-tool-choice --reasoning-parser glm47
```

`VLLM_PP_LAYER_PARTITION` balances the stages: the MTP layer adds about 5 GiB
to the last one, and the first carries the three dense layers. Smaller prefill
chunks (`--max-num-batched-tokens` 384) keep all eight stages busy on a long
prompt.

The gfx1030 kernels are selected automatically. These environment variables
control the opt-in paths, which trade memory or accuracy for speed:

| Variable | Default | Effect |
| --- | --- | --- |
| `VLLM_ROCM_KMAJOR_UNQUANTIZED` | 0 | Store unquantized fp16 linear layers and an untied `lm_head` K-major for `gemm_w8_rdna2` (faster batched decode / spec verify; used for the GLM numbers above) |
| `VLLM_ROCM_W8A16_UNQUANTIZED` | 0 | Store unquantized linear layers (and unquantized MoE experts) as int8 weight-only: half the decode bandwidth and memory |
| `VLLM_ROCM_W8A16_LM_HEAD` | 0 | Store an untied `lm_head` as int8 per-row (about 6-7% of decode time on 27B-35B models; the logits change slightly) |
| `VLLM_ROCM_W4A8_PREFILL` | 0 | Run 4-bit layers' and int4 experts' prefill GEMMs as W4A8 (about 2x faster prefill GEMMs, NLL +1-3%) |
| `VLLM_ROCM_W4A8_MIN_ROWS` | 128 | Row threshold for `VLLM_ROCM_W4A8_PREFILL` |
| `VLLM_ROCM_RDNA2_ATTENTION` | 1 | HIP prefill attention instead of Triton |
| `VLLM_ROCM_RDNA2_GEMM_TUNE` | 0 | Tune rocBLAS solutions for weight shapes missing from the table while loading (about 30 s per shape, once) |
| `VLLM_ROCM_RDNA2_GEMM_TABLE` | shipped | Path to the tuned rocBLAS table; `0` uses rocBLAS's own choices |
| `VLLM_MTP_DRAFT_VOCAB_SIZE` | 0 | MTP drafters compute logits only for the first N token ids (cheaper drafts, possibly lower acceptance) |
| `VLLM_UVA_INPUT_EMBEDDINGS` | 0 | Keep untied input embeddings in pinned host memory to free device memory for the KV cache or a drafter |

Not supported on gfx1030: FP8 compute (FP8 checkpoints run weight-only),
TileLang, and aiter kernels. GLM-5.3-Flash has been tested with text, images
and video.

---

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
Easy, fast, and cheap LLM serving for everyone
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

🔥 We have built a vLLM website to help you get started with vLLM. Please visit [vllm.ai](https://vllm.ai) to learn more.
For events, please visit [vllm.ai/events](https://vllm.ai/events) to join us.

---

## About

vLLM is a fast and easy-to-use library for LLM inference and serving.

Originally developed in the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley, vLLM has grown into one of the most active open-source AI projects built and maintained by a diverse community of many dozens of academic institutions and companies from over 2000 contributors.

vLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with [**PagedAttention**](https://blog.vllm.ai/2023/06/20/vllm.html)
- Continuous batching of incoming requests, chunked prefill, prefix caching
- Fast and flexible model execution with piecewise and full CUDA/HIP graphs
- Quantization: FP8, MXFP8/MXFP4, NVFP4, INT8, INT4, GPTQ/AWQ, GGUF, compressed-tensors, ModelOpt, TorchAO, and [more](https://docs.vllm.ai/en/latest/features/quantization/index.html)
- Optimized attention kernels including FlashAttention, FlashInfer, TRTLLM-GEN, FlashMLA, and Triton
- Optimized GEMM/MoE kernels for various precisions using CUTLASS, TRTLLM-GEN, CuTeDSL
- Speculative decoding including n-gram, suffix, EAGLE, DFlash
- Automatic kernel generation and graph-level transformations using torch.compile
- Disaggregated prefill, decode, and encode

vLLM is flexible and easy to use with:

- Seamless integration with popular Hugging Face models
- High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more
- Tensor, pipeline, data, expert, and context parallelism for distributed inference
- Streaming outputs
- Generation of structured outputs using xgrammar or guidance
- Tool calling and reasoning parsers
- OpenAI-compatible API server, plus Anthropic Messages API and gRPC support
- Efficient multi-LoRA support for dense and MoE layers
- Support for NVIDIA GPUs, AMD GPUs, Intel GPUs, and x86/ARM/PowerPC CPUs. Additionally, diverse hardware plugins such as Google TPUs, Intel Gaudi, IBM Spyre, Huawei Ascend, Rebellions NPU, Apple Silicon, MetaX GPU, and more.

vLLM seamlessly supports 200+ model architectures on Hugging Face, including:

- Decoder-only LLMs (e.g., Llama, Qwen, Gemma)
- Mixture-of-Expert LLMs (e.g., Mixtral, DeepSeek-V3, Qwen-MoE, GPT-OSS)
- Hybrid attention and state-space models (e.g., Mamba, Qwen3.5)
- Multi-modal models (e.g., LLaVA, Qwen-VL, Pixtral)
- Embedding and retrieval models (e.g., E5-Mistral, GTE, ColBERT)
- Reward and classification models (e.g., Qwen-Math)

Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Getting Started

Install vLLM with [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`:

```bash
uv pip install vllm
```

Or [build from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source) for development.

Visit our [documentation](https://docs.vllm.ai/en/latest/) to learn more.

- [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
- [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Contributing

We welcome and value any contributions and collaborations.
Please check out [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) for how to get involved.

## Citation

If you use vLLM for your research, please cite our [paper](https://arxiv.org/abs/2309.06180):

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## Contact Us

<!-- --8<-- [start:contact-us] -->
- For technical questions and feature requests, please use GitHub [Issues](https://github.com/vllm-project/vllm/issues)
- For discussing with fellow users, please use the [vLLM Forum](https://discuss.vllm.ai)
- For coordinating contributions and development, please use [Slack](https://slack.vllm.ai)
- For security disclosures, please use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature
- For collaborations and partnerships, please contact us at [collaboration@vllm.ai](mailto:collaboration@vllm.ai)
<!-- --8<-- [end:contact-us] -->

## Media Kit

- If you wish to use vLLM's logo, please refer to [our media kit repo](https://github.com/vllm-project/media-kit)
