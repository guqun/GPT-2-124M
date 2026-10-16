# GPT-2 124M Learning Lab

[English](README.en.md) | [中文](README.md)

A hands-on study project for understanding GPT-2 through model inspection, an independent PyTorch implementation, and numerical comparisons against Hugging Face Transformers.

The learning sequence follows Andrej Karpathy's [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU) and [build-nanogpt](https://github.com/karpathy/build-nanogpt). This repository adapts that material into small lessons with detailed comments, diagrams, and reproducible experiments. The stage-by-stage comparison harness is an additional teaching exercise.

Repository: [guqun/GPT-2-124M](https://github.com/guqun/GPT-2-124M).

## Current progress

The first two learning blocks of section 2.1 are complete:

1. **Inspect a pretrained checkpoint:** distinguish configuration, model code, trained weights, and tokenizer files; inspect tensor names, shapes, parameter counts, and weight sharing.
2. **Implement and verify the forward pass:** build GPT-2 with PyTorch, import the same pretrained weights used by the reference model, compare intermediate outputs, and check causal masking.

The next learning block is text generation with our own model. Generation, the training loop, single-batch overfitting, data loading, and a focused initialization lesson remain ahead. The current experiments use existing pretrained weights; this repository has not yet reproduced GPT-2 pretraining.

## Project guide

| File or directory | Purpose |
| --- | --- |
| [inspect_checkpoint.py](inspect_checkpoint.py) | Inspect a locally cached GPT-2 configuration and pretrained weights using the Hugging Face model. |
| [gpt2_model.py](gpt2_model.py) | Independent GPT-2 architecture, forward computation, and Hugging Face weight import. |
| [compare_forward.py](compare_forward.py) | Run both implementations on identical inputs, compare intermediate tensors, and check causality. |
| [lesson02.md](lesson02.md) | Step-by-step instructions for the forward comparison lesson, in Chinese. |
| [docs/lesson01-02-notes.md](docs/lesson01-02-notes.md) | Consolidated learning notes, questions, explanations, and self-checks, in Chinese. |
| [docs/diagrams/](docs/diagrams/) | Diagram mapping Python calls to the mathematical operations. |
| [artifacts/](artifacts/) | Saved inspection results, comparison reports, and learning-note archives. |
| [python_foundations/](python_foundations/README.md) | Python and PyTorch reading material for experienced Java, C++, and Dart developers. |

## Model and data flow

The implemented model uses the GPT-2 124M configuration:

| Property | Value |
| --- | ---: |
| Vocabulary size | 50,257 |
| Maximum sequence length | 1,024 |
| Hidden dimension | 768 |
| Transformer blocks | 12 |
| Attention heads per block | 12 |
| Dimension per head | 64 |
| MLP intermediate dimension | 3,072 |
| Unique trainable parameters | 124,439,808 |

```text
Token IDs [B, T]
    │
    ├─ Token embedding lookup
    └─ Position embedding lookup
              │ add, then embedding dropout
              ▼
         [B, T, 768]
              │
         12 Transformer blocks
           r = x + Attention(LN1(x))
           x = r + MLP(LN2(r))
              │
         Final LayerNorm
              │
         lm_head (shares the token embedding weight)
              ▼
         Logits [B, T, 50257]
```

`B` is the batch size and `T` is the number of token positions. Each block has its own parameters. Attention projects Q/K/V, splits them into heads, applies scaled dot-product attention with a causal mask, concatenates the head outputs, and projects back to 768 dimensions. The MLP expands 768 → 3,072, applies the GPT-2 GELU approximation, and projects back to 768.

Our code defines this computation using PyTorch tensors and layers; it does not call Hugging Face's `GPT2Block` for its forward pass. The Hugging Face model is the separate reference implementation. Its `Conv1D` projection weights are transposed when copied into our `nn.Linear` layers.

The [call map and reading guide](docs/diagrams/gpt2-forward-call-map.md) connect `GPT.forward`, `Block.forward`, and the Attention/MLP calls. The diagram and source comments are currently in Chinese.

![GPT-2 code calls and computation flow](docs/diagrams/gpt2-forward-call-map.png)

## Run the experiments

### Environment and checkpoint

Use a Python environment with PyTorch, Transformers, and safetensors installed. The saved forward comparison was run with **Python 3.9.6, PyTorch 2.2.2, and Transformers 4.57.6**, on **CPU with float32**. These are the recorded versions, rather than a claim of compatibility with every newer release.

Both scripts read an existing local checkpoint. They do not download weights. The model directory must contain at least:

```text
config.json
model.safetensors
```

By default, they use this fixed Hugging Face cache snapshot:

```text
~/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e
```

Use `--model-dir` if the files are stored elsewhere. Model weights and virtual environments are excluded from this Git repository.

### Inspect the checkpoint

From the repository directory, using the Python interpreter for your environment:

```bash
python inspect_checkpoint.py --model-dir /path/to/gpt2-snapshot
```

Useful options:

- `--all`: list every unique named parameter.
- `--token-id 17250`: inspect a different row of the token embedding table.
- `--json /tmp/gpt2-checkpoint-report.json`: save the inspection results as JSON.

The existing [inspection report](artifacts/lesson01-report.json) records configuration values, shapes, example parameter values, parameter counts, and embedding/output weight sharing.

### Compare the forward pass

```bash
python compare_forward.py --model-dir /path/to/gpt2-snapshot
```

The script loads identical pretrained weights into both implementations, switches both to evaluation mode, and runs under `torch.inference_mode()`. The reference uses eager attention with its KV cache disabled. Inputs are fixed token ID sequences with equal lengths within each batch; padding is not part of this experiment.

Results are printed to the terminal and saved to `artifacts/lesson02-report.json`, replacing the previous report at that path. Use `--report /tmp/gpt2-forward-report.json` to keep the checked-in report unchanged. A failed comparison produces a nonzero exit status.

### Existing local VS Code setup

For the original learning workspace, select this interpreter with **Python: Select Interpreter**:

```text
/Users/guqun/micrograd/venv/bin/python
```

Then open either script and choose **Run Python File**. If you cloned the repository elsewhere, select your own environment's interpreter and pass `--model-dir` as needed. The scripts locate their default paths independently of the terminal's working directory.

## Recorded verification

The checked-in [forward report](artifacts/lesson02-report.json) contains three input cases: two four-token sequences, one single-token sequence, and five repetitions of the same token ID at different positions.

- **45 stage comparisons passed:** embedding output, all 12 blocks, final LayerNorm, and logits for each case.
- The comparison checks matching shapes, finite values, and the elementwise condition `abs(actual - reference) <= 1e-4 + 1e-4 * abs(reference)`.
- The largest recorded final-logit absolute difference was approximately **3.81 × 10⁻⁵**.
- Changing the final input token left earlier-position logits unchanged in the causal check.
- The report also confirms **124,439,808** unique parameters, shared token embedding/output weights, and distinct QKV parameter objects in the first two blocks.

These checks establish forward agreement for the recorded cases. They do not validate a training loop or generation loop, which are later lessons.

## Python supplement and review

The [Python foundations material](python_foundations/README.md) contains 17 units for developers already comfortable with Java, C++, or Dart. It focuses on Python semantics and the Tensor, autograd, and `nn.Module` concepts needed to read model code.

- [Markdown textbook](python_foundations/textbook.md), readable on GitHub.
- [Interactive HTML textbook](python_foundations/textbook.html), opened locally in a browser after cloning.
- [Runnable examples](python_foundations/examples/), a [configuration exercise](python_foundations/exercises/config_report.py), and its [reference solution](python_foundations/answers/config_report.py).

For the model lessons, return to the [consolidated notes](docs/lesson01-02-notes.md) and trace the same input from the checkpoint through the Python calls to the final logits.
