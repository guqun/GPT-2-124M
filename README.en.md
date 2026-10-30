# GPT-2 124M Learning Lab

[English](README.en.md) | [中文](README.md)

A hands-on study project for understanding GPT-2 through model inspection, an independent PyTorch implementation, and numerical comparisons against Hugging Face Transformers.

The learning sequence follows Andrej Karpathy's [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU) and [build-nanogpt](https://github.com/karpathy/build-nanogpt). This repository adapts that material into small lessons with detailed comments, diagrams, and reproducible experiments. The stage-by-stage comparison harness is an additional teaching exercise.

Repository: [guqun/GPT-2-124M](https://github.com/guqun/GPT-2-124M).

## Current progress

**Current learning: Stage 2 → Unit 2 → Section 2.5, block 15: periodic HellaSwag evaluation (2026-10-30).** The learner reports initial understanding of scoring, padding, masks, answer comparison and accuracy. A new entry point now connects evaluation to training; section 2.5 is not marked complete.

New [train_hellaswag.py](train_hellaswag.py), [hellaswag_eval.py](hellaswag_eval.py) and [Chinese lesson](lesson15.md) preserve all previous Python code. Training still starts from random parameters and uses the earlier shard loaders, accumulation, validation and generation. Evaluation runs initially and after updates 2 and 3 by default, without changing parameters or training state.

The download retry succeeded: the repository includes the first **8 real HellaSwag validation examples**, with source URL and checksum. This is a preview, not a full benchmark. No network is required at runtime. Pass a local JSONL file with `--hellaswag-file`, and use `--hellaswag-limit 0` to evaluate the entire file. CPU tiny execution and scoring/state-isolation checks passed; full-dataset and CUDA/NCCL runs remain pending.

```bash
/Users/guqun/micrograd/venv/bin/python /Users/guqun/micrograd/gpt2_pretrain_lab/train_hellaswag.py --device cpu --model-size tiny
```


New [train_shards.py](train_shards.py) and the [Chinese lesson](lesson12-shard-training.md) preserve all prior code. Two separate loaders supply train and validation data. Thirteen related checks and tiny CPU single-process / real two-process Gloo short runs passed; all 29 previous Python files remain unchanged. The default data consists of two distinct teaching sentences, not FineWeb. Real FineWeb and CUDA/NCCL runs, and learner practice with the new entry point, remain pending. The Notion master plan stays at October 27; section 2.4 remains incomplete, with counts unchanged at 1/5 and 5/32.

The existing [data-pipeline notes](docs/fineweb-edu-sharding-notes.md) and [same Notion page](https://app.notion.com/p/404eaea31414818b9df0c8ab083cdbe0) were expanded on October 30. Section 13 connects raw text, EOT/token IDs, npy shards, CPU/device transfers, shifted x/y, prediction positions, microbatch accumulation, parameter updates, and return paths from validation/generation. The local note includes the full code flow diagram; archiving these notes does not mark the course section complete.

The [periodic-generation review](docs/periodic-generation-notes.md) ([Notion](https://app.notion.com/p/406eaea31414814ea785fee733047ffc)) and [full-model learner report](artifacts/lesson14-user-gpt2-cpu-report.json) remain available. Learner multi-process practice, CUDA/NCCL, and earlier performance work remain pending.

To revisit periodic generation with the already approved local environment:

```bash
/Users/guqun/micrograd/venv/bin/python /Users/guqun/micrograd/gpt2_pretrain_lab/train_generation.py --device cpu --model-size tiny
```

The earlier [tiny CPU report](artifacts/lesson14-cpu-report.json) and the [full GPT-2 learner report](artifacts/lesson14-user-gpt2-cpu-report.json) each record generation after updates 2 and 3 and match their respective lesson 13 training/validation records. In the learner run, 94 of 96 new tokens after update 2 were newlines; 64 of 96 after update 3 were commas. This confirms the intended observation flow, not text quality or CUDA performance. The diagram keeps its labeled tiny example. See the [course alignment record](artifacts/lesson14-course-alignment.json).

The new [periodic-generation code flow diagram](docs/diagrams/gpt2-periodic-generation-flow.md) ([PNG](docs/diagrams/gpt2-periodic-generation-flow.png) / [SVG](docs/diagrams/gpt2-periodic-generation-flow.svg), Chinese) maps update conditions, the G1–G6 generation loop, tensor shapes, independent sampling RNG, mode restoration, and return to training. Source lines and numerical examples are checked against the code and archived CPU report.

The [validation code flow diagram](docs/diagrams/gpt2-validation-loop-flow.md) ([PNG](docs/diagrams/gpt2-validation-loop-flow.png) / [SVG](docs/diagrams/gpt2-validation-loop-flow.svg), Chinese) maps the training loop, validation calls, return points, and recorded CPU execution order. Its renderer reads source anchors and report values directly.

Stage 1 covers the previously completed neural-network and Transformer foundations. Within Stage 2, Unit 1 (tokenization) has covered all four planned sections and delivered its planned experiments; Unit 2 has completed section 2.1, while 2.2 has covered its theory with experiments still pending. The completed-section counts remain **1/5 for Unit 2** and **5/32 for Stage 2**, with theory progress recorded separately. Units 3–7 have not started. Independent mastery is recorded separately from content coverage and verified code.

The first two learning blocks of Stage 2 → Unit 2 → Section 2.1 have been covered and their experiments verified:

1. **Inspect a pretrained checkpoint:** distinguish configuration, model code, trained weights, and tokenizer files; inspect tensor names, shapes, parameter counts, and weight sharing.
2. **Implement and verify the forward pass:** build GPT-2 with PyTorch, import the same pretrained weights used by the reference model, compare intermediate outputs, and check causal masking.

The third block, text generation with our own model, has been introduced and run. On 2026-10-18, the learner reported a basic understanding; self-checks remain available for review. The fourth block's Tiny Shakespeare single-batch experiment has been verified and discussed as a review of Stage 1 foundations. Block 5 has now covered sequential sampling, shifted targets, predictions at every position, causal masking, the batch dimension, and cursor wraparound. The learner reported a basic understanding; its code, lesson, and flow diagram are archived. The learner now reports a basic understanding of block 6 (weight tying); their independent training run is not confirmed. The learner reports a basic understanding of block 7 (parameter initialization). After reviewing the seven blocks and their experiments, the learner confirmed section 2.1 complete and moved to 2.2. Self-checks are review material rather than an additional completion requirement. Each block has a separate script and lesson, preserving earlier materials. Forward comparison and generation use pretrained weights; these training experiments start with random weights. Full GPT-2 pretraining remains ahead.

The [learning-plan review](docs/learning-plan-review.md) distinguishes original course content, the agreed roadmap, local adaptations, and evidence of learner progress. The [Notion plan](https://app.notion.com/p/340eaea31414819197bde537d339d0fc) was synchronized on 2026-10-27 through periodic-generation understanding, the learner’s full GPT-2 CPU run, and the next HellaSwag topic. The [Notion validation note](https://app.notion.com/p/405eaea3141481c98d06ce42da3d9212) includes one complete flow diagram. Dated records retain their historical meaning. Lessons introduce the problem, purpose, and expected learning before examples and code; the [section 2.1 review](docs/section-2.1-review.md) separates coverage, verified experiments, and independent mastery.

The existing [AdamW and bias-correction notes](docs/adamw-bias-correction-notes.md) ([Notion](https://app.notion.com/p/400eaea3141481a39db3f37c52896a35), Chinese) retain the moment, geometric-series, normalization, expectation, clipping, and three-phase learning-rate explanations. Section 17 adds gradient accumulation: loss/K, four rounds of elementwise .grad accumulation, nested loops, and one final clipping/AdamW update. The flow PNG and SVG are archived. The learner’s October 26 single-process run exercised the integrated update path; independent configuration, DDP practice, and CUDA comparisons remain separate, so section 2.3 has not been marked complete.

The [GPU training review notes](docs/gpu-training-notes.md) (Chinese) collect devices, MPS, measurement, TF32, BF16 casts, gradients, and FP32 parameter updates, with the precision-flow diagram. A [Notion copy](https://app.notion.com/p/3feeaea3141481af8e6fd509bee27fc7) is archived.

The new [training optimization notes, part II](docs/gpu-training-optimization-notes.md) ([Notion](https://app.notion.com/p/3ffeaea3141481f9bd20ca4745fa8b80)) cover `torch.compile`, kernels, FlashAttention, vocabulary alignment, and extra-logit gradients. They include the flow image, equations, code connections, measured results, and a checked masking example. Masking is a supplementary teaching option and has not been integrated into full-model training.

The subsequent [FlashAttention flow diagram](docs/diagrams/gpt2-flash-attention-flow.md) is available as [PNG](docs/diagrams/gpt2-flash-attention-flow.png) and [SVG](docs/diagrams/gpt2-flash-attention-flow.svg), with Chinese labels. It compares full intermediate matrices with tiled accumulation, works through two blocks yielding `300 / 10 = 30`, and explains stable online normalization and backward recomputation. This illustrates the algorithm; it is not a CUDA performance measurement.

The [gradient accumulation flow diagram](docs/diagrams/gpt2-gradient-accumulation-flow.md) expands loss scaling and backward within each micro-batch, four rounds of elementwise 2×2 gradient accumulation, and one final clipping/AdamW update. [PNG](docs/diagrams/gpt2-gradient-accumulation-flow.png) and [SVG](docs/diagrams/gpt2-gradient-accumulation-flow.svg) have Chinese labels. The teaching example was checked against a full-batch calculation; existing training entry points and course completion counts are unchanged.

The [DDP / NCCL / All-Reduce flow diagram](docs/diagrams/gpt2-ddp-allreduce-flow.md) follows two GPU lanes through local gradients, Reduce-Scatter, All-Gather, averaging, and local optimizer updates. It distinguishes CPU orchestration from GPU computation. The [PNG](docs/diagrams/gpt2-ddp-allreduce-flow.png) and [SVG](docs/diagrams/gpt2-ddp-allreduce-flow.svg) use Chinese labels and illustrative numbers for the section 2.4 discussion, rather than measured CUDA results.

The [DDP lm_head worked example](docs/diagrams/gpt2-ddp-lm-head-example.md) connects that communication to a linear forward pass, softmax, cross-entropy, local weight gradients, averaging, and AdamW. The same `W[0,0]` stays highlighted throughout the [PNG](docs/diagrams/gpt2-ddp-lm-head-example.png) and [SVG](docs/diagrams/gpt2-ddp-lm-head-example.svg). This isolates an untied 2×2 output head with fixed features; CPU/float64 calculations match a joint-batch gradient and update. Full GPT-2 weight-tying contributions are explained separately.

The [DDP, mean-loss, and gradient-synchronization notes](docs/ddp-batch-gradient-notes.md) ([Notion](https://app.notion.com/p/402eaea31414814c80bad0fa8f02bdd7), Chinese) connect GPU data ownership, collective communication, full-batch matrix derivatives, and cross-device averaging. They address why each scalar parameter has one gradient, why a batch objective is defined, and how the mean factor propagates once through the computation graph. Existing diagrams and numerical checks are included. After this October 23 archive, the Notion roadmap was separately synchronized to the discussion progress; training behavior and completion counts were unchanged. Subsequent block 11 checks and block 12 discussions are recorded below.

## Learning block 13: Validation during training

[train_validation.py](train_validation.py) and [lesson13.md](lesson13.md) preserve earlier entry points. Read `evaluate_validation()` (V1–V6), then `train_with_validation()` (T1–T3): separate data/cursors, eval/no-grad, averaging over batches and ranks, and restoration of training mode. The author's historical `21d3d32` implementation is the reference; local data and CPU/Gloo adaptations are explicit.

```bash
.venv/bin/python train_validation.py --device cpu --model-size tiny
```

On the current machine use the authorized existing interpreter `/Users/guqun/micrograd/venv/bin/python`. Default checks follow 0, 2, and 3 updates, using two batches per process. Metrics are printed and saved as JSON; no early stopping, best-checkpoint selection, or metric-driven learning rate is added. Five behavior checks and single-/two-process CPU runs passed. FineWeb-Edu and CUDA runs remain pending.

## Section 2.4 practice III: connecting shards to training

Read [train_shards.py](train_shards.py): M1–M4 sets up the run; T1–T6 combines updates, validation, and generation; V1–V4 evaluates fixed batches across validation files. Shard rollover preserves model parameters, optimizer history, and the learning-rate schedule. Each microbatch stays in one file, while multiple microbatches in one optimizer update may use different files.

```bash
/Users/guqun/micrograd/venv/bin/python /Users/guqun/micrograd/gpt2_pretrain_lab/train_shards.py --device cpu --model-size tiny
```

See the [lesson](lesson12-shard-training.md), [CPU report](artifacts/lesson12-shard-training-cpu.json), [two-process Gloo report](artifacts/lesson12-shard-training-ddp-cpu.json), and [source/check record](artifacts/lesson12-shard-training-alignment.json). `--data-dir` accepts existing GPT-2 uint16 shards; omitted, it creates a separate teaching demo. The archived runs verify control flow, not language quality or completed pretraining.

The [code flow diagram (PNG)](docs/diagrams/gpt2-shard-training-flow.png), [scalable SVG](docs/diagrams/gpt2-shard-training-flow.svg), and [Chinese reading guide](docs/diagrams/gpt2-shard-training-flow.md) connect shard reads, microbatch accumulation, AdamW updates, validation, and generation. Source line references and the three-update file/cursor trace come from the implementation and archived CPU run.

## Block 12 practice II: reading shards and forming batches

New [load_shards.py](load_shards.py) and [reading guide](lesson12-shard-batches.md) preserve previous preparation and training code. Read S1–S7 for disk loading, split selection, B×T+1 slicing, shifted x/y, cursor advancement, and shard rollover.

```bash
python load_shards.py
```

The default repeats the previous two-sentence demo five times and writes separate 16/16/8-token shards. The train split uses the latter two files; four B=2/T=3 batches exercise within-shard advancement, switching, and wrapping. See the [CPU report](artifacts/lesson12-shard-batches-cpu.json) and [source/adaptation record](artifacts/lesson12-shard-loader-alignment.json). Five behavior tests passed, including two rank-specific loader objects compared with one combined loader. No model training, actual DDP processes, GPU verification, or real FineWeb run is claimed. This lesson has started; completion counts and Notion remain unchanged.

## Block 12 practice: preparing token shards

Read [lesson12-data-preparation.md](lesson12-data-preparation.md), then P1–P5 in [prepare_fineweb.py](prepare_fineweb.py). The script uses GPT-2 tiktoken, prepends EOT per document, stores uint16 IDs, and writes fixed-capacity `.npy` files including a short final shard. No model is constructed.

The [code flow diagram](docs/diagrams/gpt2-data-preparation-flow.md) ([PNG](docs/diagrams/gpt2-data-preparation-flow.png) / [SVG](docs/diagrams/gpt2-data-preparation-flow.svg)) connects main, lazy per-document encoding, both loops, full/tail shard writes, and the three actual copy iterations for the eight-ID demo. Source line numbers and archived report values are checked when rebuilding the image.

```bash
python prepare_fineweb.py
```

The default is explicitly a two-sentence teaching demo. `--source fineweb-preview` requests 20 real records and uses 2048-token shards; that network path has not succeeded on this machine. Each run uses a new output directory. See [optional dependencies](requirements-data.txt) and the [demo report](artifacts/lesson12-data-preparation-demo.json). The subsequent [integration lesson](lesson12-shard-training.md) now connects shards to training.

## Learning block 12: FineWeb-Edu, sharding, and document boundaries

The [new notes](docs/fineweb-edu-sharding-notes.md) ([Notion](https://app.notion.com/p/404eaea31414818b9df0c8ab083cdbe0), Chinese) archive the October 25 discussion: why move beyond Tiny Shakespeare, offline tokenization and uint16 storage, documents versus shards versus sequences, EOT, document attention masks, and incomplete shard tails. A flow diagram, comparisons, small token examples, and review answers support the explanation. Sources include the video transcript around 3:09:48–3:18:58, historical commit 69cb21f, and the official dataset card; supplementary tradeoffs are labeled separately.

The learner chose to follow the original course: EOT with ordinary causal attention, and skipping an incomplete shard tail instead of filling a batch from the next shard. The core concepts are covered, with understanding still developing. The earlier discussion has since been followed by independent preparation, loading, and training-integration code. Real FineWeb-Edu runs remain pending; section 2.4 stays incomplete and completion counts are unchanged.

## Learning block 11: DDP, data allocation, and accumulation

[train_ddp.py](train_ddp.py) and [lesson11.md](lesson11.md) follow the video around 2:46:52–3:09:48 and historical commit ba2554a. Earlier entry points remain intact. Each process reads rank-aware token ranges, accumulates local gradients under `no_sync()`, synchronizes on the last backward, and performs one clipped/scheduled AdamW update. The lesson explicitly documents the public-API synchronization choice and a global-cursor wraparound correction.

```bash
.venv/bin/python -m torch.distributed.run \
  --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29511 \
  train_ddp.py --device cpu --preview-only
```

Replace `--preview-only` with `--model-size tiny` to train a small GPT on real Tiny Shakespeare tokens. Defaults are B=2/T=32 per process, 512 positions per optimizer update, and three updates; R=2 gives K=4. The default model size without this option is full GPT-2 124M. All models start with random weights. This entry uses eager execution and the existing explicit attention implementation, leaving the pending CUDA optimization comparisons separate.

Six behavior tests pass, including real two-process Gloo gradients, parameters, and AdamW moments compared against a single-process joint batch. [Small-model DDP](artifacts/lesson11-ddp-cpu-report.json), [full-model single-process smoke](artifacts/lesson11-gpt2-single-process-smoke.json), and [equivalence results](artifacts/lesson11-ddp-equivalence.json) are archived. These October 24 checks supersede the earlier notes' pending CPU implementation status; they do not establish CUDA/NCCL behavior, speedup, or learner mastery. FineWeb-Edu concepts are now covered in block 12; its implementation/run and the validation topic remain pending.

## Learning block 10: torch.compile

[train_compile.py](train_compile.py) and [lesson10.md](lesson10.md) follow [the video around 1:47:34–1:59:46](https://www.youtube.com/watch?v=l8pRSuU81PU&t=6454s) and [historical commit fb8bd6e](https://github.com/karpathy/build-nanogpt/commit/fb8bd6e). Wrap the model once after moving it to the device, then reuse block 9's training loop. The compiled entry references the same parameters; cross-entropy outside the model and AdamW remain eager.

With no arguments, the script selects CUDA when available, otherwise CPU; automatic precision is BF16 on supported CUDA devices and FP32 otherwise. MPS is excluded from automatic device selection for this lesson's compiler setup. Eager and compiled modes use the same selection rules and print their resolved configuration. Default batch dimensions are B=4/T=32; the example below uses a smaller B=1/T=8.

```bash
.venv/bin/python train_compile.py --device cpu --precision fp32 --execution compile --batch-size 1 --seq-len 8 --threads 1 --warmup-steps 3 --steps 5
```

On the current Mac, use `/Users/guqun/micrograd/venv/bin/python`. Change only `--execution` to `eager` for a matched comparison. Reports distinguish the first update, including compilation/cache loading, from measured updates after warmup. A semantically equivalent change from `enumerate(..., start=1)` to `enumerate(..., 1)` avoids a PyTorch 2.2.2 graph break. CPU checks do not establish CUDA speedup; CUDA comparisons remain pending.

All 31 tests passed. Full-model [eager](artifacts/lesson10-cpu-eager.json) and [compiled](artifacts/lesson10-cpu-compile.json) runs each completed eight updates with one CPU thread; measured means were about 849 and 846 ms per step, showing no clear speedup in this short experiment. A four-thread compiled backward stalled in a BLAS/OpenMP wait on this Intel Mac, so this lesson defaults to one CPU thread. The lesson documents the observation and startup cost.

Mac GPU training is available with `--device mps --execution eager --precision fp32`. A full-model [MPS smoke run](artifacts/lesson10-mps-eager-smoke.json) completed three small-batch updates. Eager execution still runs on the selected GPU; it disables compilation. MPS eager training and the pending CUDA compilation benchmark are separate experiments.

## Learning block 9: BF16 automatic mixed precision

[train_mixed_precision.py](train_mixed_precision.py) and [lesson09.md](lesson09.md) follow [the video around 1:39:28–1:47:34](https://www.youtube.com/watch?v=l8pRSuU81PU&t=5968s) and [historical commit 177e4cd](https://github.com/karpathy/build-nanogpt/commit/177e4cd5b4cc05df4bb637ed1a9e55911d6f1e2c). Forward computation and cross-entropy run inside `autocast`; backward and AdamW run outside it. Parameters retain FP32 storage while eligible operations and their outputs use BF16.

```bash
.venv/bin/python train_mixed_precision.py --device cpu --probe-only --batch-size 1 --seq-len 8
.venv/bin/python train_mixed_precision.py --device cpu --precision bf16 --batch-size 1 --seq-len 8 --warmup-steps 1 --steps 2
```

The current Mac uses `/Users/guqun/micrograd/venv/bin/python`. The first command observes both precisions on identical parameters and inputs without updates. The second runs three training updates. All 29 tests pass; full-model [dtype observations](artifacts/lesson09-cpu-dtypes.json) and [CPU training results](artifacts/lesson09-cpu-bf16-smoke.json) are archived. Parameters, parameter gradients, loss, and AdamW moments remain FP32; logits use BF16. These B=1/T=8 CPU checks establish behavior, not GPU speedup. Matched CUDA performance commands are included in the lesson; GPU measurements remain pending.

## Learning block 8: GPU execution, timing, and TF32

[benchmark_training.py](benchmark_training.py) and [lesson08.md](lesson08.md) begin section 2.2, following [the video around 1:22–1:39](https://www.youtube.com/watch?v=l8pRSuU81PU&t=4938s) and [historical commit 5265b20](https://github.com/karpathy/build-nanogpt/commit/5265b2042aaaa84943313002671b8b3ca7749b06). The script checks devices, synchronizes accelerator work before reading elapsed time, excludes warmup from summary statistics, and reports total training tokens divided by total measured time.

```bash
.venv/bin/python benchmark_training.py --inspect-only
.venv/bin/python benchmark_training.py --device cpu
```

The current Mac uses `/Users/guqun/micrograd/venv/bin/python`. Defaults are the full 124M model, B=4/T=32, two warmup updates and five measured updates. The 25 tests available at this block passed; [the CPU report](artifacts/lesson08-cpu-baseline.json) is archived. The lesson includes CUDA `highest`/`high` comparison commands. The isolated tool environment did not detect CUDA or MPS; a regular-terminal check on 2026-10-20 confirmed MPS eager training, documented in block 10. CUDA/TF32 performance measurements remain pending. The learner reports a basic understanding of this block and block 9 above.

## Learning block 7: parameter initialization

[train_initialization.py](train_initialization.py) and [lesson07.md](lesson07.md) follow [the video around 1:13:03](https://www.youtube.com/watch?v=l8pRSuU81PU&t=4383s) and [historical commit 9ac321e](https://github.com/karpathy/build-nanogpt/commit/9ac321e299f06c0414bd577fd2776f4562283a8a). Shared weights are retained. Ordinary weights use a normal distribution with std 0.02, Linear biases start at zero, and residual output projections use std `0.02/sqrt(2L)`. LayerNorm retains its default scale and offset. Initialization runs before training, not before each batch.

```bash
.venv/bin/python train_initialization.py --inspect-only
.venv/bin/python train_initialization.py
```

The current Mac uses `/Users/guqun/micrograd/venv/bin/python`. Inspection includes actual initial statistics and the video's independent residual-sum example. All 22 behavior tests pass; the full 50-step CPU/FP32 run is in [the report](artifacts/lesson07-initialization-report.json). [Historical-source comparisons](artifacts/lesson07-course-alignment.json) match initial parameters, logits/loss, gradients, and one AdamW update for two seeds and a small configuration with head dimension 64. This does not assert parity with the video's full GPU training trajectory. The learner reports a basic understanding and has confirmed section 2.1 complete; self-checks remain available for review.

## Learning block 6: weight tying

[train_weight_tying.py](train_weight_tying.py) and [lesson06.md](lesson06.md) follow [the video around 1:05:55](https://www.youtube.com/watch?v=l8pRSuU81PU&t=3955s) and [historical commit ec39245](https://github.com/karpathy/build-nanogpt/commit/ec39245425b2c82a38c0787f27e5f9d7082c973b). The script assigns the input embedding parameter to the existing output parameter, preserving the author's initialization direction. It starts a fresh random model, reuses block 5's sequential loader, and introduces no dedicated GPT-2 initialization yet.

```bash
.venv/bin/python train_weight_tying.py --inspect-only
.venv/bin/python train_weight_tying.py
```

The current Mac uses the authorized existing interpreter `/Users/guqun/micrograd/venv/bin/python` in place of `.venv/bin/python`. Inspection reports 163,037,184 → 124,439,808 unique parameters. All 19 behavior tests pass, including summed input/output gradients and a single optimizer update of the shared parameter. The full 50-step run is recorded in [the report](artifacts/lesson06-weight-tying-report.json). Losses use changing training batches and do not establish a held-out benefit from tying.

## Project guide

| File or directory | Purpose |
| --- | --- |
| [inspect_checkpoint.py](inspect_checkpoint.py) | Inspect a locally cached GPT-2 configuration and pretrained weights using the Hugging Face model. |
| [gpt2_model.py](gpt2_model.py) | Independent GPT-2 architecture, forward computation, and Hugging Face weight import. |
| [compare_forward.py](compare_forward.py) | Run both implementations on identical inputs, compare intermediate tensors, and check causality. |
| [lesson02.md](lesson02.md) | Step-by-step instructions for the forward comparison lesson, in Chinese. |
| [generate_text.py](generate_text.py) | Greedy and temperature/top-k generation with the independent GPT-2, step logs, and an optional reference check. |
| [lesson03.md](lesson03.md) | Text generation lesson, measured examples, and self-checks, in Chinese. |
| [Generation flow map](docs/diagrams/gpt2-generation-flow.md) | Diagram and source navigation using matching M/G/S step labels, with tensor shapes, selection branches, and stopping paths. |
| [train_step.py](train_step.py) | Tiny Shakespeare single-batch overfitting: random initialization, shifted targets, cross-entropy, and 50 AdamW updates. |
| [lesson04.md](lesson04.md) | Source mapping, training data and expectations, execution order, shapes, flow diagram, and measured results, in Chinese. |
| [train_dataloader.py](train_dataloader.py) | Sequential full-corpus batches: persistent cursor, shifted targets, wraparound, and a data-only preview. |
| [lesson05.md](lesson05.md) | Comparison with Stage 1 random sampling and block 4 fixed data, source mapping, and measured checks, in Chinese. |
| [train_weight_tying.py](train_weight_tying.py) | Input/output parameter sharing, inspection, and sequential training. |
| [lesson06.md](lesson06.md) | Shared lookup/scoring, parameter savings, and gradient accumulation, in Chinese. |
| [train_initialization.py](train_initialization.py) | Course initialization, residual-sum demonstration, and sequential training. |
| [lesson07.md](lesson07.md) | Initialization timing, distributions, residual scaling, source boundaries, and measured checks, in Chinese. |
| [benchmark_training.py](benchmark_training.py) / [lesson08.md](lesson08.md) | Device selection, synchronized timing, throughput, and a CUDA TF32 comparison. |
| [train_mixed_precision.py](train_mixed_precision.py) / [lesson09.md](lesson09.md) | BF16 autocast, per-layer dtype observations, and matched training comparisons with FP32 parameters. |
| [train_compile.py](train_compile.py) / [lesson10.md](lesson10.md) | Compile the existing model, retain shared parameters and AMP training, compare eager/compiled execution, and separate startup from subsequent timings. |
| [Data-loader flow map](docs/diagrams/gpt2-dataloader-flow.md) | Saved PNG/SVG diagram with M1–M4 / L1–L4 source locations, tensor shapes, cursor wraparound, and the training loop. |
| [data/tiny_shakespeare/](data/tiny_shakespeare/README.md) | Original course text, pinned source, and checksum. |
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

The table describes the default tied-weight model used for pretrained comparisons and generation. The first overfitting stage follows the earlier video version, with separate input/output tables and PyTorch default initialization; it temporarily has 163,037,184 parameters. Weight tying is introduced later in the video.

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

On the verified macOS Python 3.9 setup, run these commands from the repository root:

```bash
/usr/bin/python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/hf download openai-community/gpt2 config.json model.safetensors tokenizer.json \
  --revision 607a30d783dfa663caf39e06633721c8d4cfcd7e \
  --local-dir .cache/gpt2/607a30d783dfa663caf39e06633721c8d4cfcd7e
```

The weights are approximately 548 MB and the tokenizer is approximately 1.4 MB. The environment and model cache are Git-ignored. After setup, run `.venv/bin/python inspect_checkpoint.py`, `.venv/bin/python compare_forward.py`, or `.venv/bin/python generate_text.py`.

The scripts read existing local files. They do not download weights. The model directory must contain at least:

```text
config.json
model.safetensors
tokenizer.json  # additionally required for text generation
```

By default, they first look for this fixed snapshot inside the repository:

```text
.cache/gpt2/607a30d783dfa663caf39e06633721c8d4cfcd7e
```

They also support the original Hugging Face cache snapshot:

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

### Generate text

```bash
.venv/bin/python generate_text.py --prompt "Hello, I'm a language model," --check-reference --report .cache/runtime-check/lesson03-greedy.json
.venv/bin/python generate_text.py --prompt "Hello, I'm a language model," --strategy sample --temperature 0.8 --top-k 50 --seed 42 --report .cache/runtime-check/lesson03-sample.json
```

The recorded examples and diagram use the fixed classroom prompt `Hello, I'm a language model,`, with a budget of 20 new tokens. Running without `--prompt` uses the current default configured in `main()`, which can be changed for personal experiments. The generation loop calls the independent model, takes the final position's logits, selects one token, and appends it to the prefix. It uses one sequence, CPU/FP32, and no KV cache. EOS, the new-token budget, or the teaching implementation's total-length cap of 1024 stops the loop.

The classroom greedy experiment produced the same complete token ID sequence as Hugging Face generation. The sampling example was also run, and five behavior tests passed. See [the lesson](lesson03.md) for measured outputs and the scope of these checks. Default reports go to `.cache/runtime-check/lesson03-report.json`; another run replaces the same report path.

```bash
.venv/bin/python -m unittest discover -s tests -v
```

### Overfit one Tiny Shakespeare batch

```bash
.venv/bin/python train_step.py
```

This uses the first 1,000 characters of the included Tiny Shakespeare file: 285 encoded tokens, of which the first 129 form `x/y=[4,32]`. Defaults are CPU/FP32, seed 42, and 50 AdamW updates at `lr=3e-4`, using AdamW's other defaults, including `weight_decay=0.01`. Data is fixed across updates. At this video stage, input/output embeddings remain separate and modules retain their PyTorch default initialization. The forward architecture keeps 12 layers, 12 heads, and width 768.

The [recorded run](artifacts/lesson04-overfit-report.json) reduced same-batch loss from `10.927578` to `0.002746`; next-token predictions with ground-truth prefixes improved from `0/128` to `128/128`. This is a training sanity check, not a generation or generalization metric. New reports go to `.cache/runtime-check/lesson04-report.json`; no model checkpoint is saved. Seven training behavior tests and five generation tests pass.

The source stages are [ce84eb3](https://github.com/karpathy/build-nanogpt/commit/ce84eb34ea813ec463a2587ddd8147f158df4c82), switching from pretrained to random parameters, and [7822fce](https://github.com/karpathy/build-nanogpt/commit/7822fce8ebf46ba3907ee8f7cf0356f8ec1c63fa), adding the fixed-batch loop. [lesson04.md](lesson04.md) records video references and remaining local differences: tokenizer library, device, fixed seed, and observational logging. Fine-tuning belongs to later roadmap units.

### Train with sequential batches

```bash
.venv/bin/python train_dataloader.py --preview-only
.venv/bin/python train_dataloader.py
```

The preview shows three batches without constructing a model. The training command encodes the complete Tiny Shakespeare corpus into 338,025 GPT-2 tokens and performs 50 updates on different batches. `DataLoaderLite` takes `B*T+1` IDs, constructs shifted `[B,T]` inputs and targets, advances by `B*T`, and resets when another full batch will not fit. The loader is constructed once; `next_batch()` runs inside the training loop. Each script invocation creates a fresh random model with the previous block's untied embeddings and default initialization.

This follows [the video around 1:01:52](https://www.youtube.com/watch?v=l8pRSuU81PU&t=3712s) and [historical commit 631f7d6](https://github.com/karpathy/build-nanogpt/blob/631f7d6f080effc4342b5e2f2f41879642c76fd1/train_gpt2.py). The [50-step report](artifacts/lesson05-dataloader-report.json) records pre-update loss on changing training batches; these values can fluctuate and do not measure held-out performance. All 17 tests pass. [Loader parity checks](artifacts/lesson05-loader-alignment.json) cover short boundaries and 2642 calls on the full corpus, using identical token IDs in the historical and local implementations. See [lesson05.md](lesson05.md) for adaptation details. The previous script and lesson remain available for single-batch overfitting.

### VS Code setup

Open this repository as the workspace and select its interpreter with **Python: Select Interpreter**:

```text
.venv/bin/python
```

Then open the desired script and choose **Run Python File**. The workspace settings provide this interpreter as the default. Use `--model-dir` for a checkpoint stored elsewhere. The scripts locate their default paths independently of the terminal's working directory.

## Recorded verification

The checked-in [forward report](artifacts/lesson02-report.json) contains three input cases: two four-token sequences, one single-token sequence, and five repetitions of the same token ID at different positions.

- **45 stage comparisons passed:** embedding output, all 12 blocks, final LayerNorm, and logits for each case.
- The comparison checks matching shapes, finite values, and the elementwise condition `abs(actual - reference) <= 1e-4 + 1e-4 * abs(reference)`.
- The largest recorded final-logit absolute difference was approximately **1.79 × 10⁻⁴**; all cases passed the elementwise tolerance above.
- Changing the final input token left earlier-position logits unchanged in the causal check.
- The report also confirms **124,439,808** unique parameters, shared token embedding/output weights, and distinct QKV parameter objects in the first two blocks.

These forward checks establish agreement for the recorded cases. They passed again after adding model options for the earlier training stage. Generation and fixed-batch overfitting have separate checks in the third and fourth lessons; full pretraining remains ahead.

## Python supplement and review

The [Python foundations material](python_foundations/README.md) contains 17 units for developers already comfortable with Java, C++, or Dart. It focuses on Python semantics and the Tensor, autograd, and `nn.Module` concepts needed to read model code.

- [Markdown textbook](python_foundations/textbook.md), readable on GitHub.
- [Interactive HTML textbook](python_foundations/textbook.html), opened locally in a browser after cloning.
- [Runnable examples](python_foundations/examples/), a [configuration exercise](python_foundations/exercises/config_report.py), and its [reference solution](python_foundations/answers/config_report.py).

For the model lessons, return to the [consolidated notes](docs/lesson01-02-notes.md) and trace the same input from the checkpoint through the Python calls to the final logits.
