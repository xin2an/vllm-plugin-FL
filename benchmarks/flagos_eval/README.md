# FlagOS Evaluation Suite

Evaluation toolkit for large language models with LM Eval and vLLM Benchmark.

## Quick Start

### 1. Dependencies

- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [vllm](https://github.com/vllm-project/vllm)
- [vllm-plugin-FL](https://github.com/flagos-ai/vllm-plugin-FL)
- Python 3.10+

### 2. Run Evaluation

```bash
cd flagos_eval

# LM Evaluation (~35 minutes)
./run_eval.sh /path/to/model/ hf_xxxxxxxxxxxxx

# Performance Benchmark
./run_benchmark.sh /path/to/model/
```

## Evaluation Tasks

### LM Evaluation

Tasks:
- **General**: BBH
- **Math**: GSM8K
- **Coding**: HumanEval, MBPP
- **Multilingual**: MGSM (Chinese)

Output Files:
- `results_summary.csv` - Summary
- `output/*/results*.json` - Detailed results

### Performance Benchmark

Metrics:
- **Throughput**: tokens/s, requests/s
- **Latency**: Mean, P50, P90, P99 (ms)

Output Files:
- `bench_summary.csv` - Summary
- `bench_results/*.json` - Detailed results

## Project Structure

```
flagos_eval/
├── run_eval.sh                  # LM evaluation script
├── run_benchmark.sh             # Performance benchmark script
├── collect_results.py           # LM evaluation result collector
├── collect_bench_results.py     # Performance benchmark result collector
├── output/                      # LM evaluation results
├── bench_results/               # Performance benchmark results
├── results_summary.csv          # LM evaluation summary
└── bench_summary.csv            # Performance benchmark summary
```
