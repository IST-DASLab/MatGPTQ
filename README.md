# MatGPTQ

[![arXiv](https://img.shields.io/badge/arXiv-2602.03537-b31b1b.svg)](https://arxiv.org/abs/2602.03537)

**MatGPTQ: Accurate and Efficient Post-Training Matryoshka Quantization**

Official implementation of MatGPTQ (Matryoshka GPTQ), a new PTQ pipeline that produces a single parent model jointly optimized for multiple target precisions in one-shot, based on a small calibration set.

## Abstract

Matryoshka Quantization (MatQuant) is a recent quantization approach showing that a single integer-quantized model can be served across multiple precisions, by slicing the most significant bits (MSB) at inference time. This enables a single checkpoint to cover a wide range of memory and latency budgets, but renders quantization much more challenging. In particular, the initial MatQuant relies on expensive quantization-aware training (QAT) variants,  rather than fast one–shot post training quantization (PTQ), and lacks open-source and kernel support. We address all of these limitations by introducing _Post-Training Matryoshka Quantization_ (MatGPTQ), a new PTQ pipeline that produces a single parent model jointly optimized for multiple target precisions in one-shot, based on a small calibration set. MatGPTQ casts Matryoshka quantization as a multi–precision objective with bit-slicing and cross–bit error compensation, resulting in an algorithm that produces a multi-bit-width, "sliceable" model in a single pass. We also incorporate a new budget–aware search for heterogeneous per–layer bit-witdhs and provide efficient kernels that implement slicing and mixed–precision execution. Across standard LLMs and benchmarks, MatGPTQ preserves high–bit accuracy while substantially improving performance at low-bit-witdh settings. Overall, we establish a new state of the art for Matryoshka–style _post–training_ quantization and make single–checkpoint, multi–precision deployment open and practical.

## Repository structure

- ```scripts/``` —  contains bash scripts with the required arguments to run the method
- ```src/``` —  directory for helper methods and utility functions 
- ```evo_quant_search.py``` — evolutionary quantization bitwidth allocation
- ```quant.py``` — MatGPTQ/GPTQ quantization
- ```lmeval.py``` — LM Eval Harness evalution script 
- ```eval_ppl.py``` — perplexity evalution script

## Installation

Create a virtual environment and install dependencies (we recommend Python 3.12):

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

**Note:** The code has been tested with CUDA 12.4 and PyTorch 2.7.1

## Quantization

We provide `quant.py` for producing the MatGPTQ/[GPTQ](https://github.com/IST-DASLab/gptq) models. To produce the respective model see either `scripts/run_gptq.sh` or `scripts/run_matgptq.sh` for examples on how to run quantized training:

```bash
bash scripts/run_matgptq.sh
```

### Mix'n'Match

We provide `evo_quant_search.py` for producing the Mix'n'Match MatGPTQ models. To produce the respective model see `scripts/run_quant_search.sh` for an example on how to run [EvoPress](https://github.com/IST-DASLab/EvoPress) for MatGPTQ:

```bash
bash scripts/run_quant_search.sh
```

## Evaluations

We provide `lmeval.py` and `eval_ppl.py` scripts for evaluation on the [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) benchmarks and perplexity measurements. The interface of `lmeval.py` mostly follows the instructions from the original. In addition, one should specify the path to quantized weights via the `quant_weights_path` argument and the default uniform **quantization** bitwidth `quant_uniform_bitwidth` and master bitwidth `--quant_master_bitwidth`, or a path to a `.txt` file with chosen compression levels via the `--quant_non_uniform_config_path` argument. Furthermore, with `--method`, you define whether to evaluate MatGPTQ or GPTQ.

## Deployment

**Work In Progress**

## Citation

If you use MatGPTQ in your research, please cite:

```bibtex
@misc{kleinegger2026matgptqaccurateefficientposttraining,
      title={MatGPTQ: Accurate and Efficient Post-Training Matryoshka Quantization}, 
      author={Maximilian Kleinegger and Elvir Crnčević and Dan Alistarh},
      year={2026},
      eprint={2602.03537},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.03537}, 
}
```