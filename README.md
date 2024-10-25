# LLaVA-LLaMA-3

This repo is deprecated. The updated repo is opensourced at [LLaVA-Video-Llama-3](https://github.com/Victorwz/LLaVA-Video-Llama-3), which supports the visual understanding towards multiple-images and short-videos.

*A reproduction with LLaMA-3 backbone, rather than an official implementation

🤝Community Contributions: [[LLaVA-LLaMA-3-8b](https://huggingface.co/weizhiwang/LLaVA-LLaMA-3-8B)]

## Updates
- This repo is upgraded to llava-next codebase to also support phi-3, llama-3 and mistral-v0.1 models.
- A new [`preprocess_llama3`](llava/train/train.py#492) function in ``llava/train/train.py`` for being compatible with LLaMA-3
- A new [`conv_llama_3`](llava/conversation.py#264) conversation templates in ``llava/conversations.py`` for being compatible with LLaMA-3
- This repo is compatible with latest huggingface `transformers==4.41.2` in order to support Phi-3 LLM backbone.

## Install

If you are using Windows, do *NOT* proceed, see instructions [here](https://github.com/haotian-liu/LLaVA/blob/main/docs/Windows.md).

1. Setup
```Shell
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Install additional packages for training cases
```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

### Fine-Tune Your Own LLaVA-Llama-3 Model
Please follow the updated fine-tuning script with DeepSpeed ZeRO-3: [`finetune.sh`](https://github.com/Victorwz/LLaVA-Llama-3/blob/main/scripts/finetune.sh). The following parameters are updated to accomodate Llama-3:
- `--version`: v3, which adopts the tokenization and preprocessing function with Llama-3 tokenizer.

Please download the pre-trained vision-language projector weights in [Projector_MODEL](https://huggingface.co/weizhiwang/llava-v1.5-llama-3-8b-pretrain-clip-large-336px).

In terms of the image data preparation, please follow [`DATA.md`](DATA.md).

## Demo with Gradio
Please follow [`DEMO.md`](DEMO.md).


### CLI Inference

Chat about images using LLaVA without the need of Gradio interface. It also supports multiple GPUs, 4-bit and 8-bit quantized inference. With 4-bit quantization, for our LLaVA-Llama-8B, it uses less than 8GB VRAM on a single GPU.

```Shell
python -m llava.serve.cli \
    --model-path weizhiwang/LLaVA-Llama-3-8B \
    --image-file "https://llava-vl.github.io/static/images/view.jpg" \
    --load-4bit
```

<img src="images/demo_cli.gif" width="70%">



## Evaluation

In LLaVA-1.5, the authors evaluate models on a diverse set of 12 benchmarks. To ensure the reproducibility, we evaluate the models with greedy decoding. We do not evaluate using beam search to make the inference process consistent with the chat demo of real-time outputs.

See [Evaluation.md](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md).


## Credits
This is a reproduction project, all research credits should be attributed to original authors for LLaVA. Please cite their papers listed below as well.

```bibtex
@misc{wang2024llavallama3,
  title={LLaVA-Llama-3-8B: A reproduction towards LLaVA-3 based on Llama-3-8B LLM backbone},
  author={Wang, Weizhi},
  year={2024}
}
```

```bibtex
@misc{wang2024llavallama3,
  title={LLaVA-Llama-3-8B: A reproduction towards LLaVA-v1.5 based on Llama-3-8B LLM backbone},
  author={Wang, Weizhi},
  year={2024}
}
```
