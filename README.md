
HuggingFace 下载被墙了。用国内源：

```bash
export MINERU_MODEL_SOURCE=modelscope
python tests/test_parse.py
```

或者指定 modelscope 镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
python tests/test_parse.py
```

两个都试一下，第一个是 MinerU 内置的 modelscope 源，第二个是 HuggingFace 国内镜像站。