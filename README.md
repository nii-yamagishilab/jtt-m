# jtt-m

This is an implementation of our paper: [Outlier-Aware Training for Improving Group Accuracy Disparities](https://arxiv.org/abs/2210.15183).
If you find our code useful, please cite:

```bash
@inproceedings{chen-2022-outlier,
  title = "Outlier-Aware Training for Improving Group Accuracy Disparities",
  author = "Chen, Li-Kuang and Kruengkrai, Canasai and Yamagishi, Junichi",
  booktitle = "Proceedings of the 2nd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 12th International Joint Conference on Natural Language Processing: Student Research Workshop",
  year = "2022",
  url = "https://arxiv.org/abs/2210.15183",
}
```

## Requirements

The code is tested on Python 3.9 and PyTorch 1.10.1.
We recommend to create a new environment for experiments using conda:
```bash
conda create -y -n jtt-m python=3.9
conda activate jtt-m
```

Then, from the `jtt-m` project root, run:
```bash
pip install -r requirements.txt
```

For further development or modification, we recommend installing `pre-commit`:
```bash
pre-commit install
```

To ensure that PyTorch is installed and CUDA works properly, run:
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

We should see:
```bash
1.10.1+cu111
True
```

:warning: We use PyTorch 1.10.1 with CUDA 11.0. You may need another CUDA version suitable for your environment.

## Experiments

See [experiments](experiments).
