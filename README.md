# GANE: A Generative Adversarial Network Embedding
This is the codes and data for GANE model in our paper "[GANE: A Generative Adversarial Network Embedding][1]".

If you would like to acknowledge our efforts, please cite the following paper:

    @article{hong2019gane,
    title={GANE: A Generative Adversarial Network Embedding},
    author={Hong, Huiting and Li, Xin and Wang, Mingzhong},
    journal={IEEE transactions on neural networks and learning systems},
    year={2019},
    publisher={IEEE}
    }

# Prerequisites

python 2.7

tensorflow

cPickle

numpy

multiprocessing


# Usage
There are four files:
- `utils.py`: Utile functions for data preparing;
- `dis_model.py`: The discriminator of GANE;
- `gen_model.py`: The generator of GANE;
- `gane.py`: The trainer to minimax our discriminator and generator.

## How to run
```shell
python gane.py --emb_dim 128 --epochs=150 --suffix 128d --init_lr_gen 1e-5 --init_lr_dis 1e-5
```



## Note
- The output files (learned embeddings) will be stored in the `output-suffix` directory during training process, if ``--suffix suffix``.



[1]: https://ieeexplore.ieee.org/document/8758400/
