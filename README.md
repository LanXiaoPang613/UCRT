# UCRT: a two-stage noisy label learning framework with uniform consistency selection and robust training
<h5 align="center">

*Qian Zhang, Qiu Chen*

[[preprint]](https://xxx)
[[License: MIT License]](https://github.com/LanXiaoPang613/UCRT/blob/main/LICENSE)

</h5>

The PyTorch implementation code of the paper, [UCRT: a two-stage noisy label learning framework with uniform consistency selection and robust training](https://xxx).

**Abstract**


## Installation

```shell
# Please install PyTorch using the official installation instructions (https://pytorch.org/get-started/locally/).
pip install -r requirements.txt
```

## Training

To train on the CIFAR dataset(https://www.cs.toronto.edu/~kriz/cifar.html), run the following command:

```shell
# stage one for CDN noise
python Train_cifar_ucrt_stage1.py --r 0.2 --noise_mode 'sym' --data_path './data/cifar-10-batches-py' --dataset 'cifar10' --num_class 10
# stage two for CDN noise
python Train_cifar_ucrt_stage2.py --r 0.2 --noise_mode 'sym' --data_path './data/cifar-10-batches-py' --dataset 'cifar10' --num_class 10
```

To train on the Animal-10N dataset(https://dm.kaist.ac.kr/datasets/animal-10n/), run the following command:

```shell
# stage one for Animal-10N
python Train_animal_ucrt_stage1.py --data_path './data/Animal-10N' --dataset 'animal10N' --num_class 10
# stage two for Animal-10N
python Train_animal_ucrt_stage2.py --data_path './data/Animal-10N' --dataset 'animal10N' --num_class 10
```

## Citation

If you have any questions, do not hesitate to contact zhangqian@jsou.edu.cn

Also, if you find our work useful please consider citing our work:

```bibtex
Zhang, Qian and Chen, Qiu, 
UCRT: a two-stage noisy label learning framework with uniform consistency selection and robust training. 
Applied Intelligence, 2026.
```

## Acknowledgement

* [DivideMix](https://github.com/LiJunnan1992/DivideMix): The algorithm that our framework is based on.
* [UNICON](https://github.com/nazmul-karim170/UNICON-Noisy-Label): Inspiration for the webvision dataset code.
* [LongReMix](https://github.com/filipe-research/LongReMix): Inspiration for our framework.
