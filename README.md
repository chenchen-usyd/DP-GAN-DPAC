# DP-GAN-DPAC

This is the official code base for our CVPR2023 paper: Private Image Generation with Dual-Purpose Auxiliary Classifier.

Contact: Chen Chen (cche0711@uni.sydney.edu.au)


## Dependencies 
The environment can be prepared with the following commands:

``` setup
conda create --name virtualenv python=3.7
conda activate virtualenv
conda install pytorch==1.11.0 torchvision==0.12.0 -c pytorch
conda install tensorflow==1.14.0
pip install "pillow<7"
pip install -r requirements.txt
```

## Dataset
- MNIST and FashionMNIST datasets can be automatically downloaded during pre-training and training.
- For CelebA, please refer to their official websites for downloading, and put the files into the `data/celeba/` directory.

## Training 
### Step 1. To warm-start the discriminators:
#### MNIST
```warm-start-mnist
cd source
sh pretrain_mnist.sh
```

#### Fashion MNIST
```warm-start-fmnist
cd source
sh pretrain_fashionmnist.sh
```

#### CelebA
```warm-start-celeba
cd source
sh pretrain_celeba_gender.sh
```

- The pre-training can be done in parallel by adjusting the `'meta_start'`, `'dis_per_job'`, and `'njobs'` arguments and run the script multiple times in parallel.

### Step 2. To train the differentially private generator:
#### MNIST
```train
cd source
python main.py -data 'mnist' -name 'ResNet_default' -ldir '../results/mnist/pretrain/ResNet_default' -ndis 1000 -sstep 1000 -iter 20000 -wd .8 -wc1 .2 -C1in 6000 -C1_fake_iters 10 -C1_real_iters 10 -ngpus 2 
```

#### Fashion MNIST
```train
cd source
python main.py -data 'fashionmnist' -name 'ResNet_default' -ldir '../results/fashionmnist/pretrain/ResNet_default' -ndis 1000 -sstep 1000 -iter 20000 -wd .8 -wc1 .2 -C1in 6000 -C1_fake_iters 10 -C1_real_iters 10 -ngpus 2 
```

#### CelebA
```train
cd source
python main_celeba_gender.py -data 'celeba_gender' -name 'ResNet_default' -ldir '../results/celeba_gender/pretrain/ResNet_default' -ndis 2543 -bs 32 -sstep 1000 -iter 20000 -wd .8 -wc1 .2 -C1in 1000 -C1_fake_iters 10 -C1_real_iters 10 -noise 0.61135 -diters 10 -ngpus 5
```

- Please refer to `source/config.py` (or execute `python main.py -h`) for the complete list of arguments. 
- Please allocate appropriate number of GPUs by adjusting the `'-ngpus'` argument according to your GPU memory. Our implementations use 2 cards of RTX 3090 with 24GB memory for MNIST and Fashion MNIST, and 5 cards for CelebA.

## Evaluation
### Privacy
- To compute the privacy cost:

#### MNIST and Fashion MNIST
```privacy-mnist
cd evaluation
python privacy_analysis.py -name 'ResNet_default' -bs 32 -ndis 1000 -noise 1.07 -iters 20000
```

#### CelebA
```privacy-celeba
cd evaluation
python privacy_analysis.py -name 'ResNet_default' -bs 32 -ndis 2543 -noise 0.61135 -iters 20000
```

### Standard Utility - gen2real with MLP classifiers
#### MNIST
```mlp-mnist
cd evaluation
python eval_mlp.py --gen_data './../results/mnist/main/ResNet_default/gen_data.npz' -data 'mnist'
``` 

#### Fashion MNIST
```mlp-fmnist
cd evaluation
python eval_mlp.py --gen_data './../results/fashionmnist/main/ResNet_default/gen_data.npz' -data 'fashionmnist'
``` 

#### CelebA
```mlp-celeba
cd evaluation
python eval_celeba.py --gen_data './../results/celeba_gender/main/ResNet_default/gen_data.npz' -arch 'mlp'
``` 

### Standard Utility - gen2real with CNN classifiers
#### MNIST
```cnn-mnist
cd evaluation
python eval_cnn.py --gen_data './../results/mnist/main/ResNet_default/gen_data.npz' -data 'mnist' -lr 0.05
``` 

#### Fashion MNIST
```cnn-fmnist
cd evaluation
python eval_cnn.py --gen_data './../results/fashionmnist/main/ResNet_default/gen_data.npz' -data 'fashionmnist' -lr 0.05
``` 

#### CelebA
```cnn-celeba
cd evaluation
python eval_celeba.py --gen_data './../results/celeba_gender/main/ResNet_default/gen_data.npz' -lr 0.05
``` 

### Reversed Utility - real2gen with MLP classifiers
#### MNIST
```mlp-mnist-r2g
cd evaluation
python eval_mlp.py --gen_data './../results/mnist/main/ResNet_default/gen_data.npz' -data 'mnist'
``` 

#### Fashion MNIST
```mlp-fmnist-r2g
cd evaluation
python eval_mlp.py --gen_data './../results/fashionmnist/main/ResNet_default/gen_data.npz' -data 'fashionmnist'
``` 

#### CelebA
```mlp-celeba-r2g
cd evaluation
python eval_celeba_real2gen.py --gen_data './../results/celeba_gender/main/ResNet_default/gen_data.npz' -arch 'mlp'
``` 

### Reversed Utility - real2gen with CNN classifiers
#### MNIST
```cnn-mnist-r2g
cd evaluation
python eval_cnn_real2gen.py --gen_data './../results/mnist/main/ResNet_default/gen_data.npz' -data 'mnist' -ep 100 -lr 0.01
``` 

#### Fashion MNIST
```cnn-fmnist-r2g
cd evaluation
python eval_cnn_real2gen.py --gen_data './../results/fashionmnist/main/ResNet_default/gen_data.npz' -data 'fashionmnist' -ep 100 -lr 0.01
``` 

#### CelebA
```cnn-celeba-r2g
cd evaluation
python eval_celeba_real2gen.py --gen_data './../results/celeba_gender/main/ResNet_default/gen_data.npz'
``` 

### Inception Score

#### MNIST
```IS-mnist
cd evaluation
python train_mnist_inception_score.py -data 'mnist'
python eval_mnist_inception_score.py -data 'mnist' --gen_data './../results/mnist/main/ResNet_default/gen_data.npz'
```

#### Fashion MNIST
```IS-fmnist
cd evaluation
python train_mnist_inception_score.py -data 'fashionmnist'
python eval_mnist_inception_score.py -data 'fashionmnist' --gen_data './../results/fashionmnist/main/ResNet_default/gen_data.npz'
```

#### CelebA
```IS-celeba
cd evaluation
python eval_celeba_inception_score.py --gen_data './../results/celeba_gender/main/ResNet_default/gen_data.npz'
```

### Frechet Inception Distance (FID)

#### MNIST
```FID-mnist
cd evaluation
python eval_fid.py -data 'mnist' --gen_data './../results/mnist/main/ResNet_default/gen_data.npz' 
```

#### Fashion MNIST
```FID-fmnist
cd evaluation
python eval_fid.py -data 'fashionmnist' --gen_data './../results/fashionmnist/main/ResNet_default/gen_data.npz' 
```

#### CelebA
```FID-celeba
cd evaluation
python eval_celeba_fid.py -data 'celeba' --gen_data './../results/celeba_gender/main/ResNet_default/gen_data.npz' 
```

## Acknowledgements

Our implementation uses the source code from the following repositories:

* [Improved Training of Wasserstein GANs (Pytorch)](https://github.com/caogang/wgan-gp.git)

* [Improved Training of Wasserstein GANs (Tensorflow)](https://github.com/igul222/improved_wgan_training)

* [GANs with Spectral Normalization and Projection Discriminator](https://github.com/pfnet-research/sngan_projection)

* [Large Scale GAN Training for High Fidelity Natural Image Synthesis](https://github.com/ajbrock/BigGAN-PyTorch.git)

* [Progressive Growing of GANs](https://github.com/tkarras/progressive_growing_of_gans.git)

* [GS-WGAN](https://github.com/DingfanChen/GS-WGAN)

* [DP-Sinkhorn](https://github.com/nv-tlabs/DP-Sinkhorn_code/tree/main/src/data)

* [Inception Score (Pytorch)](https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py)

* [Frechet Inception Distance (Pytorch)](https://github.com/mseitzer/pytorch-fid)

