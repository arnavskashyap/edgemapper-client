"LiteGfm: A Lightweight Self-supervised Monocular Depth Estimation Framework for Artifacts Reduction via Guided Image Filtering"

### Setup
We ran our experiments with PyTorch 1.10.2, CUDA 11.3, Python 3.6.13, and Ubuntu 20.04.


### Data Preparation
Please refer to [Monodepth2](https://github.com/nianticlabs/monodepth2) to prepare the KITTI data. 


### Single Image Test
You can predict the depth for a single image with:
    python test_simple.py --load_weights_folder path/to/your/weights/folder --image_path path/to/your/test/image


### Evaluation
    python evaluate_depth.py --load_weights_folder path/to/your/weights/folder --data_path path/to/kitti_data/ --model litegfm


### Training
#### dependency installation 
    pip install 'git+https://github.com/saadnaeem-dev/pytorch-linear-warmup-cosine-annealing-warm-restarts-weight-decay'
    
#### start training
    python train.py --data_path path/to/your/data --model_name litegfm --num_epochs 60 --batch_size 16 --lr 0.0005 5e-6 31 0.0001 1e-5 31
