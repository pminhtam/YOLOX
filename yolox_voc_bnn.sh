#!/bin/bash
#SBATCH --job-name=yoloxVOC_bnn
#SBATCH --output=slurm_bnn_yolox_voc_%A.out
#SBATCH --error=slurm_bnn_yolox_voc_%A.err
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --mem-per-cpu=6G
#SBATCH --cpus-per-task=32
#SBATCH --partition=research

nvidia-smi
module purge
module load python/miniconda3/miniconda3
eval "$(conda shell.bash hook)"
export HTTP_PROXY=http://proxytc.vingroup.net:9090/
export HTTPS_PROXY=http://proxytc.vingroup.net:9090/
export http_proxy=http://proxytc.vingroup.net:9090/
export https_proxy=http://proxytc.vingroup.net:9090/
rm -rf /home/tampm2/.conda/envs/yolox_voc_bnn
conda create --name yolox_voc_bnn python=3.7 --force
conda activate yolox_voc_bnn
pip install -r requirements.txt
pip install setuptools==59.5.0
pip install tensorboard
pip install -v -e .
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install wandb
wandb login c5a6f4c212b00734d9517784e3e892155a301de0
#tar -xvf /vinai-public-dataset/VOC2012/VOCtrainval_11-May-2012.tar -C data/
#tar -xvf /vinai-public-dataset/VOC2007/VOCtrainval_06-Nov-2007.tar -C data/
#tar -xvf /vinai-public-dataset/VOC2007/VOCtest_06-Nov-2007.tar -C data/
python -m yolox.tools.train -n yolox-s -d 1 -b 32 -o  --cache  -expn yolox_s_voc_bnn_fix  --binary_backbone --binary_head --resume
### _fix : just binaize conv with ksize = 3x3 or larger.