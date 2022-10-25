#!/bin/bash
#SBATCH --job-name=lBDClReMOT
#SBATCH --output=slurm_yolox_l_mot_bnn_d_clip_reg_%A.out
#SBATCH --error=slurm_yolox_l_mot_bnn_d_clip_reg_%A.err
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=32
#SBATCH --partition=applied
#SBATCH --nodelist=sdc2-hpc-dgx-a100-014

nvidia-smi
module purge
module load python/miniconda3/miniconda3
eval "$(conda shell.bash hook)"
export HTTP_PROXY=http://proxytc.vingroup.net:9090/
export HTTPS_PROXY=http://proxytc.vingroup.net:9090/
export http_proxy=http://proxytc.vingroup.net:9090/
export https_proxy=http://proxytc.vingroup.net:9090/
conda activate yolox_s_coco_c_ori
#rm -rf /home/tampm2/.conda/envs/yolox_l_mot_bnn_d_clip_reg
#conda create --name yolox_l_mot_bnn_d_clip_reg python=3.7 --force
#conda activate yolox_l_mot_bnn_d_clip_reg
#pip install -r requirements.txt
#pip install setuptools==59.5.0
#pip install tensorboard
#pip install -v -e .
#pip install torch-tb-profiler
#pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
#pip install wandb
#wandb login c5a6f4c212b00734d9517784e3e892155a301de0
python -m yolox.tools.train -n yolox-l -d 1 -b 8 -o -expn yolox_l_mot_bnn_d_clip_reg_weight2 --binary_backbone --binary_head --data_dir /lustre/scratch/client/vinai/users/tampm2/ssd.pytorch/data/MOT_LT --clip_grad --resume
### _fix : just binaize conv with ksize = 3x3 or larger.
### weight2 : prior and regression loss weight == 2