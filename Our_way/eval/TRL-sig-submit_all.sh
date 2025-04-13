#!/bin/bash

# 第一步：生成 4 个配置文件
python TRL-sig-test-generate.py

# 第二步：循环提交 4 个独立作业
datasets=("LogicBench" "LogiQA" "LogiQA2" "Reclor")

for ds in "${datasets[@]}"; do
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=llama3_${ds}
#SBATCH --gres=gpu:4090d:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=200G
#SBATCH --time=168:00:00
#SBATCH --output=logs/${ds}_%j.out
#SBATCH --error=logs/${ds}_%j.err

hostname
nvidia-smi

source ~/.bashrc
conda activate trl

python config_sig_${ds}.py
EOF

    echo "已提交作业: ${ds}"
done