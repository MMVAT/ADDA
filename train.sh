# #!/bin/bash

# SEED=0

# python main.py --mode train -g default -e cmpae_s${SEED} -w --seed ${SEED} --gpu 0

# echo "Begin to select thresholds from the validation set."
# echo "Note: In this stage, five processes will run in parallel to save time."
# echo "Please wait for about 10 min."

# python main.py --mode select_thresholds -g default -e cmpae_s${SEED} --seed ${SEED} -w -s --gpu 0 --start_class 0 &
# python main.py --mode select_thresholds -g default -e cmpae_s${SEED} --seed ${SEED} -w -s --gpu 1 --start_class 5 &
# python main.py --mode select_thresholds -g default -e cmpae_s${SEED} --seed ${SEED} -w -s --gpu 2 --start_class 10 &
# python main.py --mode select_thresholds -g default -e cmpae_s${SEED} --seed ${SEED} -w -s --gpu 3 --start_class 15 &
# python main.py --mode select_thresholds -g default -e cmpae_s${SEED} --seed ${SEED} -w -s --gpu 4 --start_class 20 &

# wait

# python main.py --mode test -g default -e cmpae_s${SEED} --seed ${SEED} -w -s --gpu 0


GPU=3
save='save/'
thres="0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7"
metric='avg_all'
eta=0.96
t=0.15

SEED_COUNT=1

for ((i=0; i<SEED_COUNT; i++)); do
    SEEDS[i]=$i
done

echo "将使用以下seed值进行实验: ${SEEDS[@]}"

# 循环遍历所有seed值
for SEED in "${SEEDS[@]}"; do
    echo -e "\n===== Running experiment with SEED=${SEED} ====="
    
    # 训练阶段
    echo "开始训练阶段..."
    python main.py --mode train -g default -e cmpae_s${SEED} -w --seed ${SEED} --gpu $GPU --model_save_dir ${save} --select_metric ${metric} --thres_candi ${thres} --mutual-eta ${eta} --temperature ${t}

    # 并行选择阈值阶段
    echo -e "\n开始验证集阈值选择阶段(SEED=${SEED})..."
    python main.py --mode select_thresholds -g default -e cmpae_s${SEED} --seed ${SEED} -w --gpu 0 --model_save_dir ${save} --select_metric ${metric} --thres_candi ${thres} --mutual-eta ${eta} --temperature ${t} --start_class 0 &
    python main.py --mode select_thresholds -g default -e cmpae_s${SEED} --seed ${SEED} -w --gpu 1 --model_save_dir ${save} --select_metric ${metric} --thres_candi ${thres} --mutual-eta ${eta} --temperature ${t} --start_class 6 &
    python main.py --mode select_thresholds -g default -e cmpae_s${SEED} --seed ${SEED} -w --gpu 2 --model_save_dir ${save} --select_metric ${metric} --thres_candi ${thres} --mutual-eta ${eta} --temperature ${t} --start_class 12 &
    python main.py --mode select_thresholds -g default -e cmpae_s${SEED} --seed ${SEED} -w --gpu 3 --model_save_dir ${save} --select_metric ${metric} --thres_candi ${thres} --mutual-eta ${eta} --temperature ${t} --start_class 18 &
    wait

    # 测试阶段
    echo -e "\n开始测试阶段(SEED=${SEED})..."
    python main.py --mode test -g default -e cmpae_s${SEED} --seed ${SEED} -w -s --gpu $GPU --model_save_dir ${save} --select_metric ${metric} --thres_candi ${thres} --mutual-eta ${eta} --temperature ${t}
    
    echo "===== SEED=${SEED} 的实验已完成 ====="
done
