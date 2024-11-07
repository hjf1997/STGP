config=$1
batch_size=$2
gpu_ids=$3
seed=$4

for i in {1..2}
do
python train.py --config ${config} --batch_size ${batch_size} --gpu_ids ${gpu_ids} --stage pre_training --enable_val --save_best --seed $((${seed}+${i}))
done

#$((${seed}+${i}))