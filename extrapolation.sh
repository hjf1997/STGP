config=$1
batch_size=$2
gpu_ids=$3
seed=$4

for i in {1..5}
do
python train.py --config ${config} --batch_size ${batch_size} --gpu_ids ${gpu_ids} --stage extrapolation_prompting --enable_val --eval_epoch_freq 5 --early_stop 50 --save_epoch_freq 10 --save_best --seed $((${seed}+${i}))
done