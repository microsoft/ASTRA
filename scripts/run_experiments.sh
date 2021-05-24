#!/bin/bash

# NAACL 2021 Experiments

# TREC
for seed in 0 20 7 1993 128 42; do
  logdir="${baselogdir}/seed${seed}"
  python ../astra/main.py --dataset trec --logdir ${logdir} --seed $seed --learning_rate 0.0001 --finetuning_rate 0.0001 --datapath ../data
done

# SMS
for seed in 0 20 7 1993 128 42; do
  logdir="${baselogdir}/seed${seed}"
  python ../astra/main.py --dataset sms --logdir ${logdir} --seed $seed --learning_rate 0.0001 --datapath ../data
done

# YOUTUBE
for seed in 0 20 7 1993 128 42; do
  logdir="${baselogdir}/seed${seed}"
  python ../astra/main.py --dataset youtube --logdir ${logdir} --seed $seed --learning_rate 0.0001 --hard_student_rule --datapath ../data
done

# CENSUS
for seed in 0 20 7 1993 128 42; do
  logdir="${baselogdir}/seed${seed}"
  python ../astra/main.py --dataset census --logdir ${logdir} --seed $seed --learning_rate 0.001 --hard_student_rule --datapath ../data
done

# MIT-R
for seed in 0 20 7 1993 128 42; do
  logdir="${baselogdir}/seed${seed}"
  python ../astra/main.py --dataset mitr --logdir ${logdir} --seed $seed --learning_rate 0.001 --finetuning_rate 0.0001 --hard_student_rule --num_iter 3 --soft_labels --datapath ../data
done
