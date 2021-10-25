#!/bin/bash

#######################################################
## DIFFERENT OPTIMISMS                               ##
## (TOP-M PROBLEM)                                   ##
#######################################################

cd ../code

njobs=$(python -c "import multiprocessing;print(multiprocessing.cpu_count()-1)")
nsimu=500
K=15
N=8
m=3
sigma=1
delta=0.05
problem="gaussian"
algos="MisLid"
c=1
c_data=1
data_type="deviation_Linf"
beta="heuristic"
gain_type_values=("misspecified" "aggressive_misspecified" "empirical")
py_cmd="python3"
M=2
folder="ExperimentC"
for GAIN_TYPE in "${gain_type_values[@]}"; do
	echo $folder": "$algos" gain_type="$GAIN_TYPE
	exp_name=$folder"_gain_type="$GAIN_TYPE
	cmd=$py_cmd" main.py --data_type $data_type --c $c --c_data $c_data --problem_type $problem --beta_linear $beta --delta $delta --m $m --sigma $sigma --K $K --N $N --sigma $sigma --nsimu $nsimu --method $algos --n_jobs $njobs --M "$M
	cmd_=$cmd" --exp_name "$exp_name" --gain_type "$GAIN_TYPE
	echo $cmd_
	if [ "$1" == "" ];
	then
		$cmd_
	fi
	mkdir -p "../results/"$folder
	if [ "$GAIN_TYPE" == "aggressive_misspecified" ];
	then
		GAIN_TYPE="aggressive"
	fi
	cp "../results/"$exp_name"/method="$algos"_beta_linear="$beta"_delta="$delta"_c=1.0_sigma=1.0.csv" "../results/"$folder"/method="$algos"("$GAIN_TYPE")_.csv"
done
algo_list=$(IFS=, ; echo "${algos[*]}")
cmd=$cmd" --boxplot None --exp_name "$folder
$cmd
