#!/bin/bash

##############################################################
## MISSPECIFIED INSTANCES VS UNSTRUCTURED/LINEAR ALGORITHMS ##
## (TOP-M PROBLEM)                                          ##
##############################################################

cd ../code

njobs=$(python -c "import multiprocessing;print(multiprocessing.cpu_count())-1")
nsimu=500
K=10
N=5
sigma=1
delta=0.05
m=3
problem="gaussian"
algos=("LinGapE" 'MisLid' 'LUCB')
epsilon=0.
beta="heuristic"
data_type="misspecified_linear"
c_values=(0 5)
py_cmd="python3"
folder="ExperimentA"
# Delta = mu_{(m)}-mu_{(m+1)}=0.2785601566147834 in linear model
# - c>Delta: switch the optimal and suboptimal arms in the underlying linear model
# perfectly linear algorithms should fail here
# - c<0: doesn't switch optimal and suboptimal arms in the underlying linear model
for C in "${c_values[@]}"; do
	for ALGO in "${algos[@]}"; do
		c_data=$C
		echo $folder": '"$data_type"' "$ALGO" c=c_data="$C" (beta='"$beta"')"
		cmd=$py_cmd" main.py --data_type $data_type --c $C --problem_type $problem --beta_linear $beta --delta $delta --m $m --sigma $sigma --K $K --N $N --nsimu $nsimu --method $ALGO --n_jobs $njobs --epsilon $epsilon --exp_name "$folder"_c="$C
		echo $cmd
		if [ "$1" == "" ];
		then
			$cmd
		fi
	done
	algo_list=$(IFS=, ; echo "${algos[*]}")
	cmd=$cmd" --boxplot "$algo_list
	$cmd
done
