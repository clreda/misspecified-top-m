#!/bin/bash

#######################################################
## APPLICATION TO REAL-LIFE DATA (DRUG REPURPOSING)  ##
## (TOP-5 PROBLEM)                                   ##
#######################################################

cd ../code

njobs=$(python -c "import multiprocessing;print(multiprocessing.cpu_count()-1)")
nsimu=100
K=10
N=5
m=5
sigma=1.0
delta=0.05
beta="heuristic"
problem="gaussian"
algos=("LinGapE" "MisLid" "LUCB")
data_type="linearized_dr"
epsilon=0.06
c=0.02
py_cmd="python3"
folder="ExperimentD"
for ALGO in "${algos[@]}"; do
	if [ "$ALGO" == "LUCB" ];
	then
		epsilon=0.06
	else
		epsilon=0
	fi
	echo $folder": "$ALGO" drug repurposing for epilepsy "$data_type" K="$K" N="$N" c="$c
	cmd=$py_cmd" main.py --data_type $data_type --c $c --problem_type $problem --beta_linear $beta --delta $delta --m $m --sigma $sigma --K $K --N $N --sigma $sigma --nsimu $nsimu --method $ALGO --n_jobs $njobs --exp_name "$folder" --epsilon "$epsilon
	echo $cmd
	if [ "$1" == "" ];
	then
		$cmd
	fi
done
algo_list=$(IFS=, ; echo "${algos[*]}")
cmd=$cmd" --boxplot "$algo_list
$cmd

## Run with the tricks
folder_tricks=$folder"_tricks"
mkdir -p "../results/"$folder_tricks
ALGO="MisLid"
nm="beta_linear="$beta"_delta="$delta"_c="$c"_sigma="$sigma
cp "../results/"$folder"/method="$ALGO"_"$nm".csv" "../results/"$folder_tricks"/method="$ALGO"(default)_"$nm".csv"
learner_names=("AdaHedge" "Greedy")
for LEARNER in "${learner_names[@]}"; do
	subfolder=$folder"_"$LEARNER
	echo $folder": "$ALGO" drug repurposing for epilepsy with tricks (learner: "$LEARNER") "$data_type" K="$K" N="$N" c="$c
	cmd=$py_cmd" main.py --data_type $data_type --c $c --problem_type $problem --beta_linear $beta --delta $delta --m $m --sigma $sigma --K $K --N $N --sigma $sigma --nsimu $nsimu --method $ALGO --n_jobs $njobs --epsilon "$epsilon
	opt="--gain_type empirical --learner_name "$LEARNER" --subsample 1 --geometric_factor 1.2"
	cmd_=$cmd" "$opt" --exp_name "$subfolder
	echo $cmd_
	if [ "$1" == "" ];
	then
		$cmd_
	fi
	other_algos=("LUCB" "LinGapE")
	for OTHER in "${other_algos[@]}"; do
		cp "../results/"$folder"/method="$OTHER"_"$nm".csv" "../results/"$subfolder
	done
	cp "../results/"$subfolder"/method="$ALGO"_"$nm".csv" "../results/"$folder_tricks"/method="$ALGO"("$LEARNER")_"$nm".csv"
	cmdb=$cmd" --boxplot None --exp_name "$subfolder
	$cmdb
done
cmd=$cmd" --boxplot None --exp_name "$folder_tricks
$cmd
