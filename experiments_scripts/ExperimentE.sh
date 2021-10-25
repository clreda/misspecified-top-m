#!/bin/bash

########################################################
## APPLICATION TO REAL-LIFE DATA (ONLINE RECOMMENDAT°)##
## (TOP-4 PROBLEM)                                    ##
########################################################

cd ../code

njobs=$(python -c "import multiprocessing;print(multiprocessing.cpu_count()-1)")
nsimu=100
K=103
N=8
m=4
sigma=1
delta=0.05
beta="heuristic"
problem="gaussian"
algos=("LinGapE" "LUCB" "MisLid")
data_type="linearized_lastfm"
epsilon=0 # gap value is 0.022
c=0.206
py_cmd="python3"
learners=("Greedy" "AdaHedge")
folder="ExperimentE"
for ALGO in "${algos[@]}"; do
	echo $folder": "$ALGO" recommendation (LastFM) "$data_type" K="$K" N="$N" c="$c
	cmd=$py_cmd" main.py --data_type $data_type --c $c --problem_type $problem --beta_linear $beta --delta $delta --m $m --sigma $sigma --K $K --N $N --sigma $sigma --nsimu $nsimu --method $ALGO --n_jobs $njobs --exp_name "$folder" --epsilon "$epsilon
	echo $cmd
	if [ "$1" == "" ];
	then
		$cmd
	fi
done

nm="beta_linear="$beta"_delta="$delta"_c="$c"_sigma=1.0"
folder_tricks=$folder"_tricks"
mkdir -p $folder_tricks
for LEARNER in "${learners[@]}"; do
	ALGO="MisLid"
	subfolder=$folder"_"$LEARNER
	echo $subfolder": "$ALGO" recommendation (LastFM) "$data_type" K="$K" N="$N" c="$c
	cmd__=$py_cmd" main.py --data_type $data_type --c $c --problem_type $problem --beta_linear $beta --delta $delta --m $m --sigma $sigma --K $K --N $N --sigma $sigma --nsimu $nsimu --method $ALGO --n_jobs $njobs --epsilon "$epsilon
	opt="--gain_type empirical --learner_name "$LEARNER" --subsample 1 --geometric_factor 1.2 --exp_name "$subfolder
	cmd_=$cmd__" "$opt
	echo $cmd_
	if [ "$1" == "" ];
	then
		$cmd_
	fi
	if [ "$LEARNER" == "AdaHedge" ];
	then
		cp "../results/"$subfolder"/method="$ALGO"_"$nm".csv" "../results/"$folder
	fi
	cp "../results/"$subfolder"/method="$ALGO"_"$nm".csv" "../results/"$folder_tricks"/method="$ALGO"("$LEARNER")_"$nm".csv"
done

cmd=$cmd" --boxplot None"
$cmd

cmd=$cmd__" --boxplot None --exp_name "$folder_tricks
$cmd
