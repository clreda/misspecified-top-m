#!/bin/bash

#######################################################
## VARIATION DUE TO GAP BETWEEN Cdata and C          ##
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
beta="heuristic"
problem="gaussian"
ALGO="MisLid"
c_values=(0.5 1 2)
c_data=1
# gap mu_3-mu_4 ~ 0.818-0.418 = 0.4
M=2
data_type="deviation_Linf"
py_cmd="python3"
folder="ExperimentB"
for C in "${c_values[@]}"; do
	echo $folder": "$ALGO" c_data="$c_data" c="$C
	cmd=$py_cmd" main.py --data_type $data_type --c $C --c_data $c_data --problem_type $problem --beta_linear $beta --delta $delta --m $m --sigma $sigma --K $K --N $N --sigma $sigma --nsimu $nsimu --method $ALGO --n_jobs $njobs --M "$M
	cmd_=$cmd" --exp_name "$folder"_c="$C"_c_data="$c_data
	echo $cmd_
	if [ "$1" == "" ];
	then
		$cmd_
	fi
	mkdir -p "../results/"$folder"/"
	if [ "$C" == "1" ] || [ "$C" == "2" ];
	then 
		C_f=$C".0"
	else
		C_f=$C
	fi
	cmdp="cp ../results/"$folder"_c="$C"_c_data="$c_data"/method="$ALGO"_beta_linear="$beta"_delta="$delta"_c="$C_f"_sigma=1.0.csv ../results/"$folder"/method="$ALGO"(eps="$C")_.csv"
	$cmdp
done
cmdb=$cmd" --exp_name "$folder" --boxplot None"
$cmdb
