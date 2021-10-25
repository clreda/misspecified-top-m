# Dealing With Misspecification In Fixed-Confidence Linear Top-m Identification

This repository is the official implementation of [Dealing With Misspecification In Fixed-Confidence
Linear Top-m Identification] (to appear in the proceedings of NIPS'21). 

## Requirements

To install requirements (Python 3.6.9):

```setup
python3 -m pip install -r requirements.txt
```

## Getting started

### Reproduce results from the paper

In order to run ExperimentXXX in the paper, do as follows

- Run command
```bash
cd experiments_scripts/
./ExperimentXXX.sh
```

- That starts the computation, when it is done, the following files are present in the results/ folder

	+ ExperimentXXX/method=[algorithm]_[list of options = values].csv

		Contains a matrix of 3 columns ("complexity": number of sampled arms, "regret": error in identification, "linearity": 1 if the algorithm considers data as linear, 0 otherwise, "running time": time in seconds for running the iteration) and XXX rows (controlled by parameter n_simu in the command) corresponding to each iteration of the algorithm.

	+ ExperimentXXX/method=[algorithm]_[list of options = values]-emp_rec.csv

		Contains a matrix of XXX columns (number of arms in the experiment, controlled by parameter K in the command), and two rows, first row being the names of the arms, and the second one being the percentage of the time a given arm was returned in the set of good arms across iterations.

	+ ExperimentXXX/params.json

		Saves in a JSON file the parameters set in the call to the code.

- PNG file ExperimentXXX/boxplot.png is created in folder boxplots/

You can only run the code to plot the boxplot from a previously run ExperimentXXX

- Run command
```bash
cd experiments_scripts
./ExperimentXXX.sh boxplot
```

ExperimentXXX won't be run, but if the corresponding results folder is present, then it creates the boxplot in folder boxplots/ExperimentXXX

### Run

Have a look at file **code/main.py** to see the arguments needed.

## Add new elements of code

- Add a new bandit by creating a new instance of class *Misspecified* in file **code/misspecified.py**
- Add a new dataset by adding a few lines of code to file **code/data.py**
- Add new types of rewards by creating a new instance of class *problem* in file **code/problems.py**
- Add new types of online learners by creating a new instance of class *Learner* in file **code/learners.py**

## Results

Please refer to the paper.

## Contributing

All of the code is under MIT license. Everyone is most welcome to submit pull requests.
