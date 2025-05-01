#!/bin/bash

decpy_path="/mnt/c/MAROUAN/Desktop/PFE/decentralizepy/eval/" # Path to eval folder
graph=fullyConnected_16.edges # Absolute path of the graph file generated using the generate_graph.py script
run_path="/mnt/c/MAROUAN/Desktop/PFE/decentralizepy/eval/data" # Path to the folder where the graph and config file will be copied and the results will be stored
config_file=config_EL.ini
cp $graph $config_file $run_path

env_python="/mnt/c/Users/MAROUAN/AppData/Local/Programs/Python/Python310/python.exe"
machines=1 # number of machines in the runtime
iterations=50
test_after=10
eval_file=testingEL_Oracle.py # decentralized driver code (run on each machine)
log_level=INFO # DEBUG | INFO | WARN | CRITICAL

m=0 # machine id corresponding consistent with ip.json
echo M is $m

procs_per_machine=16 # 16 processes on 1 machine
echo procs per machine is $procs_per_machine

log_dir=$run_path/$(date '+%Y-%m-%dT%H-%M')/machine$m # in the eval folder
mkdir -p $log_dir

$env_python $eval_file -ro 0 -tea $test_after -ld $log_dir -mid $m -ps $procs_per_machine -ms $machines -is $iterations -gf $graph -ta $test_after -cf $config_file -ll $log_level -wsd $log_dir