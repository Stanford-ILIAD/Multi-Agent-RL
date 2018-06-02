import sys
import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--step_size', type=float, default=0.005)
parser.add_argument('--max_backtracks', type=int, default=10)
parser.add_argument('--resume_from', type=str,default='../rllab/data/experiment_2018_05_26_22_03_10_979751_PDT_26be4/itr_999.pkl')
parser.add_argument('--n_parallel', type=int, default=1)
parser.add_argument('--num_iters',type=int,default=1)
args = parser.parse_args()

t_iters = 1000 + args.num_iters
for evader_param1 in np.arange(0.1,0.65,0.05):
	os.system("python3 run_waterworldsimple.py rllab --control centralized "+ 
		"--evader_param1 " + str(evader_param1) +  " --evader_param2 0.2 --step_size " + str(args.step_size) + 
		" --max_backtracks "+  str(args.max_backtracks)  +" --full_observability True " + 
		" --resume_from " +  args.resume_from +
		" --n_iter " + str(t_iters) + " --n_parallel " + str(args.n_parallel))
