#!/usr/bin/env python
#
# File: vis_waterworld.py
#
# Created: Thursday, July 14 2016 by rejuvyesh <mail@rejuvyesh.com>
#
from __future__ import absolute_import, print_function

import argparse
import json
import pprint
import os
import os.path
import pickle
from gym import spaces
import h5py
import numpy as np
import tensorflow as tf

import rltools.algos
import rltools.log
import rltools.util
import rltools.samplers
from madrl_environments import ObservationBuffer
from madrl_environments.pursuit import WaterWorld
from rltools.baselines.linear import LinearFeatureBaseline
from rltools.baselines.mlp import MLPBaseline
from rltools.baselines.zero import ZeroBaseline
from rltools.policy.gaussian import GaussianMLPPolicy
import matplotlib.pyplot as plt

from vis import Evaluator, Visualizer, FileHandler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str)  # defaultIS.h5/snapshots/iter0000480
    parser.add_argument('--vid', type=str, default='/tmp/madrl.mp4')
    parser.add_argument('--deterministic', action='store_true', default=False)
    parser.add_argument('--heuristic', action='store_true', default=False)
    parser.add_argument('--evaluate', action='store_true', default=False)
    parser.add_argument('--n_trajs', type=int, default=10)
    parser.add_argument('--n_steps', type=int, default=500)
    parser.add_argument('--same_con_pol', action='store_true')
    parser.add_argument('--param', type=int, default=1)
    args = parser.parse_args()
    print(args.param)
    
    ret_map = {}

    fh = FileHandler(args.filename)
    print(args.param)
    env = WaterWorld(radius=fh.train_args['radius'],
                      food_reward=fh.train_args['food_reward'],
                      evader_params = np.array([args.param * 0.05,0.05]))

    if fh.train_args['buffer_size'] > 1:
        env = ObservationBuffer(env, fh.train_args['buffer_size'])

    hpolicy = None
    if args.heuristic:
        from heuristics.waterworld import WaterworldHeuristicPolicy
        hpolicy = WaterworldHeuristicPolicy(env.agents[0].observation_space,
                                            env.agents[0].action_space)

    if args.evaluate:
        minion = Evaluator(env, fh.train_args, args.n_steps, args.n_trajs, args.deterministic,
                           'heuristic' if args.heuristic else fh.mode)
        # print(minion['ret'])
        evr = minion(fh.filename, file_key=fh.file_key, same_con_pol=args.same_con_pol,
                     hpolicy=hpolicy)
        try:
            ret_map=pickle.load(open("values2.p","rb"))
        except:
            ret_map = {}
        param_ = float(0.05 * args.param)
        print(param_)
        print(evr)
        ret_map[param_] = evr['ret']
        pickle.dump(ret_map, open("values3.p","wb"))
        # print(evr)
        from tabulate import tabulate
        # print(tabulate(evr, headers='keys'))
    else:
        minion = Visualizer(env, fh.train_args, args.n_steps, args.n_trajs, args.deterministic,
                            'heuristic' if args.heuristic else fh.mode)
        rew, info = minion(fh.filename, file_key=fh.file_key, vid=args.vid, hpolicy=hpolicy)
        pprint.pprint(rew)
        pprint.pprint(info)

    # plt.plot(ret_map.keys(), ret_map.values(),"ro")
    # plt.show()


if __name__ == '__main__':
    main()
