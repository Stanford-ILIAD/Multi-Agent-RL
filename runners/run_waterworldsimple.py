#!/usr/bin/env python
#
# File: run_waterworld.py
#
# Created: Wednesday, August 31 2016 by rejuvyesh <mail@rejuvyesh.com>
#
from runners import RunnerParser
from runners.curriculum import Curriculum

from madrl_environments.pursuit import WaterWorld
from madrl_environments import StandardizedEnv, ObservationBuffer

# yapf: disable
ENV_OPTIONS = [
    ('radius', float, 0.015, 'Radius of agents'),
    ('food_reward', float, 10, ''),
    ('control_penalty', float, -0.5, ''),
    ('buffer_size', int, 1, ''),
    ('curriculum', str, None, ''),
]
# yapf: enable

def main(parser):
    mode = parser._mode
    args = parser.args
    env = WaterWorld(radius=args.radius, food_reward=args.food_reward, control_penalty=args.control_penalty)
    if args.buffer_size > 1:
        env = ObservationBuffer(env, args.buffer_size)

    if mode == 'rllab':
        from runners.rurllab import RLLabRunner
        run = RLLabRunner(env, args)
    elif mode == 'rltools':
        from runners.rurltools import RLToolsRunner
        run = RLToolsRunner(env, args)
    else:
        raise NotImplementedError()

    if args.curriculum:
        curr = Curriculum(args.curriculum)
        run(curr)
    else:
        run()


if __name__ == '__main__':
    main(RunnerParser(ENV_OPTIONS))
