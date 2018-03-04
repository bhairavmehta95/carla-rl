#!/usr/bin/env python3
import argparse
from baselines.common.cmd_util import mujoco_arg_parser
from baselines import bench, logger

def train(env_id, num_timesteps, seed, nenvs):
    from baselines.common import set_global_seeds
    from baselines.common.vec_env.vec_normalize import VecNormalize
    from baselines.ppo2 import ppo2
    from baselines.ppo2.policies import CnnPolicy
    import gym
    import tensorflow as tf
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()

    def make_env(i, gpu_num):
        def make_carla():
            from environment.straight_env import StraightDriveEnv
            env = StraightDriveEnv(port=i, gpu_num=gpu_num)
            return env
        return make_carla

    env = DummyVecEnv([make_env(8013+i*3, i % 3 + 1) for i in range(nenvs)])
    env = VecNormalize(env)
    
    set_global_seeds(seed)
    policy = CnnPolicy
    ppo2.learn(policy=policy, env=env, nsteps=20, nminibatches=32,
        lam=0.95, gamma=0.99, noptepochs=10, log_interval=1,
        ent_coef=0.0,
        lr=3e-4,
        cliprange=0.2,
        total_timesteps=num_timesteps)


def main():
    args = mujoco_arg_parser().parse_args()
    logger.configure(dir=args.logdir, format_strs=['tensorboard', 'stdout'])
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed, nenvs=8)


if __name__ == '__main__':
    main()
