"""
Runs behavior cloning and DAgger for homework 1
作业1是运行行为克隆和DAgger算法

Functions to edit:
    1. run_training_loop
需要修改的函数:
    1. run_training_loop
"""
from tqdm import tqdm
import pickle
import os
import time
import gym

import numpy as np
import torch

from cs285.infrastructure import pytorch_util as ptu
from cs285.infrastructure import utils
from cs285.infrastructure.logger import Logger
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.policies.MLP_policy import MLPPolicySL
from cs285.policies.loaded_gaussian_policy import LoadedGaussianPolicy


# how many rollouts to save as videos to tensorboard/保存到tensorboard的视频卷轴数量
MAX_NVIDEO = 2
MAX_VIDEO_LEN = 40  # we overwrite this in the code below/我们在下面的代码中会重写这个值

MJ_ENV_NAMES = ["Ant-v4", "Walker2d-v4", "HalfCheetah-v4", "Hopper-v4"]
"""Usage
1. Behavior Cloning
python cs285/scripts/run_hw1.py \
	--expert_policy_file cs285/policies/experts/Ant.pkl \
	--env_name Ant-v4 --exp_name bc_ant --n_iter 1 \
	--expert_data cs285/expert_data/expert_data_Ant-v4.pkl

python cs285/scripts/run_hw1.py \
	--expert_policy_file cs285/policies/experts/Walker2d.pkl \
	--env_name Walker2d-v4 --exp_name bc_Walker2d --n_iter 1 \
	--expert_data cs285/expert_data/expert_data_Walker2d-v4.pkl

python cs285/scripts/run_hw1.py \
	--expert_policy_file cs285/policies/experts/HalfCheetah.pkl \
	--env_name HalfCheetah-v4 --exp_name bc_halfcheetah --n_iter 1 \
	--expert_data cs285/expert_data/expert_data_HalfCheetah-v4.pkl

python cs285/scripts/run_hw1.py \
	--expert_policy_file cs285/policies/experts/Hopper.pkl \
	--env_name Hopper-v4 --exp_name bc_hopper --n_iter 1 \
	--expert_data cs285/expert_data/expert_data_Hopper-v4.pkl


2. DAgger

python cs285/scripts/run_hw1.py \
	--expert_policy_file cs285/policies/experts/Ant.pkl \
	--env_name Ant-v4 --exp_name da_ant --n_iter 10 \
	--do_dagger --expert_data cs285/expert_data/expert_data_Ant-v4.pkl

python cs285/scripts/run_hw1.py \
	--expert_policy_file cs285/policies/experts/Walker2d.pkl \
	--env_name Walker2d-v4 --exp_name da_Walker2d --n_iter 10 \
	--do_dagger --expert_data cs285/expert_data/expert_data_Walker2d-v4.pkl

python cs285/scripts/run_hw1.py \
	--expert_policy_file cs285/policies/experts/HalfCheetah.pkl \
	--env_name HalfCheetah-v4 --exp_name da_halfcheetah --n_iter 10 \
	--do_dagger --expert_data cs285/expert_data/expert_data_HalfCheetah-v4.pkl

python cs285/scripts/run_hw1.py \
	--expert_policy_file cs285/policies/experts/Hopper.pkl \
	--env_name Hopper-v4 --exp_name da_hopper --n_iter 10 \
	--do_dagger --expert_data cs285/expert_data/expert_data_Hopper-v4.pkl


"""

def run_training_loop(params):
    """
    Runs training with the specified parameters
    (behavior cloning or dagger)

    Args:
        params: experiment parameters
    """

    #############
    ## INIT/初始化
    #############

    # Get params, create logger, create TF session/获取参数，创建日志记录器，创建TF会话
    logger = Logger(params['logdir'])

    # Set random seeds/设置随机种子
    seed = params['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    ptu.init_gpu(
        use_gpu=not params['no_gpu'],
        gpu_id=params['which_gpu']
    )

    # Set logger attributes/设置日志记录器属性
    log_video = True
    log_metrics = True

    #############
    ## ENV/环境设置
    #############

    # Make the gym environment/创建gym环境
    env = gym.make(params['env_name'], render_mode=None)
    env.reset(seed=seed)

    # Maximum length for episodes/回合的最大长度
    params['ep_len'] = params['ep_len'] or env.spec.max_episode_steps
    MAX_VIDEO_LEN = params['ep_len']

    assert isinstance(env.action_space, gym.spaces.Box), "Environment must be continuous"
    # Observation and action sizes/观察空间和动作空间的维度
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]

    # simulation timestep, will be used for video saving/仿真时间步长，用于视频保存
    if 'model' in dir(env):
        fps = 1/env.model.opt.timestep
    else:
        fps = env.env.metadata['render_fps']

    #############
    ## AGENT/智能体
    #############

    # TODO: Implement missing functions in this class./实现这个类中缺失的函数。
    actor = MLPPolicySL(
        ac_dim,
        ob_dim,
        params['n_layers'],
        params['size'],
        learning_rate=params['learning_rate'],
    )

    # replay buffer/经验回放缓冲区
    replay_buffer = ReplayBuffer(params['max_replay_buffer_size'])

    #######################
    ## LOAD EXPERT POLICY/加载专家策略
    #######################

    print('Loading expert policy from...', params['expert_policy_file'])
    expert_policy = LoadedGaussianPolicy(params['expert_policy_file'])
    expert_policy.to(ptu.device)
    print('Done restoring expert policy...')

    #######################
    ## TRAINING LOOP/训练循环
    #######################

    # init vars at beginning of training/在训练开始时初始化变量
    total_envsteps = 0
    start_time = time.time()

    for itr in range(params['n_iter']):
        print("\n\n********** Iteration %i ************"%itr)

        # decide if videos should be rendered/logged at this iteration/决定是否在此迭代中渲染/记录视频
        log_video = ((itr % params['video_log_freq'] == 0) and (params['video_log_freq'] != -1))
        # decide if metrics should be logged/决定是否记录指标
        log_metrics = (itr % params['scalar_log_freq'] == 0)

        print("\nCollecting data to be used for training...")
        if itr == 0:
            # BC training from expert data./从专家数据中训练行为克隆模型
            paths = pickle.load(open(params['expert_data'], 'rb'))
            envsteps_this_batch = 0
        else:
            # DAGGER training from sampled data relabeled by expert/从专家重新标记的采样数据进行DAgger训练
            assert params['do_dagger']
            # TODO: collect `params['batch_size']` transitions/TODO:收集 `params['batch_size']` 个转换样本
            # HINT: use utils.sample_trajectories/提示:使用utils.sample_trajectories
            # TODO: implement missing parts of utils.sample_trajectory/TODO:实现utils.sample_trajectory的缺失部分
            paths, envsteps_this_batch = utils.sample_trajectories(env, actor, params['batch_size'], params['ep_len'])

            # relabel the collected obs with actions from a provided expert policy/使用提供的专家策略重新标记收集的观察
            if params['do_dagger']:
                print("\nRelabelling collected observations with labels from an expert policy...")

                # TODO: relabel collected obsevations (from our policy) with labels from expert policy/使用专家策略的标签重新标记收集的观察
                # HINT: query the policy (using the get_action function) with paths[i]["observation"]/使用paths[i]["observation"]查询策略(使用get_action函数)
                # and replace paths[i]["action"] with these expert labels/并用这些专家标签替换paths[i]["action"]
                for path in paths:
                    path['action'] = expert_policy.get_action(path['observation'])

        total_envsteps += envsteps_this_batch
        # add collected data to replay buffer/将收集的数据添加到经验回放缓冲区
        replay_buffer.add_rollouts(paths)

        # train agent (using sampled data from replay buffer)  /训练智能体(使用经验回放缓冲区中的采样数据)
        print('\nTraining agent using sampled data from replay buffer...')
        training_logs = []
        for _ in tqdm(range(params['num_agent_train_steps_per_iter'])):

          # TODO: sample some data from replay_buffer   /从经验回放缓冲区中采样一些数据
          # HINT1: how much data = params['train_batch_size']   /数据量=params['train_batch_size']
          # HINT2: use np.random.permutation to sample random indices  /使用np.random.permutation来采样随机索引
          # HINT3: return corresponding data points from each array (i.e., not different indices from each array)   /从每个数组返回相应的数据点(即，不是每个数组的不同索引)
          # for imitation learning, we only need observations and actions.  /对于模仿学习，我们只需要观察和动作。
          indices = np.random.permutation(replay_buffer.obs.shape[0])[:params['train_batch_size']]
          ob_batch, ac_batch = replay_buffer.obs[indices], replay_buffer.acs[indices]

          # use the sampled data to train an agent/使用采样的数据训练一个智能体
          train_log = actor.update(ob_batch, ac_batch)
          training_logs.append(train_log)

        # log/save
        print('\nBeginning logging procedure...')
        if log_video:
            # save eval rollouts as videos in tensorboard event file/将评估卷轴保存为视频
            print('\nCollecting video rollouts eval')
            eval_video_paths = utils.sample_n_trajectories(
                env, actor, MAX_NVIDEO, MAX_VIDEO_LEN, True)

            # save videos/保存视频
            if eval_video_paths is not None:
                logger.log_paths_as_videos(
                    eval_video_paths, itr,
                    fps=fps,
                    max_videos_to_save=MAX_NVIDEO,
                    video_title='eval_rollouts')

        if log_metrics:
            # save eval metrics/保存评估指标
            print("\nCollecting data for eval...")
            eval_paths, eval_envsteps_this_batch = utils.sample_trajectories(
                env, actor, params['eval_batch_size'], params['ep_len'])

            logs = utils.compute_metrics(paths, eval_paths)
            # compute additional metrics/计算额外的指标
            logs.update(training_logs[-1]) # Only use the last log for now/现在只使用最后一个日志
            logs["Train_EnvstepsSoFar"] = total_envsteps
            logs["TimeSinceStart"] = time.time() - start_time
            if itr == 0:
                logs["Initial_DataCollection_AverageReturn"] = logs["Train_AverageReturn"]

            # perform the logging/执行日志记录
            for key, value in logs.items():
                print('{} : {}'.format(key, value))
                logger.log_scalar(value, key, itr)
            print('Done logging...\n\n')

            logger.flush()

        if params['save_params']:
            print('\nSaving agent params')
            actor.save('{}/policy_itr_{}.pt'.format(params['logdir'], itr))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--expert_policy_file', '-epf', type=str, required=True)  # relative to where you're running this script from/与您从何处运行此脚本有关
    parser.add_argument('--expert_data', '-ed', type=str, required=True) #relative to where you're running this script from/与您从何处运行此脚本有关
    parser.add_argument('--env_name', '-env', type=str, help=f'choices: {", ".join(MJ_ENV_NAMES)}', required=True)
    parser.add_argument('--exp_name', '-exp', type=str, default='pick an experiment name', required=True)
    parser.add_argument('--do_dagger', action='store_true')
    parser.add_argument('--ep_len', type=int,default=1000)

    parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=1000)  # number of gradient steps for training policy (per iter in n_iter)/训练策略时的梯度步数（每次训练iter步，共训练n_iter次）
    parser.add_argument('--n_iter', '-n', type=int, default=1)

    parser.add_argument('--batch_size', type=int, default=1000)  # training data collected (in the env) during each iteration/每次(在环境中)迭代期间收集的训练数据
    parser.add_argument('--eval_batch_size', type=int,
                        default=5000)  # eval data collected (in the env) for logging metrics/用于记录指标的(在环境中)评估的数据
    parser.add_argument('--train_batch_size', type=int,
                        default=100)  # number of sampled data points to be used per gradient/train step    /每个梯度/训练步骤使用的采样数据点数

    parser.add_argument('--n_layers', type=int, default=2)  # depth, of policy to be learned    /要学习的策略的深度
    parser.add_argument('--size', type=int, default=64)  # width of each layer, of policy to be learned   /要学习的策略的每一层的宽度
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)  # LR for supervised learning   /监督学习的学习率

    parser.add_argument('--video_log_freq', type=int, default=5)
    parser.add_argument('--scalar_log_freq', type=int, default=1)
    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', type=int, default=0)
    parser.add_argument('--max_replay_buffer_size', type=int, default=1000000)
    parser.add_argument('--save_params', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    # convert args to dictionary    /覆盖参数
    params = vars(args)

    ##################################
    ### CREATE DIRECTORY FOR LOGGING    /创建日志目录
    ##################################

    if args.do_dagger:
        # Use this prefix when submitting. The auto-grader uses this prefix.    /提交时使用此前缀。自动评分器使用此前缀。
        logdir_prefix = 'q2_'
        assert args.n_iter>1, ('DAGGER needs more than 1 iteration (n_iter>1) of training, to iteratively query the expert and train (after 1st warmstarting from behavior cloning).')
    else:
        # Use this prefix when submitting. The auto-grader uses this prefix.    /提交时使用此前缀。自动评分器使用此前缀。
        logdir_prefix = 'q1_'
        assert args.n_iter==1, ('Vanilla behavior cloning collects expert data just once (n_iter=1)')

    # directory for logging   /日志目录
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data')
    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    logdir = logdir_prefix + args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    ###################
    ### RUN TRAINING    
    ###################

    run_training_loop(params)


if __name__ == "__main__":
    main()
