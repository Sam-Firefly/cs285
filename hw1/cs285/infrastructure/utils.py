"""A
Some miscellaneous utility functions
一些杂项实用函数

Functions to edit:
    1. sample_trajectory
"""

from collections import OrderedDict
import cv2
import numpy as np
import time

from cs285.infrastructure import pytorch_util as ptu


def sample_trajectory(env, policy, max_path_length, render=False):
    """
    Sample a rollout in the environment from a policy.
    从策略中在环境中采样一个轨迹
    """
    
    # initialize env for the beginning of a new rollout   /初始化一个新轨迹的环境
    ob =  env.reset() # TODO: initial observation after resetting the env   / TODO: 重置环境后的初始化observation

    # init vars  /初始化变量
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    while True:

        # render image of the simulated env  /渲染模拟环境的图像
        if render:
            if hasattr(env, 'sim'):
                img = env.sim.render(camera_name='track', height=500, width=500)[::-1]
            else:
                img = env.render(mode='single_rgb_array')
            image_obs.append(cv2.resize(img, dsize=(250, 250), interpolation=cv2.INTER_CUBIC))
    
        # TODO use the most recent ob to decide what to do  /TODO 用最近的ob来决定要做什么
        ob_tensor = ptu.from_numpy(ob)
        ac_tensor = policy.forward(ob_tensor).rsample() # HINT: this is a numpy array /提示：这是一个numpy数组
        ac = ptu.to_numpy(ac_tensor) # ac = ac[0]

        # TODO: take that action and get reward and next ob /TODO：执行该动作并获得奖励和下一个ob
        next_ob, rew, done, _ = env.step(ac)
        
        # TODO rollout can end due to done, or due to max_path_length   /TODO 轨迹可能因为任务完成完成或达到最大长度而结束
        steps += 1
        if steps >= max_path_length or done:
            rollout_done = 1 # HINT: this is either 0 or 1   /提示：这是0或1
        else :
            rollout_done = 0

        # record result of taking that action   /记录执行该动作的结果
        obs.append(ob)
        acs.append(ac)
        rewards.append(rew)
        next_obs.append(next_ob)
        terminals.append(rollout_done)

        ob = next_ob # jump to next timestep  /跳到下一个时间步

        # end the rollout if the rollout ended  /如果轨迹结束，则结束轨迹
        if rollout_done:
            break

    return {"observation" : np.array(obs, dtype=np.float32),
            "image_obs" : np.array(image_obs, dtype=np.uint8),
            "reward" : np.array(rewards, dtype=np.float32),
            "action" : np.array(acs, dtype=np.float32),
            "next_observation": np.array(next_obs, dtype=np.float32),
            "terminal": np.array(terminals, dtype=np.float32)}


def sample_trajectories(env, policy, min_timesteps_per_batch, max_path_length, render=False):
    """
    Collect rollouts until we have collected min_timesteps_per_batch steps.
    收集轨迹直到达到每批次最小时间步数
    """

    timesteps_this_batch = 0
    paths = []
    while timesteps_this_batch < min_timesteps_per_batch:   

        #collect rollout
        path = sample_trajectory(env, policy, max_path_length, render)
        paths.append(path)

        #count steps
        timesteps_this_batch += get_pathlength(path)

    return paths, timesteps_this_batch


def sample_n_trajectories(env, policy, ntraj, max_path_length, render=False):
    """
    Collect ntraj rollouts.
    收集ntraj个轨迹
    """

    paths = []
    for i in range(ntraj):
        # collect rollout
        path = sample_trajectory(env, policy, max_path_length, render)
        paths.append(path)
    return paths


########################################
########################################


def convert_listofrollouts(paths, concat_rew=True):
    """
        Take a list of rollout dictionariesand return separate arrays,
        where each array is a concatenation of that array from across the rollouts
        将一个轨迹字典列表转换为单独的数组，其中每个数组是跨轨迹的该数组的连接
    """
    observations = np.concatenate([path["observation"] for path in paths])
    actions = np.concatenate([path["action"] for path in paths])
    if concat_rew:
        rewards = np.concatenate([path["reward"] for path in paths])
    else:
        rewards = [path["reward"] for path in paths]
    next_observations = np.concatenate([path["next_observation"] for path in paths])
    terminals = np.concatenate([path["terminal"] for path in paths])
    return observations, actions, rewards, next_observations, terminals


########################################
########################################
            

def compute_metrics(paths, eval_paths):
    """
    Compute metrics for logging.
    计算用于记录的指标
    """

    # returns, for logging  /返回值，用于记录
    train_returns = [path["reward"].sum() for path in paths]
    eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]

    # episode lengths, for logging  /用于记录的轨迹长度
    train_ep_lens = [len(path["reward"]) for path in paths]
    eval_ep_lens = [len(eval_path["reward"]) for eval_path in eval_paths]

    # decide what to log    /决定要记录什么
    logs = OrderedDict()
    logs["Eval_AverageReturn"] = np.mean(eval_returns)
    logs["Eval_StdReturn"] = np.std(eval_returns)
    logs["Eval_MaxReturn"] = np.max(eval_returns)
    logs["Eval_MinReturn"] = np.min(eval_returns)
    logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

    logs["Train_AverageReturn"] = np.mean(train_returns)
    logs["Train_StdReturn"] = np.std(train_returns)
    logs["Train_MaxReturn"] = np.max(train_returns)
    logs["Train_MinReturn"] = np.min(train_returns)
    logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

    return logs


############################################
############################################


def get_pathlength(path):
    return len(path["reward"])
