"""
Defines a pytorch policy as the agent's actor   /定义一个pytorch策略作为agent的actor

Functions to edit:
    2. forward
    3. update
"""

import abc
import itertools
from typing import Any
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu
from cs285.policies.base_policy import BasePolicy


def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int
) -> nn.Module:
    """
        Builds a feedforward neural network     /建立一个前馈神经网络

        arguments:  /参数
            n_layers: number of hidden layers   /隐藏层的数量
            size: dimension of each hidden layer    /每个隐藏层的维度
            activation: activation of each hidden layer /每个隐藏层的激活函数

            input_size: size of the input layer  /输入层的大小
            output_size: size of the output layer   /输出层的大小
            output_activation: activation of the output layer   /输出层的激活函数

        returns:
            MLP (nn.Module)
    """
    layers = []
    in_size = input_size
    for _ in range(n_layers):
        layers.append(nn.Linear(in_size, size))
        layers.append(nn.Tanh())
        in_size = size
    layers.append(nn.Linear(in_size, output_size))

    mlp = nn.Sequential(*layers)
    return mlp


class MLPPolicySL(BasePolicy, nn.Module, metaclass=abc.ABCMeta):
    """
    Defines an MLP for supervised learning which maps observations to continuousactions.    
    为有监督学习定义一个MLP, 将观察映射到连续动作

    Attributes
    ----------
    mean_net: nn.Sequential
        A neural network that outputs the mean for continuous actions   /一个输出连续动作均值的神经网络
    logstd: nn.Parameter
        A separate parameter to learn the standard deviation of actions /一个单独的参数来学习动作的标准差

    Methods
    -------
    forward:
        Runs a differentiable forwards pass through the network   /通过网络运行可微分的前向传递
    update:
        Trains the policy with a supervised learning objective  /使用监督学习目标训练策略
    """
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        self.mean_net = build_mlp(
            input_size=self.ob_dim,
            output_size=self.ac_dim,
            n_layers=self.n_layers, size=self.size,
        )
        self.mean_net.to(ptu.device)
        self.logstd = nn.Parameter(

            torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
        )
        self.logstd.to(ptu.device)
        self.optimizer = optim.Adam(
            itertools.chain([self.logstd], self.mean_net.parameters()),
            self.learning_rate
        )

    def save(self, filepath):
        """
        :param filepath: path to save MLP   
        """
        torch.save(self.state_dict(), filepath)

    def forward(self, observation: torch.FloatTensor) -> distributions.Distribution:
        """
        Defines the forward pass of the network /定义网络的前向传递

        :param observation: observation(s) to query the policy  /要查询策略的observation
        :return:
            action: sampled action(s) from the policy   /从策略中抽样的动作
        """

        # TODO: implement the forward pass of the network.
        # You can return anything you want, but you should be able to differentiate
        # through it. For example, you can return a torch.FloatTensor. You can also
        # return more flexible objects, such as a
        # `torch.distributions.Distribution` object. It's up to you!

        # TODO: 实现网络的前向传播。
        # 你可以返回任何想要的内容，但必须能够对其进行梯度计算。
        # 例如，你可以返回一个 torch.FloatTensor。
        # 你也可以返回更灵活的对象，比如 `torch.distributions.Distribution` 对象。
        # 具体返回什么由你决定！

        # 通过均值与标准差生成正态分布，返回的是分布对象而不是具体的值，这样支持探索
        # 可以通过act_dist.rsample()来获取可微分的采样
        # 这样处理便于后续计算策略梯度和其他相关量
        mean = self.mean_net(observation)
        act_dist = distributions.Normal(mean, torch.exp(self.logstd))
        return act_dist

    def update(self, observations, actions):
        """
        Updates/trains the policy   更新/训练策略

        :param observations: observation(s) to query the policy     /要查询策略的observation
        :param actions: actions we want the policy to imitate       /我们希望策略模仿的动作
        :return:
            dict: 'Training Loss': supervised learning loss         /监督学习损失
        """
        # TODO: update the policy and return the loss               /更新策略并返回损失
        
        self.optimizer.zero_grad()                                  # 梯度清零
        act_dist = self.forward(ptu.from_numpy(observations))       # 通过网络得到动作分布
        loss = -act_dist.log_prob(ptu.from_numpy(actions)).mean()   # log_prob计算动作的对数概率，mean计算均值
        loss.backward()                                             # 反向传播
        self.optimizer.step()                                       # 更新参数

        return {
            # You can add extra logging information here, but keep this line    /你可以在这里添加额外的日志信息，但请保留这一行
            'Training Loss': ptu.to_numpy(loss),
        }
