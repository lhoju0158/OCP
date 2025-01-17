##########################################################################################
# Imports
##########################################################################################
from typing import Type

import numpy as np
import or_gym
from or_gym.envs.new.opc import OCPEnv_1

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from torch import nn
from stable_baselines3.common.evaluation import evaluate_policy
from or_gym.utils import create_env
import pandas as pd
import torch

from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
import time
from stable_baselines3.common.vec_env import DummyVecEnv

from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter


##########################################################################################
# CustomCallback for Tensorboard
##########################################################################################


class CustomTensorboardCallback(BaseCallback):
    def __init__(self, verbose=0, log_dir=None):
        super(CustomTensorboardCallback, self).__init__(verbose)
        self.writer = None
        self.log_dir = log_dir
        self.allocated_video_storage = 0
        self.allocated_video_bandwidth = 0
        self.unallocated_video_bandwidth = 0
        self.proxy_total_storage = 0
        self.proxy_total_bandwidth = 0
        self.bandwidth_variance = 0

    def _on_training_start(self):
        for output_format in self.logger.output_formats:
            if isinstance(output_format, type(self.logger.output_formats[0])):
                self.writer = output_format.writer
                break

    def _on_step(self) -> bool:
        # 환경에서 현재 상태 가져오기
        # print(f"in customtensor!!")

        ## 주의!! 이건 update_state가 된 이후에 가지고 오는 정보이다
        ## 환경 state 중 이전 환경에 대한 정보는 현재 num_timesteps에 -1을 해야 한다
        step_limit = self.training_env.get_attr("step_limit")[0]
        state = self.training_env.get_attr("state")[0]
        demand = self.training_env.get_attr("demand")[0]
        # print(f"demand = {demand}")
        proxy_state = state["proxy_state"]
        current_allocated_video_state = demand[self.num_timesteps - 1]  # 이전 환경 정보
        # print(f"current_allocated_video_state = {current_allocated_video_state}")
        # print(f"self.num_timesteps = {self.num_timesteps}")

        # current_video_state = state["current_video_state"] # update가 되서..
        current_video_state = state["current_allocated_video_state"]

        # print("========== In training.py ==========")
        # print(f"step_limit = {step_limit}\nstate = {state}")

        # tensorboard에 넣어야 할 대상
        # video 관련
        # - 할당 대상이 되는 video의 누적 storage (실제로 저장 된 storage)
        self.allocated_video_storage += current_allocated_video_state[2]
        # - 할당 대상이 되는 video의 누적 bandwidth (실제로 저장 된 bandwidth)
        self.allocated_video_bandwidth += current_allocated_video_state[1]
        # - 할당 대상이 되는 bandwidth 중 저장되지 않은 누적 bandwidth
        # print(f"===== Let's check! ======")
        # print(f"current_video_state[1] = {current_video_state[1]}, current_allocated_video_state[2] = {current_allocated_video_state[2]}")
        self.unallocated_video_bandwidth += max(
            0, round((current_video_state[1] - current_allocated_video_state[1]), 5)
        )
        # print(f"So, self.unallocated_video_bandwidth = {self.unallocated_video_bandwidth}")

        # proxy 관련
        # - 현재 proxy의 누적 storage 사용량
        self.proxy_total_storage = np.sum(proxy_state[:, 2])
        # - 현재 proxy의 누적 bandwith 사용량
        self.proxy_total_bandwidth = np.sum(proxy_state[:, 1])

        # optimize 관련련
        # - 현재 proxyd의 bandwidth 분산 계산산
        self.bandwidth_variance = np.var(proxy_state[:, 1])

        # TensorBoard에 기록
        # video 관련
        self.writer.add_scalar(
            "video/allocated_video_storage",
            self.allocated_video_storage,
            self.num_timesteps,
        )
        self.writer.add_scalar(
            "video/allocated_video_bandwidth",
            self.allocated_video_bandwidth,
            self.num_timesteps,
        )
        self.writer.add_scalar(
            "video/unallocated_video_bandwidth",
            self.unallocated_video_bandwidth,
            self.num_timesteps,
        )

        # proxy 관련
        self.writer.add_scalar(
            "proxy/proxy_total_storage",
            self.proxy_total_storage,
            self.num_timesteps,
        )
        self.writer.add_scalar(
            "proxy/proxy_total_bandwidth",
            self.proxy_total_bandwidth,
            self.num_timesteps,
        )

        # optimize 관련
        self.writer.add_scalar(
            "optimization/bandwidth_variance",
            self.bandwidth_variance,
            self.num_timesteps,
        )
        if self.num_timesteps % (step_limit) == 0:
            self._on_episode_end()
            # print("in customtensor, episode is ended")

        return True

    def _on_episode_end(self):
        # 에피소드가 끝날 때 누적 데이터 초기화
        self.allocated_video_storage = 0
        self.allocated_video_bandwidth = 0
        self.unallocated_video_bandwidth = 0
        self.proxy_total_storage = 0
        self.proxy_total_bandwidth = 0
        self.bandwidth_variance = 0
        # print("Episode ended, resetting cumulative bandwidth values.")

    def _on_training_end(self):
        self.writer.close()
        print("Training ended. TensorBoard writer closed.")


##########################################################################################
# Custom wrapper to normalize environment observations
##########################################################################################

# def mask_fn(env: OCPEnv_1):
#     return env.valid_action_mask()


def mask_fn(env: OCPEnv_1):
    try:
        if isinstance(env, DummyVecEnv):  # for saved model load
            masks = env.env_method("valid_action_mask")
            return np.array(masks).squeeze()
        else:  # for training model
            return env.valid_action_mask()
    except Exception as e:
        print(f"Error in mask_fn: {e}")
        # 환경의 실제 타입 출력
        print(f"Environment type: {type(env)}")
        if isinstance(env, DummyVecEnv):
            print(f"Inner environment type: {type(env.envs[0])}")
        raise e


##########################################################################################
# Training function
##########################################################################################
def train_model(
    model_path: str,
    tensorboard_log: str = "./OCP_test",
    algorithm_class: Type[OnPolicyAlgorithm] = MaskablePPO,
    gamma: float = 0.99,
    learning_rate: float = 0.0003,
    normalize_env: bool = True,
    activation_fn: Type[nn.Module] = nn.ReLU,
    net_arch=[256, 256],
    n_times: int = 1000 * 100,
    verbose: int = 1,
    seed: int = 317,
) -> OnPolicyAlgorithm:

    env_config = {"seed": seed}
    env: OCPEnv_1 = or_gym.make(
        "OCP-v0",
        env_config=env_config,
        disable_env_checker=True,
    ).unwrapped
    # .unwrappd를 추가해서 기본 환경에 접근하기
    # autoreset = True를 통해 자동

    # print(f"The type of environment before wrapping: {type(env)}")  # 환경 타입 확인
    # The type of environment before wrapping: <class 'or_gym.envs.new.opc.OCPEnv_1'>

    # Wrapping the environment
    env = ActionMasker(env, mask_fn)  # ActionMasker 추가
    env = DummyVecEnv([lambda: env])  # DummyVecEnv로 래핑
    env.reset()  # 초기화

    # print(
    #     f"The type of environment after wrapping: {type(env)}"
    # )  # 최종 래핑된 환경 타입 확인
    # The type of environment after wrapping: <class 'stable_baselines3.common.vec_env.DummyVecEnv'>

    model = MaskablePPO(
        "MultiInputPolicy",  # 관찰공간이 dictionary 형태일 경우 MlpPolicy가 아니라 MultiInputPolicy를 사용해야 한다
        env,
        n_steps=n_times,
        gamma=gamma,
        learning_rate=learning_rate,
        policy_kwargs=dict(net_arch=net_arch, activation_fn=activation_fn),
        tensorboard_log=tensorboard_log,
        verbose=verbose,
        seed=seed,
        # device="cuda",  # GPU 사용
    )
    # ActionMasker는 단일 환경에 대해 적용되어야 한다
    # DummyVecEnv는 여러 환경을 벡터화하는 래퍼이므로, ActionMasker가 적용된 후에 래핑되어야 한다

    # print(f"the availablity of GPU is {torch.cuda.is_available()}")
    callback = CustomTensorboardCallback(verbose=verbose)
    model.learn(total_timesteps=n_times, callback=callback, progress_bar=True)

    model.save(model_path)
    print("Model is saved successfully")
    env.close()
    return model


##########################################################################################
# Testing function
##########################################################################################


def test_model(model_path: str, n_episodes: int = 10, seed: int = 317):
    env_config = {"seed": seed}
    env: OCPEnv_1 = or_gym.make("OCP-v0", env_config=env_config)
    env = ActionMasker(env, mask_fn)
    env = DummyVecEnv([lambda: env])  ##################################
    env.reset()
    model = MaskablePPO.load(model_path)
    print("Model loaded successfully")

    # Evaluate the model
    total_rewards = []
    episode_durations = []

    for i in range(n_episodes):
        start_time = time.time()
        # obs, action_masks = env.reset()
        obs = env.reset()
        total_reward = 0

        while True:
            action, _states = model.predict(obs, action_masks=mask_fn(env))
            # exit(0)
            obs, rewards, done, info = env.step(action)  # 여기 기대값 왜 5개 아닌가
            # print(f"in predict, reward = {rewards}") # step의 reward와 동일한 값
            total_reward += rewards

            if done:
                end_time = time.time()
                episode_durations.append(end_time - start_time)
                print(f"Episode {i} finished. Total Reward: {total_reward}")
                break

        total_rewards.append(total_reward)

    avg_reward = np.mean(total_rewards)
    avg_duration = np.mean(episode_durations)
    print(f"Avg Reward: {avg_reward}, Avg Duration: {avg_duration}s")
