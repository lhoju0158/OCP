import numpy as np
import gymnasium as gym
from or_gym.utils import assign_env_config
from gymnasium import spaces

class OCPEnv_1(gym.Env):
    def __init__(self, *args, **kwargs):
        # 
        # 캐시와 대역폭 용량 초기화
        self.cache_capacity = 1  # 각 노드의 캐시 용량
        self.bandwidth_capacity = 1  # 각 노드의 대역폭 용량
        self.t_interval = 20  # 시간 간격
        self.tol = 1e-5  # 허용 오차 (수치 계산의 안정성을 위한)
        self.step_limit = int(60 * 24 / self.t_interval)  # 하루를 시간 간격으로 나눈 스텝 수
        self.n_nodes = 50  # 캐시 노드(프록시)의 수
        self.load_idx = np.array([1, 2])  # 대역폭과 저장 공간 인덱스
        self.seed = 0  # 난수 시드 초기화
        self.mask = True  # 마스크를 사용해 유효한 액션을 필터링할지 여부
        assign_env_config(self, kwargs)
        self.action_space = spaces.Discrete(self.n_nodes)  # 행동 공간: 특정 노드를 선택
        self.valid_actions = [1] * self.n_nodes  # 유효한 액션 (초기값은 모든 노드가 가능함)

        # 관찰 공간 설정 (마스킹을 통해 유효한 행동을 제한할 수 있음)
        if self.mask:
            self.observation_space = spaces.Dict({
                "action_mask": spaces.Box(0, 1, shape=(self.n_nodes,)),  # 유효한 액션을 표시하는 마스크
                "avail_actions": spaces.Box(0, 1, shape=(self.n_nodes,)),  # 가능한 액션
                "state": spaces.Box(0, 1, shape=(self.n_nodes + 1, 3))  # 현재 노드 상태 + 수요
            })
        else:
            self.observation_space = spaces.Box(0, 1, shape=(self.n_nodes + 1, 3))  # 마스크 사용 안할 시 전체 상태
        self.reset()
        
    def _RESET(self):
        # 환경 초기화 시 수요를 생성하고 초기 상태를 설정
        self.demand = self.generate_demand()  # 각 스텝에서 필요한 대역폭 및 저장 공간 수요 생성
        self.current_step = 0  # 현재 스텝 초기화
        self.state = {
            "action_mask": np.ones(self.n_nodes),  # 초기에는 모든 액션이 유효함
            "avail_actions": np.ones(self.n_nodes),  # 사용 가능한 액션
            "state": np.vstack([
                np.zeros((self.n_nodes, 3)),  # 각 노드의 초기 상태 (비활성화됨)
                self.demand[self.current_step]])  # 현재 스텝의 수요 추가
        }
        self.assignment = {}  # 각 스텝마다 수행된 액션을 기록
        self.valid_actions = [1] * self.n_nodes  # 모든 액션을 유효로 초기화
        return self.state, {}
    
    def _STEP(self, action):
        # 주어진 액션을 수행하여 상태를 업데이트하고 보상 계산
        done = False
        truncated = False
        node_state = self.state["state"][:-1]  # 마지막 행을 제외한 노드 상태
        demand = self.state["state"][-1, 1:]  # 현재 스텝에서 필요한 대역폭 및 저장 용량
        
        if action < 0 or action >= self.n_nodes:
            raise ValueError("Invalid action: {}".format(action))  # 유효하지 않은 액션 처리
            
        elif any(node_state[action, 1:] + demand > 1 + self.tol):
            # 수요가 선택된 노드의 용량을 초과할 경우, 페널티 부여 및 에피소드 종료
            reward = -1000
            done = True
        else:
            if node_state[action, 0] == 0:
                # 선택된 노드가 비활성 상태인 경우, 활성화
                node_state[action, 0] = 1
            node_state[action, self.load_idx] += demand  # 노드에 수요 할당
            reward = np.sum(node_state[:, 0] * (node_state[:, 1:].sum(axis=1) - 2))  # 보상 계산
            self.assignment[self.current_step] = action  # 수행된 액션 기록
            
        self.current_step += 1
        if self.current_step >= self.step_limit:
            done = True  # 스텝이 제한을 초과하면 에피소드 종료
        self.update_state(node_state)
        return self.state, reward, done, truncated, {}
    
    def update_state(self, node_state):
        # 노드 상태 업데이트 및 액션 마스크 재설정
        step = self.current_step if self.current_step < self.step_limit else self.step_limit - 1
        data_center = np.vstack([node_state, self.demand[step]])  # 노드 상태와 현재 수요 병합
        data_center = np.where(data_center > 1, 1, data_center)  # 용량 초과 부분은 최대치로 클리핑
        self.state["state"] = data_center

        # 현재 시점에서 유효한 액션 마스크 생성
        if self.mask:
            action_mask = (node_state[:, 1:] + self.demand[step, 1:]) <= 1
            self.valid_actions = (action_mask.sum(axis=1) == 2).astype(int)  # 수용 가능한 노드를 1로 표시
            self.state["action_mask"] = self.valid_actions
        else:
            self.valid_actions = [1] * self.n_nodes
            self.state["action_mask"] = np.ones(self.n_nodes)

    def sample_action(self):
        # 랜덤한 액션 샘플링
        return self.action_space.sample()

    def generate_demand(self):
        # 비디오 수요 생성 (예: 대역폭 및 저장 공간 필요량)
        n = self.step_limit
        zipf_skew = 0.7  # Zipf 분포의 스큐 파라미터
        mem_probs = np.array([0.12, 0.165, 0.328, 0.287, 0.064, 0.036])  # 메모리 수요 분포
        mem_bins = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0])  # 메모리 수요 범주
        cpu_demand = np.random.normal(loc=0.5, scale=0.1, size=n)  # 평균 0.5, 표준편차 0.1로 설정된 CPU 수요
        cpu_demand = np.clip(cpu_demand, 0, 1)  # 0과 1 사이로 값 제한
        mem_demand = np.random.choice(mem_bins, p=mem_probs, size=n)  # 메모리 수요 샘플링
        return np.vstack([np.arange(n) / n, cpu_demand, mem_demand]).T

    def step(self, action):
        # 외부에서 호출되는 실제 스텝 함수
        return self._STEP(action)

    def reset(self, **kwargs):
        # 환경을 초기 상태로 재설정
        return self._RESET()

    def valid_action_mask(self):
        # 현재 상태에서 유효한 액션 마스크 반환
        return self.valid_actions
