import numpy as np
import gymnasium as gym
from or_gym.utils import assign_env_config
from gymnasium import spaces
import inspect
from scipy.optimize import minimize
import itertools
# lib list
# gymnasium         - 0.28.1 => 1.0.0
# sb3-contrib       - 2.1.0 => 2.4.0
# stable_baselines3 - 2.3.2 => 2.4.0
# stable-baselines3[extra]

# shimmy            - 1.3.0 => 2.0.0

# venv9_v2 기존 버전
# venv9 lib update한 버전

# bandwidth => 할당 이슈 비교하기, 이건 단순 copy가 아니라 다르게 해야 한다
# masking => bandwidth / storage 이건 storage =>
# update_state의 기능
# BSR (bandwidth to space ratio) 단위 면적당 bandwidth를 최대화 해야한다

# Masking의 대상은 오직 Storage이다 -> bandwidth 아니다

class OCPEnv_1(gym.Env):
    def __init__(self, *args, **kwargs):
        self.n_nodes = 50 # proxy의 개수
        self.seed = 0 # 초기 seed 값 => 근데 생각해보니깐 demand에서 seed를 정해야 재현성이 정확히 확보 될 듯
        assign_env_config(self, kwargs) # 파라미터로 온 seed 값 지정하기
        self.cache_capacity = 1 # proxy의 bandwidth, storage capacity = 1
        self.t_interval = 20
        self.step_limit = int(60 * 24 / self.t_interval) # Total steps per day based on time intervals => 총 72번의 step_limit
        # self.target_proxy = {} # 실제로 할당되는 proxy
        # self.tol = 1e-5 # 오차값
        self.tol =0.0
        self.mask = True # mask의 유무 => MaskablePPO 사용을 위해, 이건 사실 Wrapper인 ActionMasker로 나중에 감싸기 때문에 환경코드에서는 필요 없음

        self.action_space = gym.spaces.MultiBinary(self.n_nodes) # video object의 경우 copy가 가능하기 때문에 

        self.observation_space = spaces.Dict({
            # "action_masks": spaces.Box(0, 1, shape=(self.n_nodes,), dtype=bool), => action_space는 wrapper로 해야 한다!!
            "proxy_state": spaces.Box(0, 1, shape=(self.n_nodes, 3), dtype=np.float32), # 현재 proxy 할당 된 상태 (초기값 = 0)
            "current_video_state": spaces.Box(0,1,shape=(3,), dtype=np.float32), # 현재 step에서 할당 당하는 video의 싱테 (order, bandwidth, storage)
            "proxy_list": spaces.Box(0,1,shape=(self.n_nodes,),dtype=np.float32) 
            # assigned_proxy[0] => action_space의 return 값
            # assigned_proxy[1] => action_space의 결과를 보고 실제로 할당된 proxy 값 (최대 shape는 n_nodes)
            # 이렇게 한 이유 => 둘을 비교하면서 환경 state를 update 하는게 학습에 더 좋다고 생각 했음 (명확하게 hint를 주는 것)
            # 값을 단순 index로 하는 것이 아니라 normalize해서 학습의 효율 높인다
        })

        if self.seed == 0: # 만약 seed가 제대로 전달 안될 경우
            np.random.seed(self.seed)
            print(f"random seed = {self.seed}")
        self.reset()
    
    def _RESET(self):
        self.demand = self.generate_demand()  
        self.current_step = 0  

        self.state = { # observation_space와 state의 차이 = observation은 기대값의 형 정의, state는 실제 값 저장 배열
            # "action_masks": np.zeros((self.n_nodes)),
            "proxy_state": np.zeros((self.n_nodes, 3), dtype=np.float32),
            "current_video_state": self.demand[self.current_step],
            "proxy_list": np.zeros((self.n_nodes,), dtype=np.float32)
        }
        self.assignment = {} # 각 step의 할당 과정 저장 배열

        return self.state, {} # return 값 => info도 요구한다
    
    # Version 2: agent의 action_space를 대상으로 최적화 분배 진행 (균등 분배가 최적화라는 가정 => 분산 기준으로 판단)
    # 이후 모든 action_space가 아닌 최적화 분배 대상에 대해서만 update_state 진행
    # current_video_state의 bandwidth가 모두 할당되지 않을 수 있다

    def _STEP_2(self, action):
        # 그럼 reward는 얼마나 최종값이 균등하게 분배되었는지로 판단하기 => 그래서 최대한 다음 action_space에 적절한 값을 고를 것이다
        print("step start!")
        done = False
        truncated = False
        reward = 0
        temp_target_proxy = np.where(action == 1)[0] # action에서 값이 1인 index의 집합 => agent가 생각하는 copy 대상, proxy_list[0]
        actual_target_proxy = np.zeros((len(temp_target_proxy), 2)) 
        # 오류 확인 코드
        if action.ndim != 1 or len(action) != self.n_nodes or not set(action).issubset({0, 1}): # 오류 확인
            # action이 1차원인지 (Multibinary 인지) / action이 50개로 나왔는지 / binary의 값만 가지고 있는지
            raise ValueError("Invalid action: {}".format(action))
        
        if len(temp_target_proxy) == 0: # 아무것도 proxy sever에 copy 안하는 경우
            reward = -1000 # 요구되는 bandwidth와 storage가 있는데도 불구하고 할당을 못하는 상황이니깐 reward 음수값 부여
            # done = True ######################## 여기 다시 확인 => 일단 끝내는게 아니라 reward를 낮게 주면서 이 상황이 안오게 하는게 좋을 듯
            # return 직전에 상태 update 해야한다
            self.update_state_2(actual_target_proxy) 
            return self.state, reward, done, truncated, {}

        # 최적화하면서 할당 시작
        indices = np.arange(len(temp_target_proxy)) # index화 하기
        best_variance = float("inf") # 분산을 통해 각 proxy의 균등 정도 수치화 비교
        best_subset = None # action_space 중에서 최종적으로 
        best_allocation = None
        is_optimized = False
        print(f"temp_target_proxy = {temp_target_proxy}")

        if self.current_step !=0 and len(temp_target_proxy)<self.n_nodes/3:
            # len 길이에 따라 최적화 유무 결정한 이유는 학습 속도가 너무 안나와서
            for subset_size in range(1,len(temp_target_proxy) +1):
                print(f"subset_size = {subset_size}")
                subsets = [list(comb) for comb in itertools.combinations(indices, subset_size)]

                for subset in subsets:
                    # 실제 index 매핑
                    subset_indices = [temp_target_proxy[i] for i in subset]

                    # 초기값 및 제약 조건 설정
                    x0 = np.zeros(len(subset))
                    bounds = [(0, 1.0 - self.state["proxy_state"][i,1]) for i in subset_indices]

                    constraints = [
                        {"type": "eq", "fun": lambda x: np.sum(x) - self.state["current_video_state"][1]}  # bandwidth 분배
                    ]

                    # 목적 함수: 선택된 subset에 대한 균등성 평가
                    def objective(x):
                        b_final = self.state["proxy_state"][:,1].copy()
                        for i, xi in zip(subset_indices, x):
                            b_final[i] += xi
                        mu = np.mean(b_final)
                        variance = np.mean((b_final - mu) ** 2)
                        return variance

                    # 최적화 실행
                    result = minimize(objective, x0, bounds=bounds, constraints=constraints)

                    if result.success:
                        # 현재 subset의 결과가 가장 작은 분산을 가지는지 확인
                        if result.fun < best_variance:
                            is_optimized = True
                            best_variance = result.fun
                            best_subset = subset_indices
                            best_allocation = result.x
        else:
            is_optimized = True
            best_subset = temp_target_proxy
            best_allocation = [self.state["current_video_state"][1]/len(best_subset)]*len(best_subset)
        
        ## reward 관련 (reward 종류 별 중요도에 의한 가중치는 추후에 생각하기)
        # 1. step에 따라서 증가
        reward += self.current_step*10

        # 2. agent의 action_space가 모두 유효한지 그 차이에 대한 reward (실제 allocation도 update)
        if is_optimized:        
            # 일단 최적화가 이루어졌다면
            actual_target_proxy = np.zeros((len(best_subset), 2))
            actual_target_proxy[:,0] = best_subset
            actual_target_proxy[:,1] = best_allocation
            reward += (len(best_subset)-len(temp_target_proxy))

        else:
            # 최적화가 이루어 지지 않았다 => 요구하는 bandwidth를 모두 충족하지 못함
            actual_target_proxy = np.zeros((len(temp_target_proxy), 2))
            actual_target_proxy[:,0] = temp_target_proxy
            # print(f"actual_target_proxy = {actual_target_proxy}")
            for i in range(len(actual_target_proxy)):
                idx = actual_target_proxy[:,0].astype(int)
                # print(f"i = {i}, idx = {idx}")
                # exit(0)
                actual_target_proxy[i,1] = (1-self.state["proxy_state"][idx[i],1]) ## action_space에서 원하는 proxy에 남은 bandwidth
            lost_bandwidth = np.sum(actual_target_proxy[:,1])
            reward-=lost_bandwidth*10 # 할당에 실패한 bandwidth 만큼 가중치 빼기
        print(f"=== after optimize ===\nactual_target_proxy = {actual_target_proxy}")

        # 3. 단일한 step에 할당된 bandwidth에 대한 reward => 독립 보상
        reward+=np.sum(actual_target_proxy[:,1])
        
        # 4. 얼마나 균등하게 되었는가 (현재 모든 proxy 기준) => 전체 보상
        currnet_variance = np.var(self.state["proxy_state"][:,1])
        # print(f"current_variance = {currnet_variance}")
        # exit(0)
        if currnet_variance !=0:
            reward+=(1/currnet_variance) # current_variance는 작을 수록 좋음
        else:
            reward+=100
        
        ## update_state
        self.update_state_2(actual_target_proxy) # 이제 할당하기

        # 성공적으로 72회를 넘어섰을 때 -> 모든 video obj가 성공적으로 끝냄
        # done이 되는 순간은 이것만 하기 => 일단 한 step은 넘어야 한다
        if self.current_step>=self.step_limit:
            print(f"current episode is done!!!")
            done = True

        print(f"current reward = {reward}")    
        return self.state, reward, done, truncated, {} # 여기 마지막 값도 info 필요 {'action_mask': self.state["action_mask"]}


    def update_state_2(self,actual_target_proxy):
        # update_state의 대상: proxy_state, current_video_state, proxy_list, current_step
        # action_mask의 경우 ActionMaker로 Wrapping 하기 때문에 update 필요 없다

        # current_step
        self.current_step += 1
        step = self.current_step if self.current_step < self.step_limit else self.step_limit - 1 # current_step update

        # proxy_state update for bandwidh and storage
        # print(f"actual_targt_proxy[:,0] = {actual_target_proxy[:,0]}")
        # exit(0)

        for i in range(len(actual_target_proxy)):
            idx = int(actual_target_proxy[i,0])
            # print(f"actual_target_proxy = {actual_target_proxy[idx,1]}")
            # print(f"self.state~ = {self.state['proxy_state'][idx,1]}")
            # exit(0)
            self.state["proxy_state"][idx,1] = actual_target_proxy[i,1]
            self.state["proxy_state"][idx,2] = self.state["current_video_state"][2]

        # current_video_state
        self.state["current_video_state"] = self.demand[step]

        # proxy_list
        self.state["proxy_list"] = np.zeros(self.n_nodes, dtype=np.float32)  # 초기화
        for i in actual_target_proxy[:, 0].astype(int):
            self.state["proxy_list"][i] = 1.0  # 복사 대상인 proxy는 1로 설정

        # state cliping
        self.state["proxy_state"][:,1:] = np.clip(self.state["proxy_state"][:,1:],0,1) # proxy_state cliping 하기


    def valid_action_mask(self):
        return self.valid_action_mask_2()


    def valid_action_mask_storage(self):
        # storage만 masking의 대상, bandwidth는 대상이 아니다
        action_mask = np.zeros((self.n_nodes, 2), dtype=bool)

        for i in range(self.n_nodes):
            proxy_storage = self.state["proxy_state"][i, 2]
            current_video_storage = self.state["current_video_state"][2]

            # 노드가 현재 비디오를 수용할 수 있는지 확인
            can_assign = (
                proxy_storage + current_video_storage <= self.cache_capacity + self.tol
            )

            # 액션 0과 1에 대한 마스킹 설정
            if can_assign:
                action_mask[i] = [True, True] # 각 index가 복사 유무를 말해준다
            else:
                action_mask[i] = [True, False] # 복사 불가능

        return action_mask.flatten() # 자동으로 flatten 되는 듯 없어도 코드 이상 없음
        # return action_mask # 여기 flatten 유무 확인

    def generate_demand_normal(self):
        np.random.seed(self.seed)
        n = self.step_limit  # Total steps representing assessments in a day

        # Storage demand from normal distribution for a single object
        storage_demand = np.random.normal(loc=0.5, scale=0.1, size=n)
        storage_demand = np.clip(storage_demand, 0, 1)  # Ensure demand is within 0 to 1

        # Bandwidth demand from normal distribution for a single object
        bandwidth_demand = np.random.normal(loc=0.5, scale=0.1, size=n)
        bandwidth_demand = np.clip(bandwidth_demand, 0, 1)  # Ensure demand is within 0 to 1

        # Combine time ratio, bandwidth, and storage demand for the single object
        demand = np.column_stack([np.arange(n) / n, bandwidth_demand, storage_demand])

        return demand
    

    def sample_action(self):
        return self.action_space.sample()

    def reset(self, **kwargs):
        return self._RESET()
    
    def generate_demand(self):
        return self.generate_demand_normal()
    
    def step(self, action):
        # print(f"================ step start ================")
        # print(f"Step method result: {self._STEP(action)}")
        return self._STEP_2(action)
    

##########################################################################################
# Not Used
##########################################################################################

    def update_state(self):
            # current_video_state update, current_step도 ++ 하기, 
            # proxy node의 경우 _STEP에서 update 하므로 굳이 여기서 수정하지 않아도 된다
            # action_mask 또한 이후에 ActionMasker로 Wrapping 하기 때문에 update 필요 없다
            self.current_step = self.current_step+1
            # print(f"current_step = {self.current_step}")
            step = self.current_step if self.current_step < self.step_limit else self.step_limit - 1 # current_step update

            self.state["proxy_state"][:,1:] = np.clip(self.state["proxy_state"][:,1:],0,1) # proxy_state cliping 하기

            self.state["current_video_state"] = self.demand[step]

# Version 0_20250109 => 순서 파악을 위해서 기초적인 코드
    def _STEP_0(self, action):
        
        done = False
        truncated = False
        reward = 0
        # current_video_step = self.state["current_video_state"][0]*72
        current_video_bandwidth = self.state["current_video_state"][1]
        current_video_storage = self.state["current_video_state"][2]
        
        # print(f"============== current step = {current_video_step} ==============")
        # print(f"current_video_bandwidth = {current_video_bandwidth}")
        # print(f"current_video_storage = {current_video_storage}")
        self.target_proxy = np.where(action == 1)[0] # action에서 값이 1인 index의 집합 => video obj가 복사되어야 하는 대상

        if action.ndim != 1 or len(action) != self.n_nodes or not set(action).issubset({0, 1}): # 오류 확인
            # action이 1차원인지 (Multibinary 인지) / action이 50개로 나왔는지 / binary의 값만 가지고 있는지
            raise ValueError("Invalid action: {}".format(action))
        # print(f"==== in step ====")
        # print(f"target_proxy = {self.target_proxy}") # target_proxy 확인

        if len(self.target_proxy) == 0: # 아무것도 proxy sever에 copy 안하는 경우
            # reward = -1000 # 요구되는 bandwidth와 storage가 있는데도 불구하고 할당을 못하는 상황이니깐 reward 음수값 부여
            done = True ######################## 여기 다시 확인
            # return 직전에 상태 update 해야한다
            self.update_state() 
            return self.state, reward, done, truncated, {}

        is_invalidable = False
        # 여기서 기본적으로 성공 보상을 10 정도로 setting 하거나 아니면
        # 그냥 0으로 세팅해서 음수의 보상을 하던가
        
        # target_proxy에 대해서 bandwidth와 storage 할당하기
        for single_proxy in self.target_proxy:
            proxy_activate = self.state["proxy_state"][single_proxy, 0]
            proxy_bandwidth = self.state["proxy_state"][single_proxy, 1]
            proxy_storage = self.state["proxy_state"][single_proxy, 2]

            # 불가능하다면 is_invalidable을 True로 바꾸기
            if (1 + self.tol < current_video_bandwidth + proxy_bandwidth) or (1 + self.tol <current_video_storage + proxy_storage) :
                is_invalidable = True
                print(f"current single_proxy {single_proxy} is not invalidable")
                break

            # 활성화 하기
            if proxy_activate == 0:
                proxy_activate = 1.0
            
            # 이제 요구대로 할당하기
            proxy_bandwidth+=current_video_bandwidth
            proxy_storage+=current_video_storage

            # 모두 끝나고 할당 진행하기
            self.state["proxy_state"][single_proxy,0] = proxy_activate
            self.state["proxy_state"][single_proxy,1] = proxy_bandwidth
            self.state["proxy_state"][single_proxy,2] = proxy_storage

        # 이거 끝나고 더하기
        if is_invalidable:
            # 불가능한 action을 agent가 선택한 경우 # mask가 제대로 작동하지 않는 경우 => 이건 나오면 안된당
            print(f"** current action of agent is not validable now! **")
            reward = -1000
            done = True
        else:
            # 가능한 action만 선택한 경우 => 성공적으로 이번 step이 종료됨
            # 현재 상황을 기반으로 전체적인 reward를 계산하는 것이 마땅함
            ### 여기서 reward 전체적으로 계산하기!!

            reward += 10 * len(self.target_proxy) # 일단 대강 적기 
            # 어떤게 좋은지 모르겠어서..
            # done은 아직 그래도 false -> 한 에피소드가 끝낼 경우만 done = True

        # 성공적으로 72회를 넘어섰을 때 -> 모든 video obj가 성공적으로 끝냄
        if self.current_step >= self.step_limit:
            print(f"current episode is done!!!")
            done = True
    

        # return 직전에 상태 update하기
        self.update_state() 

        return self.state, reward, done, truncated, {} # 여기 마지막 값도 info 필요 {'action_mask': self.state["action_mask"]}

    def valid_action_mask_2(self):
        # 2차원 배열로 초기화
        action_mask = np.zeros((self.n_nodes, 2), dtype=bool)
        # print(f"===  in action_mask ====")
        # print(f"self.target_proxy = {self.target_proxy}")

        for i in range(self.n_nodes):
            proxy_bandwidth = self.state["proxy_state"][i, 1]
            proxy_storage = self.state["proxy_state"][i, 2]
            current_video_bandwidth = self.state["current_video_state"][1]
            current_video_storage = self.state["current_video_state"][2]

            # 노드가 현재 비디오를 수용할 수 있는지 확인
            can_assign = (
                proxy_bandwidth + current_video_bandwidth <= self.cache_capacity + self.tol and
                proxy_storage + current_video_storage <= self.cache_capacity + self.tol
            )

            # 액션 0과 1에 대한 마스킹 설정
            if can_assign:
                action_mask[i] = [True, True] # 각 index가 복사 유무를 말해준다
            else:
                action_mask[i] = [True, False] # 복사 불가능

        return action_mask.flatten() # 자동으로 flatten 되는 듯 없어도 코드 이상 없음
        # return action_mask # 여기 flatten 유무 확인
        # 2차원 배열로 초기화

    def valid_action_mask_1(self):
        # 1차원 배열로 초기화 
        # => RuntimeError: shape '[-1, 100]' is invalid for input of size 50
        # 2차원 배열을 기대함
        action_mask = np.zeros((self.n_nodes,), dtype=bool)

        for i in range(self.n_nodes):
            proxy_bandwidth = self.state["proxy_state"][i, 1]
            proxy_storage = self.state["proxy_state"][i, 2]
            current_video_bandwidth = self.state["current_video_state"][1]
            current_video_storage = self.state["current_video_state"][2]

            # 노드가 현재 비디오를 수용할 수 있는지 확인
            can_assign = (
                proxy_bandwidth + current_video_bandwidth <= self.cache_capacity + self.tol and
                proxy_storage + current_video_storage <= self.cache_capacity + self.tol
            )
            if can_assign:
                action_mask[i] = 1
            else:
                action_mask[i] = 0

            # 액션 0과 1에 대한 마스킹 설정
            # action_mask[i] = can_assign
        return action_mask    
    
