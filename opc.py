import numpy as np
import gymnasium as gym
from or_gym.utils import assign_env_config
from gymnasium import spaces
import inspect

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
        self.target_proxy = {}
        # self.tol = 1e-5 # 오차값
        self.tol =0.0
        self.mask = True # mask의 유무 => MaskablePPO 사용을 위해, 이건 사실 Wrapper인 ActionMasker로 나중에 감싸기 때문에 환경코드에서는 필요 없음

        self.action_space = gym.spaces.MultiBinary(self.n_nodes) # video object의 경우 copy가 가능하기 때문에 

        self.observation_space = spaces.Dict({
            # "action_masks": spaces.Box(0, 1, shape=(self.n_nodes,), dtype=bool),
            "proxy_state": spaces.Box(0, 1, shape=(self.n_nodes, 3), dtype=np.float32), # 현재 proxy 할당 된 상태 (초기값 = 0)
            "current_video_state": spaces.Box(0,1,shape=(3,), dtype=np.float32) # 현재 step에서 할당 당하는 video의 싱테 (order, bandwidth, storage)
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
            "current_video_state": self.demand[self.current_step] 
        }
        self.assignment = {} # 각 step의 할당 과정 저장 배열

        return self.state, {} # return 값 => info도 요구한다
    
    
    # def _STEP_1(self, action):
    #     done = False
    #     truncated = False
    #     reward = 0
    #     current_video_bandwidth = self.state["current_video_state"][1]
    #     current_video_storage = self.state["current_video_state"][2]
    #     # print(f"current_video_bandwidth = {current_video_bandwidth}")
    #     # print(f"current_video_storage = {current_video_storage}")
    #     target_proxy = np.where(action == 1)[0] # action에서 값이 1인 index의 집합 => video obj가 복사되어야 하는 대상

    #     if action.ndim != 1 or len(action) != self.n_nodes or not set(action).issubset({0, 1}): # 오류 확인
    #         # action이 1차원인지 (Multibinary 인지) / action이 50개로 나왔는지 / binary의 값만 가지고 있는지
    #         raise ValueError("Invalid action: {}".format(action))
        
    #     # print(f"target_proxy = {target_proxy}") # target_proxy 확인

    #     if len(target_proxy) == 0: # 아무것도 proxy sever에 copy 안하는 경우
    #         # reward = -1000 # 요구되는 bandwidth와 storage가 있는데도 불구하고 할당을 못하는 상황이니깐 reward 음수값 부여
    #         done = True ######################## 여기 다시 확인
    #         # return 직전에 상태 update 해야한다
    #         self.update_state() 
    #         return self.state, reward, done, truncated, {}

    #     is_invalidable = False
    #     # 여기서 기본적으로 성공 보상을 10 정도로 setting 하거나 아니면
    #     # 그냥 0으로 세팅해서 음수의 보상을 하던가
        
    #     # target_proxy에 대해서 bandwidth와 storage 할당하기
    #     for single_proxy in target_proxy:
    #         proxy_activate = self.state["proxy_state"][single_proxy, 0]
    #         proxy_bandwidth = self.state["proxy_state"][single_proxy, 1]
    #         proxy_storage = self.state["proxy_state"][single_proxy, 2]

    #         # 불가능하다면 is_invalidable을 True로 바꾸기
    #         # 여기서 불가능한건 storage가 넘는 순간 => masking의 대상이니깐
    #         if (1 + self.tol <current_video_storage + proxy_storage) :
    #             is_invalidable = True
    #             print(f"current single_proxy {single_proxy} is not invalidable")
    #             break

    #         # 이건 bandwitdh의 대상
    #         if (1+self.tol<current_video_bandwidth+proxy_bandwidth):



    #         # 활성화 하기
    #         if proxy_activate == 0:
    #             proxy_activate = 1.0
            
    #         # 이제 요구대로 할당하기
    #         proxy_bandwidth+=current_video_bandwidth
    #         proxy_storage+=current_video_storage

    #         # 모두 끝나고 할당 진행하기
    #         self.state["proxy_state"][single_proxy,0] = proxy_activate
    #         self.state["proxy_state"][single_proxy,1] = proxy_bandwidth
    #         self.state["proxy_state"][single_proxy,2] = proxy_storage

    #     # 이거 끝나고 더하기
    #     if is_invalidable:
    #         # 불가능한 action을 agent가 선택한 경우 # mask가 제대로 작동하지 않는 경우 => 이건 나오면 안된당
    #         print(f"** current action of agent is not validable now! **")
    #         reward = -1000
    #         done = True
    #     else:
    #         for(self.state["proxy_state"][:,1]==)
    #         # 가능한 action만 선택한 경우 => 성공적으로 이번 step이 종료됨
    #         # 현재 상황을 기반으로 전체적인 reward를 계산하는 것이 마땅함
    #         ### 여기서 reward 전체적으로 계산하기!!

    #         reward += 10 * len(target_proxy) # 일단 대강 적기 
    #         # 어떤게 좋은지 모르겠어서..
    #         # done은 아직 그래도 false -> 한 에피소드가 끝낼 경우만 done = True

    #     # 성공적으로 72회를 넘어섰을 때 -> 모든 video obj가 성공적으로 끝냄
    #     if self.current_step >= self.step_limit:
    #         print(f"current episode is done!!!")
    #         done = True
    

    #     # return 직전에 상태 update하기
    #     self.update_state() 

    #     return self.state, reward, done, truncated, {} # 여기 마지막 값도 info 필요 {'action_mask': self.state["action_mask"]}

    # Version 2: agent의 action_space의 기존 bandwidth를 얻어 온 다음, 
    # step에서 action_space가 가능한 답만 말하지 않았다면, reward를 음수로 주고, 진짜 할당되는 것만 
    # 그리고 update_state에서 action_space의 결과가 아닌 "실제로 bandwidth가 할당되는 것만" update하기!!!!

    def _STEP_2(self, action):
        # 최적화의 목적이 최종 값들이 서로 균등하게 분배되는 것 이라 판단
        # 그럼 reward는 얼마나 최종값이 균등하게 분배되었는지로 판단하기 => 그래서 최대한 다음 action_space에 적절한 값을 고를 것이다

            done = False
            truncated = False
            reward = 0
            temp_target_proxy = np.where(action == 1)[0] # action에서 값이 1인 index의 집합 => video obj가 복사되어야 하는 대상

            # 오류 확인 코드
            if action.ndim != 1 or len(action) != self.n_nodes or not set(action).issubset({0, 1}): # 오류 확인
                # action이 1차원인지 (Multibinary 인지) / action이 50개로 나왔는지 / binary의 값만 가지고 있는지
                raise ValueError("Invalid action: {}".format(action))
            
            # print(f"target_proxy = {target_proxy}") # target_proxy 확인

            if len(temp_target_proxy) == 0: # 아무것도 proxy sever에 copy 안하는 경우
                # reward = -1000 # 요구되는 bandwidth와 storage가 있는데도 불구하고 할당을 못하는 상황이니깐 reward 음수값 부여
                done = True ######################## 여기 다시 확인
                # return 직전에 상태 update 해야한다
                self.update_state() 
                return self.state, reward, done, truncated, {}

            is_invalidable = False
            # 여기서 기본적으로 성공 보상을 10 정도로 setting 하거나 아니면
            # 그냥 0으로 세팅해서 음수의 보상을 하던가


            ########
            # 실제로 
            ########
            
            # target_proxy에 대해서 bandwidth와 storage 할당하기 => 이거 이제 update_state로 함수 넘기기
            for single_proxy in temp_target_proxy:
                proxy_activate = self.state["proxy_state"][single_proxy, 0]
                proxy_bandwidth = self.state["proxy_state"][single_proxy, 1]
                proxy_storage = self.state["proxy_state"][single_proxy, 2]

                # 불가능하다면 is_invalidable을 True로 바꾸기
                # 여기서 불가능한건 storage가 넘는 순간 => masking의 대상이니깐
                if (1 + self.tol <self.state["current_video_state"][2] + proxy_storage) :
                    is_invalidable = True
                    print(f"current single_proxy {single_proxy} is not invalidable")
                    break

                # 이건 bandwitdh의 대상
                if (1+self.tol< self.state["current_video_state"][1]+proxy_bandwidth):



                # 활성화 하기
                if proxy_activate == 0:
                    proxy_activate = 1.0
                
                # 이제 요구대로 할당하기
                proxy_bandwidth+=self.state["current_video_state"][1]
                proxy_storage+=self.state["current_video_state"][2]

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
                for(self.state["proxy_state"][:,1]==)
                # 가능한 action만 선택한 경우 => 성공적으로 이번 step이 종료됨
                # 현재 상황을 기반으로 전체적인 reward를 계산하는 것이 마땅함
                ### 여기서 reward 전체적으로 계산하기!!

                reward += 10 * len(self.target_proxy) # 일단 대강 적기 
                # 어떤게 좋은지 모르겠어서..
                # done은 아직 그래도 false -> 한 에피소드가 끝낼 경우만 done = True

            # 성공적으로 72회를 넘어섰을 때 -> 모든 video obj가 성공적으로 끝냄
            if self.current_step>=self.step_limit:
                print(f"current episode is done!!!")
                done = True
        

            # return 직전에 상태 update하기
            self.update_state_2() 

            return self.state, reward, done, truncated, {} # 여기 마지막 값도 info 필요 {'action_mask': self.state["action_mask"]}


    def update_state_2(self):
        # current_video_state update, current_step도 ++ 하기, 
        # proxy node의 경우 _STEP에서 update 하므로 굳이 여기서 수정하지 않아도 된다
        # action_mask 또한 이후에 ActionMasker로 Wrapping 하기 때문에 update 필요 없다
        self.current_step = self.current_step + 1
        # print(f"current_step = {self.current_step}")
        step = self.current_step if self.current_step < self.step_limit else self.step_limit - 1 # current_step update

        self.state["proxy_state"][:,1:] = np.clip(self.state["proxy_state"][:,1:],0,1) # proxy_state cliping 하기

        self.state["current_video_state"] = self.demand[step]

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
        return self._STEP_0(action)
    

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
    
