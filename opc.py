import numpy as np
import gymnasium as gym
from or_gym.utils import assign_env_config
from gymnasium import spaces
import inspect


class OCPEnv_1(gym.Env):
    def __init__(self, *args, **kwargs):
        # proxy의 bandwidth, storage capacity = 1

        self.n_nodes = 50 # proxy의 개수

        assign_env_config(self, kwargs)
        self.cache_capacity = 1  # Cache capacity per proxy
        self.t_interval = 20
        self.step_limit = int(60 * 24 / self.t_interval)  # Total steps per day based on time intervals
        # 총 72번 step_limit

        # Environment
        self.tol = 1e-5  
        self.seed = 1234  
        self.mask = True  

        self.action_space = gym.spaces.MultiBinary(self.n_nodes)
        # for debuging
        # print(f"Action Space Details: {self.action_space}")
        # print(f"Action Space Type: {type(self.action_space)}")
        # print(f"Action Space Size: {self.action_space.n}")
        # print(f"Action Space Sample: {self.action_space.sample()}")
        # exit(0)


        self.proxy_validity_mask = np.ones(self.n_nodes)
        if self.mask:
            self.observation_space = spaces.Dict({
                "action_mask": spaces.Box(0, 1, shape=(self.n_nodes,)),  
                "proxy_state": spaces.Box(0, 1, shape=(self.n_nodes, 3)), # 현재 할당된 상태
                "current_video_state": spaces.Box(0,1,shape=(3, 1))
            })
        else:
            self.observation_space = spaces.Dict({
                "proxy_state": spaces.Box(0, 1, shape=(self.n_nodes, 3)),
                "current_video_state": spaces.Box(0,1,shape=(3, 1))
            })

        self.reset()

    def _RESET(self):
        self.demand = self.generate_demand()  
        self.current_step = 0  

        self.state = {
            "action_mask": np.ones(self.n_nodes), 
            "proxy_state": np.zeros((self.n_nodes, 3)),  
            "current_video_state": np.vstack(self.demand[self.current_step])
        }
        self.assignment = {}  # Record actions taken at each step

        return self.state, {'action_mask': self.state["action_mask"]}


    def step(self, action):
        print(f"================ step start ================")
        return self._STEP(action)
    
    def _STEP(self, action):
        # 여기서 action의 결과는 action_space가 multibinary이니깐 0과 1로 이루어진 50개의 숫자
        # 여기서 50개의 숫자 중 1인 곳에 현재 step에 해당하는 demand인 bandwidth과 storage를 가진 video를 할당한다
        print(f"Now agent's action = {action}")
        done = False
        truncated = False
        reward = 0

        current_video_bandwidth = self.state["current_video_state"][1]
        current_video_storage = self.state["current_video_state"][2]
        
        # 일단 몇번째 proxy에 복사해야하는지 알아내기

        target_proxy = np.where(action == 1)  # action에서 값이 1인 index의 집합
        print(f"target_proxy = {target_proxy}")

        # action_space의 결과 1이면 해당 proxy에 copy하고 0이면 그냥 두기
        if action.ndim != 1 or len(action) != self.n_nodes or not set(action).issubset({0, 1}):
            # action이 1차원인지 / 50개인지 / 0과 1만 원소로 가지고 있는지를 확인
            raise ValueError("Invalid action: {}".format(action))
        
        # 모두 0으로 action하라고 하면 안된다 -> 이것도 reward 낮게 반환하기
        if np.size(target_proxy) == 0:
            reward = -1000
            done = True
        
        # 여기서 하나라도 할당이 안되는게 있으면 바로 reward 낮추고 바로 done하기
        # 이거 다음 for 구문에 넣으면 안되는 이유: 밖에 있어야 현재 agent가 내린 action_space 전체에 대한 판단이 가능하다
        # for 구문 안에 넣으면 reward 늘리다가 이후에 낮출 수 있어서 
        # => 그냥 temp_reward로 하고 true이면 더하지 말기

        is_invalidable = False
        # 여기서 기본적으로 성공 보상을 10 정도로 setting 하거나 아니면
        # 그냥 0으로 세팅해서 음수의 보상을 하던가
        
        # target_proxy에 대해서 bandwidth와 storage 할당하기
        for single_proxy in target_proxy:
            proxy_activate = self.state["proxy_state"][single_proxy, 0]
            proxy_bandwidth = self.state["proxy_state"][single_proxy, 1]
            proxy_storage = self.state["proxy_state"][single_proxy, 2]

            ## for debug
            print(f"current single_proxy = {single_proxy}")
            print(f"proxy_activate = {proxy_activate}")
            print(f"proxy_bandwidth = {proxy_bandwidth}")
            print(f"proxy_storage = {proxy_storage}")

            # 불가능하다면 is_invalidable을 True로 바꾸기
            if 1 + self.tol < current_video_bandwidth + proxy_bandwidth or 1 + self.tol <current_video_storage + proxy_storage:
                is_invalidable = True
                print(f"current single_proxy {single_proxy} is not invalidable")
                break

            # 활성화 하기
            if proxy_activate == 0:
                proxy_activate = 1
            
            # 이제 요구대로 할당하기
            proxy_bandwidth+=current_video_bandwidth
            proxy_storage+=current_video_storage

            # 모두 끝나고 할당 진행하기
            self.state["proxy_state"][single_proxy,0] = proxy_activate
            self.state["proxy_state"][single_proxy,1] = proxy_bandwidth
            self.state["proxy_state"][single_proxy,2] = proxy_storage

        # 이거 끝나고 더하기
        if is_invalidable:
            # 불가능한 action을 agent가 선택한 경우
            reward = -1000
            done = True
        else:
            # 가능한 action만 선택한 경우 => 성공적으로 이번 step이 종료됨
            # 현재 상황을 기반으로 전체적인 reward를 계산하는 것이 마땅함
            ### 여기서 reward 전체적으로 계산하기!!
            ### 

            reward
            # done은 아직 그래도 false -> 한 에피소드가 끝낼 경우만 done = True


        # 성공적으로 72회를 넘어섰을 때 -> 모든 video obj가 성공적으로 끝냄
        if self.current_step >= self.step_limit:
            print(f"current episode is done!!!")
            done = True
    

        # return 직전에 상태 update하기
        #### 여기에 ####

        return self.state, reward, done, truncated, {'action_mask': self.state["action_mask"]}
        # obs, reward, done, truncated, info = env.step(action)
        # step 함수에 기대되는 반환값

    def update_state(self, node_state):
        # Update proxy node state and reset action masks for each object
        step = self.current_step if self.current_step < self.step_limit else self.step_limit - 1

        # Update proxy and video states in the main state
        self.state["proxy_state"] = np.where(node_state > 1, 1, node_state)  # Clip node state to maximum capacity
        # self.state["video_state"] = self.demand[step]  # Get demand for each object for the current step
        action_mask = np.ones((self.n_objects, self.n_nodes), dtype=int)

        # Create valid action masks for each object at the current step
        for obj_idx, demand in enumerate(self.demand[step]):
            valid_nodes = (node_state[:, 1:] + demand[1:] <= 1 + self.tol).all(axis=1)  # True for nodes that can satisfy demand
            action_mask[obj_idx] = valid_nodes.astype(int)  # Mark valid nodes as 1 in the action mask
        # Update action_mask in state
        self.state["action_mask"] = action_mask

    def valid_actions(self):
        # Return the valid action mask for each object at the current state
        # 여기서 action_mask와 proxy_validity_mask 모두 확인해야 한다
        # proxy_validity_mask는 영구적인 mask이기 때문에 초기에만 변화하고 이후 변하진 않는다
        print(f"valid_actions is called")

        combined_mask = self.proxy_validity_mask * self.state["action_mask"]

        return combined_mask
 

    def sample_action(self):
        return self.action_space.sample()


    def reset(self, **kwargs):
        return self._RESET()
    
    def generate_demand(self):
        return self.generate_demand_normal()

    def generate_demand_normal(self):
        n = self.step_limit  # Total steps representing assessments in a day

        # Storage demand from normal distribution for a single object
        storage_demand = np.random.normal(loc=0.5, scale=0.1, size=n)
        storage_demand = np.clip(storage_demand, 0, 1)  # Ensure demand is within 0 to 1

        # Bandwidth demand from normal distribution for a single object
        bandwidth_demand = np.random.normal(loc=0.5, scale=0.1, size=n)
        bandwidth_demand = np.clip(bandwidth_demand, 0, 1)  # Ensure demand is within 0 to 1

        # Combine time ratio, bandwidth, and storage demand for the single object
        demand = np.column_stack([np.arange(n) / n, bandwidth_demand, storage_demand])
        # Shape will be (n, 3) representing (time ratio, bandwidth demand, storage demand)
        # print(f"demand = {demand}")
        # exit(0)
        # 여기서 demand는 3개의 index로 이루어진 배열의 집합
        # 
        return demand
    












    ########
    # 나중에 고치기
    ########

    # def generate_demand_normal(self):

    #     n = self.step_limit  # Total steps representing assessments in a day
        
    #     # Storage demand from normal distribution
    #     storage_demand = np.random.normal(loc=0.5, scale=0.1, size=(self.n_objects, n))
    #     storage_demand = np.clip(storage_demand, 0, 1)  # Ensure demand is within 0 to 1

    #     # Bandwidth demand from normal distribution
    #     bandwidth_demand = np.random.normal(loc=0.5, scale=0.1, size=(self.n_objects, n))
    #     bandwidth_demand = np.clip(bandwidth_demand, 0, 1)  # Ensure demand is within 0 to 1
        
    #     # Combine time ratio, bandwidth, and storage demand for each object, resulting in (self.n_objects, n, 3)
    #     demand = np.array([
    #         np.column_stack([np.arange(n) / n, bandwidth_demand[i], storage_demand[i]])
    #         # time ratio, bandwidth, and storage demand
    #         for i in range(self.n_objects)
    #     ])

    #     # Reshape demand to ensure correct shape for each step (self.n_objects, 3) during each step
    #     demand = demand.transpose((1, 0, 2))  # Shape will be (n, self.n_objects, 3)
        
    #     return demand
    
    # def generate_demand_probs(self):
    #     # Generate demand of video objects' bandwidth and storage 
    #     # Not Proxy nodes!! => bandwitdth and storage = 1
    #     # 
    #     # Number of steps per day (e.g., 72 assessments throughout the day)
    #     n = self.step_limit  # Total steps representing assessments in a day
        
    #     # Define probabilities and bins for storage demand
    #     storage_probs = np.array([0.1, 0.15, 0.3, 0.25, 0.15, 0.05])
    #     storage_bins = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 1.0])

    #     # Define probabilities and bins for bandwidth demand
    #     bandwidth_probs = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    #     bandwidth_bins = np.array([0.1, 0.2, 0.4, 0.6, 0.8])

    #     # Generate storage demand using predefined bins and probabilities
    #     storage_demand = np.random.choice(storage_bins, p=storage_probs, size=(self.n_objects, n))

    #     # Generate bandwidth demand using predefined bins and probabilities
    #     bandwidth_demand = np.random.choice(bandwidth_bins, p=bandwidth_probs, size=(self.n_objects, n))
    #     # Combine time ratio, bandwidth, and storage demand for each object, resulting in (self.n_objects, n, 3)
    #     demand = np.array([
    #         np.column_stack([np.arange(n) / n, bandwidth_demand[i], storage_demand[i]])
    #         for i in range(self.n_objects)
    #     ])

    #     # Reshape demand to ensure correct shape for each step (self.n_objects, 3) during each step
    #     demand = demand.transpose((1, 0, 2))  # Shape will be (n, self.n_objects, 3)
        
    #     return demand