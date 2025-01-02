import numpy as np
import gymnasium as gym
from or_gym.utils import assign_env_config
from gymnasium import spaces
import inspect


class OCPEnv_1(gym.Env):
    def __init__(self, *args, **kwargs):
        self.n_nodes = 50 # proxy의 개수
        # self.n_objects = 100
        # 이제 step의 수가 video object의 수가 된다. step을 100으로 생각하기
        # 이게 아니라 전체 
        assign_env_config(self, kwargs)
        self.cache_capacity = 1  # Cache capacity per proxy
        self.t_interval = 20
        self.step_limit = int(60 * 24 / self.t_interval)  # Total steps per day based on time intervals

        # Environment
        self.tol = 1e-5  # Tolerance for numerical stability
        self.seed = 1234  # Initialize random seed
        self.mask = True  # Whether to use a mask to filter valid actions

        self.action_space = gym.spaces.MultiBinary(self.n_nodes)
        # 객체의 전체 속성과 값을 확인하기
        # print(f"Action Space Details: {self.action_space}")
        # print(f"Action Space Type: {type(self.action_space)}")
        # print(f"Action Space Size: {self.action_space.n}")
        # print(f"Action Space Sample: {self.action_space.sample()}")
        # exit(0)
        self.proxy_validity_mask = np.ones(self.n_nodes)
        # 영구적인 proxy mask
        # Observation space includes action masks, available actions, and state (proxy status + demand)
        if self.mask:
            self.observation_space = spaces.Dict({
                "action_mask": spaces.Box(0, 1, shape=(self.n_nodes,)),  
                "proxy_state": spaces.Box(0, 1, shape=(self.n_nodes, 3)), 
                # "video_state": spaces.Box(0, 1, shape=(self.n_objects, 3)),  
            })
        else:
            self.observation_space = spaces.Dict({
                "proxy_state": spaces.Box(0, 1, shape=(self.n_nodes, 3)),
                # "video_state": spaces.Box(0, 1, shape=(self.n_objects, 3)),
            })

        self.reset()

    def _RESET(self):
        # Generate demand and initialize state when resetting the environment
        self.demand = self.generate_demand()  # Generate bandwidth and storage demands for each object
        self.current_step = 0  # Initialize current step

        self.state = {
            "action_mask": np.ones(self.n_nodes),  # Initialize valid actions for each object
            "proxy_state": np.zeros((self.n_nodes, 3)),  # Initialize state for each proxy node
            # "video_state": self.demand[self.current_step],  # Demand for the current step for each video object
            # "current_step": self.current_step
        }
        self.assignment = {}  # Record actions taken at each step
       

        return self.state, {'action_mask': self.state["action_mask"]}


    def step(self, action):
        print(f"================ step start ================")
        return self._STEP(action)
    
    def single_step():
        return 0
    
    def _STEP(self, action):
        done = False
        truncated = False
        node_state = self.state["proxy_state"]  # Extract proxy node state
        object_demand = self.state["video_state"]  # Extract demand of video objects for the current step
        reward = 0

        # Iterate over each object and check assignment across multiple nodes
        for obj_idx, node_idx  in enumerate(action):
            demand = object_demand[obj_idx, 1:]  # Get bandwidth and storage demands for the object
            # bandwidth & storage
            # Calculate the total demand each node would receive based on the current object's allocation
            if all(node_state[node_idx, 1:] + demand <= 1 + self.tol):
                # Allocate demand to the node
                # bandwidth & storage
                if node_state[node_idx, 0] == 0: 
                    node_state[node_idx, 0] = 1  # Activate the node if inactive
                node_state[node_idx, 1:] += demand  # Add demand to node

                # Calculate reward
                reward += np.sum(node_state[:, 0] * (node_state[:, 1:].sum(axis=1) - 2))

                # Record assignment
                self.assignment[self.current_step] = (obj_idx, node_idx)
                self.update_state(node_state)

            else:
                reward -= 1000  # Penalty if allocation is not possible
                done = True
                print(f"done")

                break

        # Increment step and check if the episode is done
        self.current_step += 1
        if self.current_step >= self.step_limit:
            print(f"=========================== SUCCESS! ===========================")
            done = True  # End the episode if the step limit is exceeded

        # Update the state with the new node_state
        self.update_state(node_state)
        return self.state, reward, done, truncated, {'action_mask': self.state["action_mask"]}

    def update_state(self, node_state):
        # Update proxy node state and reset action masks for each object
        step = self.current_step if self.current_step < self.step_limit else self.step_limit - 1

        # Update proxy and video states in the main state
        self.state["proxy_state"] = np.where(node_state > 1, 1, node_state)  # Clip node state to maximum capacity
        self.state["video_state"] = self.demand[step]  # Get demand for each object for the current step
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
        return demand

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