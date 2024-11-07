import numpy as np
import gymnasium as gym
from or_gym.utils import assign_env_config
from gymnasium import spaces

class OCPEnv_1(gym.Env):
    def __init__(self, *args, **kwargs):
        # Initialize environment settings based on input configuration
        assign_env_config(self, kwargs)

        # Single proxy 
        self.cache_capacity = 1  # Cache capacity per proxy
        self.bandwidth_capacity = 1  # Bandwidth capacity per proxy
        self.n_nodes = 50  # Number of proxy nodes

        # Video object
        self.n_objects = 100  # Number of video objects based on the reference paper

        # Environment
        self.t_interval = 20  # Time interval
        self.tol = 1e-5  # Tolerance for numerical stability
        self.step_limit = int(60 * 24 / self.t_interval)  # Total steps per day based on time intervals
        self.seed = 1234  # Initialize random seed
        self.mask = True  # Whether to use a mask to filter valid actions

        # Action space now allows each object to be partially assigned to multiple nodes (binary choices)
        self.action_space = gym.spaces.MultiDiscrete([50] * 100)
        # self.action_space = gym.spaces.MultiDiscrete([50] * 100)
        
        # suppose that nodes and objectes are matched one by one
        # self.action_space = gym.spaces.MultiBinary(100 * 50) 

        # action_space => each proxies has 

        # action_space's size = 5000
        # set single dimension
        # self.action_dims = [self.n_objects * self.n_nodes]  # or simply 5000 if you want to hardcode

        # Initialize valid actions for each object
        # set all actions are valiable
        self.proxy_validity_mask = np.ones((self.n_objects, self.n_nodes), dtype=int)
        
        # Observation space includes action masks, available actions, and state (proxy status + demand)
        if self.mask:
            self.observation_space = spaces.Dict({
                "action_mask": spaces.Box(0, 1, shape=(self.n_objects, self.n_nodes)),  # Mask for each object's valid actions
                "proxy_state": spaces.Box(0, 1, shape=(self.n_nodes, 3)),  # State of each proxy node
                "video_state": spaces.Box(0, 1, shape=(self.n_objects, 3))  # State of each video object
            })
        else:
            self.observation_space = spaces.Dict({
                "proxy_state": spaces.Box(0, 1, shape=(self.n_nodes, 3)),
                "video_state": spaces.Box(0, 1, shape=(self.n_objects, 3))
            })
        # for debugging
        # print(f"In __init__: self.observation_space['action_mask'] shape = {self.observation_space['action_mask'].shape}") 
        # print(f"In __init__: self.observation_space['state'] shape = {self.observation_space['state'].shape}")
        self.reset()

    def _RESET(self):
        # Generate demand and initialize state when resetting the environment
        self.demand = self.generate_demand()  # Generate bandwidth and storage demands for each object
        self.current_step = 0  # Initialize current step

        self.state = {
            "action_mask": np.ones((self.n_objects, self.n_nodes)),  # Initialize valid actions for each object
            "proxy_state": np.zeros((self.n_nodes, 3)),  # Initialize state for each proxy node
            "video_state": self.demand[self.current_step]  # Demand for the current step for each video object
        }
        self.assignment = {}  # Record actions taken at each step
       
        # self.proxy_validity_mask = np.ones((self.n_objects, self.n_nodes), dtype=int)  # Initialize valid actions for all objects
        # different between state's action_mask and proxy_validity_mask
        # action_mask => matching information of each steps
        # proxy_validity_mask => eternal matching constraint

        # for debugging
        # print(f"self.n_nodes = {self.n_nodes}") # self.n_nodes = 50
        # print(f"self.demand[self.current_step].shape = {self.demand[self.current_step].shape}") # self.demand[self.current_step].shape = (72, 3)

        # print(f"In __RESET: self.state['action_mask'] shape = {self.state['action_mask'].shape}") 
        # print(f"In __RESET: self.state['state'] shape = {self.state['state'].shape}") 

        return self.state, {'action_mask': self.state["action_mask"]}
    # def _STEP(self, actions):
    #     done = False
    #     truncated = False
    #     node_state = self.state["proxy_state"]  
    #     object_demand = self.state["video_state"]  

    #     actions = actions.reshape(self.n_objects, self.n_nodes)

    #     reward = 0

    #     for obj_idx, obj_actions in enumerate(actions):
    #         demand = object_demand[obj_idx, 1:] 

    #         for node_idx, assign in enumerate(obj_actions):
    #             if assign == 1: 
    #                 if all(node_state[node_idx, 1:] + demand <= 1 + self.tol):
    #                     if node_state[node_idx, 0] == 0:
    #                         node_state[node_idx, 0] = 1  
    #                     node_state[node_idx, 1:] += demand  

    #                     reward += np.sum(node_state[:, 0] * (node_state[:, 1:].sum(axis=1) - 2))
    #                     self.assignment[self.current_step] = (obj_idx, node_idx)
    #                 else:
    #                     reward -= 1000
    #                     done = True
    #                     break

    #     self.current_step += 1
    #     if self.current_step >= self.step_limit:
    #         done = True

    #     self.update_state(node_state)

    #     return self.state, reward, done, truncated, {}

    
    def _STEP(self, actions):
        print(f"================ step start ================")
        done = False
        truncated = False
        node_state = self.state["proxy_state"]  # Extract proxy node state
        object_demand = self.state["video_state"]  # Extract demand of video objects for the current step
        # print(f"actions = {actions}")
        # print(f"actions.shape = {actions.shape}") # actions.shape = (100,)
        # n_objects = 100, n_nodes = 50
        
        # for debugging
        # print(f"self.action_mask.shape = {self.state['action_mask'].shape}")
        # print(f"self.video_state.shape = {self.state['proxy_state'].shape}")
        # print(f"self.proxy_state.shape = {self.state['proxy_state'].shape}")
        # print(f"time ratio (step / step_limit), bandwidth, and storage demand")
        # print(f"self.video_state = {self.state['video_state']}")
        
        # Initialize reward
        reward = 0
        # print(f"proxy_state = {node_state}")
        # print(f"video_state = {object_demand}")

        # Iterate over each object and check assignment across multiple nodes
        for obj_idx, node_idx  in enumerate(actions):
            print(f"action start")
            print(f"obj_idx = {obj_idx}, node_idx = {node_idx}")
            # print(f"validation = {self.state['action_mask']}")
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
        # print(f"self.proxy_validity_mask = {self.proxy_validity_mask}")
        # print(f"self.state[\"action_mask\"] = {self.state['action_mask']}") 

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
        # Sample a random action for each object-node pair
        return self.action_space.sample()

    # def generate_demand(self):
    #     return self.generate_demand_random()
    
    def step(self, actions):
        # Main step function called externally
        # print(f"step start")
        return self._STEP(actions)

    def reset(self, **kwargs):
        # Reset the environment to the initial state
        return self._RESET()
    
    def generate_demand(self):
        return self.generate_demand_normal()

    def generate_demand_normal(self):
        # Generate demand of video objects' bandwidth and storage 
        # Not Proxy nodes!! => bandwitdth and storage = 1
        # 
        # Number of steps per day (e.g., 72 assessments throughout the day)
        n = self.step_limit  # Total steps representing assessments in a day
        
        # Storage demand from normal distribution
        storage_demand = np.random.normal(loc=0.5, scale=0.1, size=(self.n_objects, n))
        storage_demand = np.clip(storage_demand, 0, 1)  # Ensure demand is within 0 to 1

        # Bandwidth demand from normal distribution
        bandwidth_demand = np.random.normal(loc=0.5, scale=0.1, size=(self.n_objects, n))
        bandwidth_demand = np.clip(bandwidth_demand, 0, 1)  # Ensure demand is within 0 to 1
        
        # Combine time ratio, bandwidth, and storage demand for each object, resulting in (self.n_objects, n, 3)
        demand = np.array([
            np.column_stack([np.arange(n) / n, bandwidth_demand[i], storage_demand[i]])
            # time ratio, bandwidth, and storage demand
            for i in range(self.n_objects)
        ])

        # Reshape demand to ensure correct shape for each step (self.n_objects, 3) during each step
        demand = demand.transpose((1, 0, 2))  # Shape will be (n, self.n_objects, 3)
        
        return demand
    
    def generate_demand_probs(self):
        # Generate demand of video objects' bandwidth and storage 
        # Not Proxy nodes!! => bandwitdth and storage = 1
        # 
        # Number of steps per day (e.g., 72 assessments throughout the day)
        n = self.step_limit  # Total steps representing assessments in a day
        
        # Define probabilities and bins for storage demand
        storage_probs = np.array([0.1, 0.15, 0.3, 0.25, 0.15, 0.05])
        storage_bins = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 1.0])

        # Define probabilities and bins for bandwidth demand
        bandwidth_probs = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        bandwidth_bins = np.array([0.1, 0.2, 0.4, 0.6, 0.8])

        # Generate storage demand using predefined bins and probabilities
        storage_demand = np.random.choice(storage_bins, p=storage_probs, size=(self.n_objects, n))

        # Generate bandwidth demand using predefined bins and probabilities
        bandwidth_demand = np.random.choice(bandwidth_bins, p=bandwidth_probs, size=(self.n_objects, n))
        # Combine time ratio, bandwidth, and storage demand for each object, resulting in (self.n_objects, n, 3)
        demand = np.array([
            np.column_stack([np.arange(n) / n, bandwidth_demand[i], storage_demand[i]])
            for i in range(self.n_objects)
        ])

        # Reshape demand to ensure correct shape for each step (self.n_objects, 3) during each step
        demand = demand.transpose((1, 0, 2))  # Shape will be (n, self.n_objects, 3)
        
        return demand