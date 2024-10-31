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
        self.action_space = spaces.MultiBinary(self.n_objects * self.n_nodes)  # Each entry represents a binary choice for each object-node pair
        
        # Initialize valid actions for each object
        # set all actions are valiable
        self.proxy_validity_mask = np.ones((self.n_objects, self.n_nodes), dtype=int)
        
        # Observation space includes action masks, available actions, and state (proxy status + demand)
        if self.mask:
            self.observation_space = spaces.Dict({
                "action_mask": spaces.Box(0, 1, shape=(self.n_objects, self.n_nodes)),  # Mask for each object's valid actions
                "avail_actions": spaces.Box(0, 1, shape=(self.n_nodes,)),  # Available actions
                "state": spaces.Box(0, 1, shape=(self.n_nodes + self.n_objects, 3))  # Node states + object states
            })
        else:
            self.observation_space = spaces.Box(0, 1, shape=(self.n_nodes + self.n_objects, 3))
        
        # for debugging
        print(f"In __init__: self.observation_space['state'] shape = {self.observation_space['state'].shape}") # self.state['state'] shape = (122, 3)

        self.reset()

    def _RESET(self):
        # Generate demand and initialize state when resetting the environment
        self.demand = self.generate_demand()  # Generate bandwidth and storage demands for each object
        self.current_step = 0  # Initialize current step



        self.state = {
            "action_mask": np.ones((self.n_objects, self.n_nodes)),  # Initialize valid actions for each object
            "avail_actions": np.ones(self.n_nodes),  # Available actions
            "state": np.vstack([
                np.zeros((self.n_nodes + self.n_objects, 3)),  # Initialize state for each proxy node
                self.demand[self.current_step]])  # Add demand for the current step
        }
        self.assignment = {}  # Record actions taken at each step
        self.proxy_validity_mask = np.ones((self.n_objects, self.n_nodes), dtype=int)  # Initialize valid actions for all objects
        
        # for debugging
        # print(f"self.n_nodes = {self.n_nodes}") # self.n_nodes = 50
        # print(f"self.demand[self.current_step].shape = {self.demand[self.current_step].shape}") # self.demand[self.current_step].shape = (72, 3)
        print(f"In __RESET: self.state['state'] shape = {self.state['state'].shape}") # After reset(): self.state['state'] shape = (122, 3)

        return self.state, {}
    
    def _STEP(self, actions):
        # Expecting a MultiBinary array for actions, each entry representing an object-node pair
        done = False
        truncated = False
        node_state = self.state["state"][:self.n_nodes]  # Extract node state
        object_demand = self.state["state"][self.n_nodes:]  # Extract demand of objects
        
        # Reshape actions to (n_objects, n_nodes) to map each object-node pair
        actions = actions.reshape(self.n_objects, self.n_nodes)

        # Iterate over each object and check assignment across multiple nodes
        for obj_idx, obj_actions in enumerate(actions):
            demand = object_demand[obj_idx, 1:]  # Get bandwidth and storage demands for the object

            # Calculate the total demand each node would receive based on the current object's allocation
            for node_idx, assign in enumerate(obj_actions):
                if assign == 1:  # Only allocate if action is 1 for that object-node pair
                    if all(node_state[node_idx, 1:] + demand <= 1 + self.tol):  # Check if demand fits within node's capacity
                        # Allocate demand to the node
                        if node_state[node_idx, 0] == 0:
                            node_state[node_idx, 0] = 1  # Activate the node if inactive
                        node_state[node_idx, 1:] += demand  # Add demand to node
                        reward = np.sum(node_state[:, 0] * (node_state[:, 1:].sum(axis=1) - 2))  # Calculate reward
                        self.assignment[self.current_step] = (obj_idx, node_idx)  # Record assignment
                    else:
                        reward = -1000  # Penalty if allocation is not possible
                        done = True
                        break

        self.current_step += 1
        if self.current_step >= self.step_limit:
            done = True  # End the episode if the step limit is exceeded
        self.update_state(node_state)
        return self.state, reward, done, truncated, {}
    
    def update_state(self, node_state):
        # Update node state and reset action masks for each object
        step = self.current_step if self.current_step < self.step_limit else self.step_limit - 1
        data_center = np.vstack([node_state, self.demand[step]])
        data_center = np.where(data_center > 1, 1, data_center)  # Clip values exceeding capacity to maximum
        self.state["state"] = data_center

        # Create valid action masks for each object at the current step
        for obj_idx, demand in enumerate(self.demand[self.current_step]):
            action_mask = (node_state[:, 1:] + demand[1:]) <= 1  # Compare demand of each object with proxy nodes
            self.proxy_validity_mask[obj_idx] = (action_mask.sum(axis=1) == 2).astype(int)  # Mark valid nodes as 1
        
        self.state["action_mask"] = self.proxy_validity_mask

    def sample_action(self):
        # Sample a random action for each object-node pair
        return self.action_space.sample()

    # def generate_demand(self):
    #     return self.generate_demand_random()
    
    def step(self, actions):
        # Main step function called externally
        return self._STEP(actions)

    def reset(self, **kwargs):
        # Reset the environment to the initial state
        return self._RESET()

    def valid_action_mask(self):
        # Return the valid action mask for each object at the current state
        return self.proxy_validity_mask

    def generate_demand(self):
        # Number of steps per day (e.g., 72 assessments throughout the day)
        n = self.step_limit  # Total steps representing assessments in a day
        
        # Probability distribution and categories for storage demand
        storage_probs = np.array([0.12, 0.165, 0.328, 0.287, 0.064, 0.036])
        storage_bins = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0])

        # Generate bandwidth demand using a normal distribution, clipped to 0-1 range
        bandwidth_demand = np.random.normal(loc=0.5, scale=0.1, size=(self.n_objects, n))
        bandwidth_demand = np.clip(bandwidth_demand, 0, 1)  # Ensure demand is within 0 to 1

        # Sample storage demand for each video object from specified bins
        storage_demand = np.random.choice(storage_bins, p=storage_probs, size=(self.n_objects, n))

        # Combine time ratio, bandwidth, and storage demand for each object, resulting in (72, 3) for each object
        return np.array([
            np.column_stack([np.arange(n) / n, bandwidth_demand[i], storage_demand[i]])
            for i in range(self.n_objects)
        ])
    
    def generate_demand_zipf(self):
        # Number of steps per day (e.g., 72 assessments throughout the day)
        n = self.step_limit  # Total steps representing assessments in a day
        
        # Zipf distribution parameter to skew bandwidth demand
        zipf_skew = 2.0  # Higher values increase skewness
        
        # Probability distribution and categories for storage demand
        storage_probs = np.array([0.12, 0.165, 0.328, 0.287, 0.064, 0.036])
        storage_bins = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0])

        # Generate bandwidth demand based on Zipf distribution
        # Zipf distribution provides demand indices with skewness, then scaled to 0-1
        zipf_indices = np.random.zipf(zipf_skew, size=(self.n_objects, n))
        bandwidth_demand = np.clip(zipf_indices / zipf_indices.max(), 0, 1)  # Scale values to between 0 and 1

        # Sample storage demand for each video object from specified bins
        storage_demand = np.random.choice(storage_bins, p=storage_probs, size=(self.n_objects, n))

        # Combine time ratio, bandwidth, and storage demand for each object, resulting in (72, 3) for each object
        return np.array([
            np.column_stack([np.arange(n) / n, bandwidth_demand[i], storage_demand[i]])
            for i in range(self.n_objects)
        ])