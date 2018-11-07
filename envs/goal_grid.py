import os
import numpy as np
import random
from enum import IntEnum
import matplotlib.pyplot as plt

import gym
from gym import error, spaces
from gym.utils import seeding

class GoalGridWorldEnv(gym.GoalEnv):
    """
    A simple 2D grid world environment with goal-oriented reward.
    Compatible with the OpenAI GoalEnv class.
    Observations are a dict of 'observation', 'achieved_goal', and 'desired goal'
    """

    class Actions(IntEnum):
        # Move
        up = 0
        right = 1
        down = 2
        left = 3

    class ObjectTypes(IntEnum):
        # Object types
        empty = 0
        agent = 1
        wall = 2
        lava = 3

    MOVE_DIRECTION = [[0,-1],[1,0],[0,1],[-1,0]] # up, right, down, left

    def __init__(self, grid_size=16, max_step=100, grid_file=None, seed=1337):
        # Action enumeration
        self.actions = GoalGridWorldEnv.Actions

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(self.actions))

        # Object types
        self.objects = GoalGridWorldEnv.ObjectTypes

        # Environment configuration
        self.grid_size = grid_size
        self.max_step = max_step

        self.end_of_game = False
        self.agent_pos = [0,0]

        if grid_file:
            curr_abs_path = os.path.dirname(os.path.abspath(__file__))
            rel_path = os.path.join(curr_abs_path, "grid_samples", grid_file)
            if os.path.exists(rel_path):
                grid_file = rel_path
                self.grid = np.loadtxt(grid_file, delimiter=',')
                # Overwrite grid size if necessary
                self.grid_size = self.grid.shape[0]
            else:
                print("Cannot find path: {}".format(rel_path))
        else:
            # Generate an empty grid
            self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int)

            # Sample the agent
            self.goal_loc = self._sample_goal_loc()
            self.goal = np.copy(self.grid)
            self.goal[self.goal_loc[0], self.goal_loc[1]] = self.objects.agent

        # Observations are dictionaries containing an
        # grid observation, achieved and desired goals
        observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.grid_size, self.grid_size, len(self.objects)),
            dtype='uint8'
        )
        self.observation_space = spaces.Dict({
            'observation': observation_space,
            'desired_goal': observation_space,
            'achieved_goal': observation_space
        })

        self.num_step = 0



    # Env methods
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.end_of_game = False
        self.num_step = 0
        self.agent_pos = [0,0]
        self.goal_loc = self._sample_goal_loc()
        self.goal = np.copy(self.grid)
        self.goal[self.goal_loc[0], self.goal_loc[1]] = self.objects.agent

        return self._get_obs()

    def step(self, action):
        """
        Taking a step in the environment.

        :param action:
        :return:
        """
        assert self.action_space.contains(action)
        # Take action, get reward, get observation
        self._take_action(action)
        obs = self._get_obs()
        info = {}
        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], info)

        self.num_step += 1

        if reward == 1.0 or self.num_step == self.max_step or self.end_of_game:
            done = 1
        else:
            done = 0

        return obs, reward, done, info

    def _take_action(self, action):
        """
        Performs the action on the grid. Updates the grid information.

        :param action:
        :return:
        """
        # Move the agent in that direction
        new_agent_pos = self._get_new_position(action)

        # Check if this position is a wall
        if self.grid[new_agent_pos[0], new_agent_pos[1]] != self.objects.wall:
            self.agent_pos = new_agent_pos

        return

    def _get_new_position(self, action):
        """
        Apply the action to change the agent's position

        :param action:
        :return:
        """
        new_agent_pos = [self.agent_pos[0] + self.MOVE_DIRECTION[action][0],
                         self.agent_pos[1] + self.MOVE_DIRECTION[action][1]]

        # Check if the new location is out of boundary
        if new_agent_pos[0] < 0 or new_agent_pos[1] < 0 \
            or new_agent_pos[0] > (self.grid_size - 1) or new_agent_pos[1] > (self.grid_size - 1):
            return self.agent_pos
        else:
            return new_agent_pos

    def _get_reward(self):
        if self.agent_pos[0] == self.goal_loc[0] and self.agent_pos[1] == self.goal_loc[1]:
            return 1.0
        else:
            return 0.0

    def _get_obs(self):
        """
        Return the observation as a dictionary of observation and goals

        :return:
        """
        obs = {
            'observation': self._get_state(self.grid),
            'desired_goal': self.one_hot(self.goal, len(self.objects)),
            'achieved_goal': self._get_state(self.grid)
        }
        return obs

    def _get_state(self, grid, use_one_hot=True):
        """
        Get the grid information.

        :return:
        """
        # Convert to one-hot representation: [NxNxK] where N is the size of grid and K is number of object types
        state = np.copy(grid)

        # Set the agent's position on the grid
        state[self.agent_pos[0],self.agent_pos[1]] = self.objects.agent

        if use_one_hot:
            state = self.one_hot(state, len(self.objects))

        return state

    def _sample_goal(self):
        """
        Sample an achievable state for the agent's goal

        :return:
        """
        pass

    def _sample_goal_loc(self):
        """
        Generate a goal location. Make sure that it is not a wall location
        :return:
        """
        coord_1 = random.randint(0, self.grid_size - 1) # TODO: Make it dependent on the seed
        coord_2 = random.randint(0, self.grid_size - 1)

        # Make sure that the sampled goal position is empty
        while self.grid[coord_1,coord_2] != self.objects.empty:
            coord_1 = random.randint(0, self.grid_size - 1)  # TODO: Make it dependent on the seed
            coord_2 = random.randint(0, self.grid_size - 1)

        return coord_1, coord_2

    def render(self, mode='human'):
        columns = 2
        rows = 1
        plt.clf()

        imgs = [self._get_state(self.grid, use_one_hot=False), self.goal]
        titles = ["Observation", "Goal"]
        for i in range(1, columns * rows + 1):
            ax = plt.subplot(rows, columns, i)
            ax.set_title(titles[i-1])
            plt.imshow(imgs[i-1])

        plt.figtext(0.5, 0.1, 'Time step: {}'.format(self.num_step), horizontalalignment='center')

        plt.pause(0.01)
        plt.clim(0, 10)
        plt.show(block=False)

        return

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Compute the step reward. This externalizes the reward function and makes
        it dependent on an a desired goal and the one that was achieved. If you wish to include
        additional rewards that are independent of the goal, you can include the necessary values
        to derive it in info and compute it accordingly.
        Args:
            achieved_goal (object): the goal that was achieved during execution
            desired_goal (object): the desired goal that we asked the agent to attempt to achieve
            info (dict): an info dictionary with additional information
        Returns:
            float: The reward that corresponds to the provided achieved goal w.r.t. to the desired
            goal. Note that the following should always hold true:
                ob, reward, done, info = env.step()
                assert reward == env.compute_reward(ob['achieved_goal'], ob['goal'], info)
        """
        if np.sum((achieved_goal - desired_goal)**2) > 0:
            return 0.0
        else:
            return 1.0

    def one_hot(self, vec, size):
        flattened = vec.flatten()
        oh = np.eye(size)[flattened.astype(int)]
        oh = oh.reshape(self.grid_size, self.grid_size, size)
        return oh

