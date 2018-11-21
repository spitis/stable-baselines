import gym
from goal_grid import GoalGridWorldEnv

# grid_file = '2_room_9x9.txt'
grid_file = 'room_5x5_empty.txt'
env = GoalGridWorldEnv(grid_size=5, max_step=25,grid_file=grid_file)
obs = env.reset()

# Act randomly
num_eps = 0
while(num_eps < 10):
    obs, reward, done, _ = env.step(env.action_space.sample())
    env.render()
    if done:
        print("Episode reward: {}".format(reward))
        obs = env.reset()
        num_eps += 1


