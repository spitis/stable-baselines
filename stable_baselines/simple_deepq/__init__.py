from stable_baselines.common.policies import MlpPolicy, CnnPolicy, LnMlpPolicy, LnCnnPolicy
from stable_baselines.simple_deepq.build_graph import build_train  # noqa
from stable_baselines.simple_deepq.dqn import SimpleDQN
from stable_baselines.common.replay_buffer import ReplayBuffer


def wrap_atari_dqn(env):
    """
    wrap the environment in atari wrappers for DQN

    :param env: (Gym Environment) the environment
    :return: (Gym Environment) the wrapped environment
    """
    from stable_baselines.common.atari_wrappers import wrap_deepmind
    return wrap_deepmind(env, frame_stack=True, scale=False)
