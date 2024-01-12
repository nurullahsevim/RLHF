import gymnasium as gym
import torch as th

from stable_baselines3 import PPO,A2C,SAC



if __name__=="__main__":

    env = gym.make("CartPole-v1", render_mode="rgb_array")

    model = A2C("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)

    vec_env = model.get_env()
    obs = vec_env.reset()
    for i in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render("human")
        # VecEnv resets automatically
        # if done:
        #   obs = vec_env.reset()


    """
    # Custom actor (pi) and value function (vf) networks
    # of two layers of size 32 each with Relu activation function
    # Note: an extra linear layer will be added on top of the pi and the vf nets, respectively
    policy_kwargs = dict(activation_fn=th.nn.ReLU,
                         net_arch=dict(pi=[32, 32], vf=[32, 32]))
    # Create the agent
    model = PPO("MlpPolicy", "CartPole-v1", policy_kwargs=policy_kwargs, verbose=1)
    # Retrieve the environment
    env = model.get_env()
    # Train the agent
    model.learn(total_timesteps=20_000)
    # Save the agent
    model.save("ppo_cartpole")

    del model
    # the policy_kwargs are automatically loaded
    model = PPO.load("ppo_cartpole", env=env)
    """