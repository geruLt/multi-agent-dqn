import numpy as np
import cv2
import random

from buffer import Buffer
from madqn import maDQN


N_AGENTS = 3
N_ACTIONS = 4
N_COLOR_CHANNELS = 1 
ENV_NAME = 'your_env'
env = your_env()

LOAD_MODELS = 1 ## Whether to load weigths or not
SAVE_NAME = '500' ## Weight name, weights must be under your_env directory example your_env/agent0-score-500.pack
best_score = ## Minimum score for your environments

N_EPISODES = 2000 
GAMMA=0.99
BATCH_SIZE=32
BUFFER_SIZE=int(7.5e5)
MIN_REPLAY_SIZE=10000
EPSILON_START= 1
EPSILON_END=0.01
EPSILON_DECAY= 400
TARGET_UPDATE_FREQ = 50
LR = 2.5e-4
PRINT_INTERVAL = 10
TRAIN_INTERVAL = 30
MAX_ACTIONS = 1000

def obs_grayscale_sizescale(observation):
    observation = [observation]*3
    new_observations = []
    for obs in observation:
        gray = cv2.cvtColor(obs,cv2.COLOR_BGR2GRAY)
        new_observation = cv2.resize(gray,(84,84),interpolation=cv2.INTER_AREA)
        new_observations.append(new_observation[np.newaxis,:,:])
    return new_observations

maDQN_agents = maDQN(n_agents=N_AGENTS, num_actions=N_ACTIONS,
                     color_channels=N_COLOR_CHANNELS, learning_rate=LR,
                     gamma=GAMMA, env_name=ENV_NAME)

if LOAD_MODELS:
    maDQN_agents.load_checkpoint(SAVE_NAME)


replay_buffer = Buffer(n_agents=N_AGENTS,buffer_size=BUFFER_SIZE,batch_size=BATCH_SIZE)


# Initialize replay buffer
obs = obs_grayscale_sizescale(env.reset())
for _ in range(MIN_REPLAY_SIZE):
    actions = np.random.randint(0,N_ACTIONS,size=N_AGENTS)
    new_obs, rewards, dones = env.step(actions)
    new_obs = obs_grayscale_sizescale(new_obs)
    transition = (obs, actions, rewards, dones, new_obs)
    replay_buffer.store(transition)
    obs = new_obs

    if any(dones):
        obs = obs_grayscale_sizescale(env.reset())


score_history = [-3100]*3

# Main Training Loop

for step in range(N_EPISODES):
    if step % 5 == 0:
        out = cv2.VideoWriter('videos/output_'+str(step)+'.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 60,(320,320))

    obs = obs_grayscale_sizescale(env.reset())

    dones = [False]*3
    episode_reward = 0
    total_actions = 0

    epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])

    if step % TARGET_UPDATE_FREQ == 0:
        maDQN_agents.target_update()

    while not any(dones):
        rnd_sample = random.random()
        if rnd_sample <= epsilon:
            actions = np.random.randint(0,N_ACTIONS,size=N_AGENTS)
        else:
            actions = maDQN_agents.choose_actions(obs)

       



        new_obs, rewards, dones = env.step(actions)
        new_obs = obs_grayscale_sizescale(new_obs)

        transition = (obs, actions, rewards, dones, new_obs)
        replay_buffer.store(transition)

        total_actions += 1
        if total_actions == MAX_ACTIONS:
            dones = [True] * 3
        if step % 5 == 0:
            frame = env.render()
            cv2.imshow('Environment',frame)
            cv2.waitKey(1)
        if step % 5 == 0:
            out.write(frame)
            if any(dones):
                out.release()



        obs = new_obs
        episode_reward += sum(rewards)



        if step % TRAIN_INTERVAL == 0:
            batch = replay_buffer.sample()
            maDQN_agents.learn(batch)

    score_history.append(episode_reward)
    # Logging
    if step % PRINT_INTERVAL == 0:
        avg_score = np.mean(score_history[-200:])
        print('Step:', step)
        print('Average Score: {:.2f}'.format(avg_score))
        np.save(ENV_NAME+'score_history.npy',np.array(score_history))
        if avg_score > best_score:
            best_score = avg_score
            maDQN_agents.save_checkpoint(best_score)
