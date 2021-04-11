import sys
sys.path.append("P:/Dokumente/3 Uni/WiSe2021/Hackathon/Hackathon_KU/src")

from env import VaccinationEnvironment, JOHNSON
from agents.agent import TD3Agent
from agents.utils import ReplayBuffer
from Group import Vaccination_Plan

import pandas as pd
from tqdm import tqdm




def training_loop(agent_config, env_config, vaccination_schedule):
    action_dim = len(env_config["groups"]) * 2 * len(vaccination_schedule)
    state_dim = len(env_config["groups"]) +  len(vaccination_schedule) * len(env_config["groups"]) + len(vaccination_schedule)
    batch_size = 100
    episodes = 100
    max_iterations = env_config["max_time_steps"]



    env = VaccinationEnvironment(env_config, vaccination_schedule)
    agent = TD3Agent([state_dim + action_dim, 256, 256, 1], [state_dim, 256, 256, action_dim])
    replay_buffer = ReplayBuffer(state_dim, action_dim)

    # get starting state
    state = []
    for items in list(env.get_cases().values()):
        for value in items:
            state.append(value)
    state.append(env.get_vaccines()[0].num)


# first sample enough data
    for i in range(2 * batch_size):
        available_vaccines=env.get_vaccines()[0].num


        
        action = agent.act(state)

        vaccination_plan = []
        for index in range(0, len(action), 2):
            group_vac_1 = int(action[index] * available_vaccines)
            group_vac_2 = int(action[index + 1] * available_vaccines)
            plan = Vaccination_Plan(JOHNSON, group_vac_1, group_vac_2)
            vaccination_plan.append([plan])
        
        info, done = env.step(vaccination_plan, False)
        reward = -sum([values[1] for values in info.values()])

        next_state = []
        for items in list(env.get_cases().values()):
            for value in items:
                next_state.append(value)
        next_state.append(available_vaccines)

        replay_buffer.add(state, action, next_state, reward, done)

        state = next_state

    # training
    rewards = []
    losses = []
    # get starting  state
    for i in tqdm(range(episodes)):
        episode_reward = []
        episode_loss=[]

        env.reset()
        state = []
        for items in list(env.get_cases().values()):
            for value in items:
                state.append(value)
        state.append(env.get_vaccines()[0].num)

        for j in range(max_iterations):
            available_vaccines=env.get_vaccines()[0].num
            action = agent.act(state)

            vaccination_plan = []
            for index in range(0, len(action), 2):
                group_vac_1 = int(action[index] * available_vaccines)
                group_vac_2 = int(action[index + 1] * available_vaccines)
                plan = Vaccination_Plan(JOHNSON, group_vac_1, group_vac_2)
                vaccination_plan.append([plan])
            
            info, done = env.step(vaccination_plan, False)
            reward = -sum([values[1] for values in info.values()])
            episode_reward.append(reward)

            next_state = []
            for items in list(env.get_cases().values()):
                for value in items:
                    next_state.append(value)
            next_state.append(available_vaccines)

            replay_buffer.add(state, action, next_state, reward, done)
            state = next_state


            # train
            train_states, train_actions, train_next_states, train_reward, train_done = replay_buffer.sample(batch_size)
            loss = agent.train(train_states, train_actions, train_next_states, train_reward, train_done)

            episode_loss.append(loss.detach().numpy())
            

            if done:
                rewards.append(sum(episode_reward))
                losses.append(sum(episode_loss)/len(episode_loss))
                break


    # finally save data:
    data = pd.DataFrame(data={"rewards":rewards, "losses": losses})
    data.to_csv("P:/Dokumente/3 Uni/WiSe2021/Hackathon/Hackathon_KU/exp/performance.csv", sep=",", index=False)



        


        

        
































config = {
    "groups": ["20s", "30s", "40s"],
    "starting_cases": [200, 100, 100],
    "group_sizes": [1000, 1000, 1000],
    "num_contacts": 4,
    "prob_transmission": 0.1,
    "prob_severe": 0.1,
    "prob_death": 0.1,
    "prob_recovery": 0.1,
    "max_time_steps":100
}


vacination_schedule = {
    "Johnson": [100]*1000,
    #"Biontech": [0]*100 + [15]*900
}

training_loop(None, config, vacination_schedule)
