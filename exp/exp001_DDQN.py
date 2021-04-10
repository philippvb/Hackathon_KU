import src.DDQN.DDQN_PER_MS as ddqn
import src.Group
import src.env

# ------------------------------------------------------------------------------------------------------------
# initialize the environment
groups = {
    "groups": ["20s", "30s", "40s"],
    "starting_cases": [100, 100, 100],
    "group_sizes": [1000, 1000, 1000],
    "num_contacts": 4,
    "prob_transmission": 0.1,
    "prob_severe": 0.1,
    "prob_death": 0.1,
    "prob_recovery": 0.1,
}
vaccination_schedule = {
    "Johnson": [10]*1000,
    "Biontech": [0]*100 + [15]*900
}
env = src.env.VaccinationEnvironment(groups, vaccination_schedule)
total_people = sum(groups["group_sizes"])
# ------------------------------------------------------------------------------------------------------------
# make the agent
discount = 0.9
eps = 0.05
max_episodes = 600
max_steps = vaccination_schedule[list(vaccination_schedule.keys())[0]]
target_update = 20
train_reps = 100
pereps = 0.05
alpha = 0.6
beta = 0.4
beta_inc = 0.0001

o_space = 2 * len(groups["groups"])
ac_space = total_people
ac_space_actions = 2 * len(vaccination_schedule.keys())
use_target = True

agent = ddqn.DQNAgent(o_space, ac_space, ac_space_actions, discount=discount, eps=eps,
                   use_target_net=use_target, update_target_every=target_update, per_eps=pereps,
                      alpha=alpha, beta=beta, beta_inc=beta_inc)
# ------------------------------------------------------------------------------------------------------------
# train
# ------------------------------------------------------------------------------------------------------------
# test
