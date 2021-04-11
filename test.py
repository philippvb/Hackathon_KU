from src.Group import Vaccination_Plan
from src.env import VaccinationEnvironment, JOHNSON, BIONTECH
config = {
    "groups": ["20s", "30s", "40s"],
    "starting_cases": [200, 100, 100],
    "group_sizes": [1000, 1000, 1000],
    "num_contacts": [4]*3,
    "prob_transmission": [0.1]*3,
    "prob_severe": [0.1]*3,
    "prob_death": [0.1]*3,
    "prob_recovery": [0.1]*3,
    "max_time_steps":101
}

config = {
    "groups": ["20-40s", "40-60s", "60-80s"],
    "starting_cases": [100, 50, 50],
    "group_sizes": [10000, 10000, 10000],
    "num_contacts": [4, 2, 1],
    "prob_transmission": [0.1]*3,
    "prob_severe": [0.1, 0.3, 0.5],
    "prob_death": [0.1, 0.3, 0.5],
    "prob_recovery": [0.1, 0.05, 0.01],
    "max_time_steps":150
}


new_config = {
    "groups": ["20s", "30s"],
    "starting_cases": [3000, 500],
    "group_sizes": [10000, 10000],
    "num_contacts": [5, 2],
    "prob_transmission": [0.2, 0.1],
    "prob_severe": [0.3, 0.1],
    "prob_death": [0.1]*2,
    "prob_recovery": [0.1]*2,
    "max_time_steps":101
}


vacination_schedule = {
    "Johnson": [300]*100 + [0]*90,
    #"Biontech": [0]*100 + [15]*900
}


env = VaccinationEnvironment(config, vacination_schedule)
vaccines = env.get_vaccines()
print(env.get_vaccinatable())

vaccination_plan = Vaccination_Plan(JOHNSON, 100, 0)


for i in range(149):
    # env.step({"20-40s":[vaccination_plan], "40-60s":[vaccination_plan], "60-80s": [vaccination_plan]}, print_warnings=False, match_keyword=True)
    # env.step([[vaccination_plan]*3])
    env.step()
    #print(env.groups_history[0])
    #print(env.get_info(header=False))
    #print(env.get_cases())

# for i in range(90):
#     env.step()

# env.plot_ratio("susceptible")
# env.plot_ratio("cases")
env.plot_ratio("deaths")
# env.plot_ratio("recovered")
# [group1, grou2, ...]
# group1 = [vac1, vac2]

#env.save("P:/Dokumente/3 Uni/WiSe2021/Hackathon/Hackathon_KU")