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

vacination_schedule = {
    "Johnson": [100]*1000,
    "Biontech": [0]*100 + [15]*900
}


env = VaccinationEnvironment(config, vacination_schedule)
vaccines = env.get_vaccines()
print(env.get_vaccinatable())

vaccination_plan = Vaccination_Plan(JOHNSON, 100, 0)

for i in range(100):
    #env.step([[vaccination_plan]], print_warnings=False)
    env.step()
    #print(env.groups_history[0])
    #print(env.get_info(header=False))
    #print(env.get_cases())

#env.plot_ratio("deaths")

# [group1, grou2, ...]
# group1 = [vac1, vac2]

env.save("P:/Dokumente/3 Uni/WiSe2021/Hackathon/Hackathon_KU")