from Group import Group, Vaccine, Vaccine_Shipment
import pandas as pd

JOHNSONN = Vaccine("Johnsonn", 0.95, 0.05, 0.01)
BIONTECH = Vaccine("Biontech", 0.95, 0.05, 0.01)

class VaccinationEnvironment:
    def __init__(self, config, vaccination_schedule):
        self.groups = []
        self.groups_history = []
        for index, group in enumerate(config["groups"]):
            susceptible = config["group_sizes"][index] - config["starting_cases"][index]
            self.groups.append(
                Group(config["num_contacts"],
                config["prob_transmission"],
                config["prob_severe"], 
                config["prob_death"],
                config["prob_recovery"],
                susceptible,
                config["starting_cases"][index]
            ))

            self.groups_history.append(pd.DataFrame("susceptible"))


        self.vacination_schedule = vaccination_schedule
        self.total_people = sum(config["group_sizes"])

        self.time_step = 0


        # add vacin shipping schedule

    def get_vaccines(self):
        return [Vaccine_Shipment(globals()[vaccine_name.upper()], self.vacination_schedule[vaccine_name][index]) for index, vaccine_name in enumerate(self.vacination_schedule.keys())]
    # returns next vaccine objects

    def step(self, vaccines):
        ratio = self.get_total_cases()/self.total_people
        # vaccines: list of tuples (group, vaccinationshipping)
        for index, vaccine in enumerate(vaccines):
            self.groups[index].step(ratio, vaccine)

    def get_total_cases(self):
        return sum([group.get_cases() for group in self.groups])
    # vaccines

    def get_info(self):
        pass
    # returns general information like current rates for group ...

    def get_groups(self):
        return self.groups

    def plot(self):
        pass
    # plots current status

    def save(self):
        pass
        # saves history



config = {
    "groups": ["20s", "30s", "40s"],
    "starting_cases": [100, 100, 100],
    "group_sizes": [1000, 1000, 1000],
    "num_contacts": 4,
    "prob_severe": 0.1,
    "prob_death": 0.1,
    "prob_recover": 0.1,
}


vacination_schedule = {
    "Johnson": [10]*1000,
    "Biontech": [0]*100 + [15]*900
}