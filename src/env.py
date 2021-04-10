from src.Group import Group, Vaccine, Vaccine_Shipment
import pandas as pd

JOHNSON = Vaccine("Johnsonn", 0.95, 0.05, 0.01, 10)
BIONTECH = Vaccine("Biontech", 0.95, 0.05, 0.01, 15)

class VaccinationEnvironment:
    def __init__(self, config, vaccination_schedule):
        self.groups = []
        self.groups_history = []
        for index, group_name in enumerate(config["groups"]):
            susceptible = config["group_sizes"][index] - config["starting_cases"][index]
            self.groups.append(
                Group(group_name,
                config["num_contacts"],
                config["prob_transmission"],
                config["prob_severe"], 
                config["prob_death"],
                config["prob_recovery"],
                susceptible,
                config["starting_cases"][index]
            ))

            self.groups_history.append(pd.DataFrame(["susceptible", "cases", "recovered", "dead"]))


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

        self.save_step()

    def save_step(self):
        info = self.get_info()
        for index, group in enumerate(self.groups):
            self.groups_history[index] = self.groups_history[index].append(info[group.name])

    def get_total_cases(self):
        return sum([group.get_cases() for group in self.groups])
    # vaccines

    def get_info(self):
        info={"description": "The data is sorted by [susceptible, active_cases, recovered, deaths]"}
        for group in self.groups:
            info[group.name] = [group.get_susceptible(), group.get_cases(), group.get_recovered(), group.get_deaths()]
        return info
    # returns general information like current rates for group ...


    def get_groups(self):
        return self.groups

    def plot(self):
        pass
    # plots current status

    def save(self):
        pass
        # saves history


