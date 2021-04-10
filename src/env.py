from sys import call_tracing
from src.Group import Group, Vaccine, Vaccine_Shipment
import pandas as pd

JOHNSON = Vaccine("Johnson", 0.95, 0.05, 0.01, 10)
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


        self.vaccination_schedule = vaccination_schedule
        self.total_people = sum(config["group_sizes"])

        self.time_step = 0


        # add vacin shipping schedule

    def get_vaccines(self):
        return [Vaccine_Shipment(globals()[vaccine_name.upper()], self.vaccination_schedule[vaccine_name][index]) for index, vaccine_name in enumerate(self.vaccination_schedule.keys())]
    # returns next vaccine objects

    def step(self, vaccines=None, check=True):
        ratio = self.get_total_cases()/self.total_people

        # look if dim match
        last_index = len(self.groups)
        if len(vaccines) != last_index:
            print("Warning: Mismatch between vaccination plan and groups.")
            last_index = len(vaccines)

        self.check_vaccination_plan(vaccines)

        # do the actual step
        for index, group in enumerate(self.groups):
            if index >= last_index:
                group.step(ratio)
            else:
                group.step(ratio, vaccines[index])

        self.save_step()
        self.time_step += 1

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

    def get_vaccinatable(self):
        vaccinatable = {}
        for group in self.groups:
            vaccinatable[group.name] = group.get_vaccinatable() # TODO: or store it as dict again??
        return vaccinatable



    def check_vaccination_plan(self, vaccination_plan):
        vaccine_plan_sum = {name: 0 for name in self.vaccination_schedule.keys()}
        for group_plan in vaccination_plan:
            for group_vaccine_plan in group_plan:
                vaccine_plan_sum[group_vaccine_plan.vaccine.name] = vaccine_plan_sum[group_vaccine_plan.vaccine.name] + group_vaccine_plan.first_vaccines + group_vaccine_plan.second_vaccines

        for index,(vaccine_plan, vaccine_schedule) in enumerate(zip(vaccine_plan_sum.values(), self.vaccination_schedule.values())):
            if vaccine_plan > vaccine_schedule[self.time_step]:
                raise ValueError (f"You want to use {vaccine_plan - vaccine_schedule[self.time_step]} more vaccines that are currently availabe from {self.vaccination_schedule.keys()[index]}")


    def get_groups(self):
        return self.groups

    def plot(self):
        pass
    # plots current status

    def save(self):
        pass
        # saves history


