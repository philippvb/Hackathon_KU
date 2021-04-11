import sys
import os

from numpy.lib.npyio import NpzFile
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Group import Group, Vaccine, Vaccine_Shipment


import pandas as pd
import matplotlib.pyplot as plt

JOHNSON = Vaccine("Johnson", 0.95, 0.05, 0.01, 10)
BIONTECH = Vaccine("Biontech", 0.95, 0.05, 0.01, 15)

class VaccinationEnvironment:
    def __init__(self, config, vaccination_schedule):
        self.groups = {}
        self.group_sizes = config["group_sizes"]
        self.groups_history = []
        self.possible_vaccines = [globals()[name.upper()] for name in vaccination_schedule.keys()]
        for index, group_name in enumerate(config["groups"]):
            susceptible = config["group_sizes"][index] - config["starting_cases"][index]
            self.groups[group_name]= Group(group_name,
                config["num_contacts"][index],
                config["prob_transmission"][index],
                config["prob_severe"][index], 
                config["prob_death"][index],
                config["prob_recovery"][index],
                susceptible,
                config["starting_cases"][index],
                possible_vaccines = self.possible_vaccines
            )

            self.groups_history.append(pd.DataFrame(columns=["susceptible", "cases", "recovered", "deaths"]))



        self.vaccination_schedule = vaccination_schedule
        self.config = config
        self.total_people = sum(config["group_sizes"])

        self.time_step = 0
        self.max_time_steps=config["max_time_steps"]


        # add vacin shipping schedule

    def reset(self, new_config=None):
        config = new_config if new_config else self.config
        self.groups = {}
        self.group_sizes = config["group_sizes"]
        self.groups_history = []
        for index, group_name in enumerate(config["groups"]):
            susceptible = config["group_sizes"][index] - config["starting_cases"][index]
            self.groups[group_name]= Group(group_name,
                config["num_contacts"][index],
                config["prob_transmission"][index],
                config["prob_severe"][index], 
                config["prob_death"][index],
                config["prob_recovery"][index],
                susceptible,
                config["starting_cases"][index],
                possible_vaccines = self.possible_vaccines
            )

            self.groups_history.append(pd.DataFrame(columns=["susceptible", "cases", "recovered", "deaths"]))

        self.total_people = sum(config["group_sizes"])
        self.time_step = 0



    def get_vaccines(self):
        return [Vaccine_Shipment(globals()[vaccine_name.upper()], self.vaccination_schedule[vaccine_name][index]) for index, vaccine_name in enumerate(self.vaccination_schedule.keys())]
    # returns next vaccine objects

    def step(self, vaccines=None, print_warnings=True, match_keyword=False):
        ratio = self.get_total_cases()/self.total_people

        # look if dim match
        last_index = len(self.groups)
        if len(vaccines) != last_index:
            if print_warnings:
                print("Warning: Mismatch between vaccination plan and groups.")
            last_index = len(vaccines)

        self.check_vaccination_plan(vaccines)

        # do the actual step
        if match_keyword:
            for group_name, vaccine in vaccines.items:
                self.groups[group_name].step(vaccine)
        else:
            for index, group in enumerate(self.groups.values()):
                if index >= last_index:
                    group.step(ratio)
                else:
                    group.step(ratio, vaccines[index])
                

        self.save_step()
        self.time_step += 1
        done = False
        if self.time_step >= self.max_time_steps:
            self.reset()
            done = True
        return self.get_info(False), done

    def save_step(self):
        info = self.get_info()
        for index, group_name in enumerate(self.groups.keys()):
            self.groups_history[index] = self.groups_history[index].append(pd.DataFrame([info[group_name]], columns=["susceptible", "cases", "recovered", "deaths"]), ignore_index=True)


    def get_total_cases(self):
        return sum([group.get_cases() for group in self.groups.values()])
    # vaccines

    def get_cases(self):
        return {group.name: [group.get_cases_vac1_ready()] + group.get_cases_vac2_ready() for group in self.groups.values()}

    def get_info(self, header=True):
        info={"description": "The data is sorted by [susceptible, active_cases, recovered, deaths]"} if header else {}
        for group in self.groups.values():
            info[group.name] = [group.get_susceptible(), group.get_cases(), group.get_recovered(), group.get_deaths()]
        return info
    # returns general information like current rates for group ...

    def get_vaccinatable(self):
        vaccinatable = {}
        for group in self.groups.values():
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

    def plot_ratio(self, key):
        plot_count = len(self.groups)
        fig, axs = plt.subplots(plot_count)
        fig.suptitle(f"Ratio of {key}")
        for index, group in enumerate(self.groups.values()):
            axs[index].set_title(group.name)
            self.groups_history[index][key].divide(self.group_sizes[index]).plot(ax = axs[index])
        plt.show()
    # plots current status


    def save(self, save_dir):
        for group_history, group_name in zip(self.groups_history, self.groups.keys()):
            group_history.to_csv(save_dir + f"/{group_name}.csv" , sep=",", index=False)
        # saves history


