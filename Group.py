import random

class Group:
    def __init__(self, num_contacts, prob_transmission, prob_severe, prob_death, prob_recovery, susceptible,
                 infectious=0, recovered=0, severe_cases=0):
        # immutable group variables
        self.prob_transmission_group = prob_transmission
        self.prob_severe_group = prob_severe
        self.prob_death_group = prob_death
        self.prob_recovery_group = prob_recovery
        self.num_contacts = num_contacts

        # subgroups
        self.vac1_ready = Subgroup(self, susceptible, infectious, recovered, severe_cases, None)
        self.in_process = {}
        self.vac2_ready = {}
        self.done = {}

    def step(self, ratio_cases, vaccine_shipments=None):
        # vaccinate
        if vaccine_shipments:
            for shipment in vaccine_shipments:
                # distribute first vaccines
                if shipment.first_vaccines > 0:
                    if not shipment.vaccine.name in self.in_process:
                        self.in_process[shipment.vaccine.name] = Processing_Group(self, 0, 0, 0, 0, shipment.vaccine)
                    self.move(self.vac1_ready, self.in_process[shipment.vaccine.name], shipment.first_vaccines)

                # distribute the second vaccines
                if shipment.second_vaccines > 0:
                    if not shipment.vaccine.name in self.done:
                        self.done[shipment.vaccine.name] = Subgroup(self, 0, 0, 0, 0, shipment.vaccine)
                    self.move(self.vac2_ready[shipment.vaccine.name], self.done[shipment.vaccine.name], shipment.second_vaccines)


        # manage subgroups
        self.vac1_ready.step(ratio_cases)
        for subgroup in self.in_process.values():
            num_to_move = subgroup.step(ratio_cases)
            if not subgroup.vaccine.name in self.vac2_ready:
                self.vac2_ready[subgroup.vaccine.name] = Subgroup(self, 0, 0, 0, 0, subgroup.vaccine)
            self.move(subgroup, self.vac2_ready[subgroup.vaccine.name], min(num_to_move, subgroup.susceptible))
        for subgroup in self.vac2_ready.values():
            subgroup.step(ratio_cases)
        for subgroup in self.done.values():
            subgroup.step(ratio_cases)
        return

    def move(self, in_group, out_group, number):
        if number > in_group.susceptible:
            raise Exception
        in_group.susceptible -= number
        out_group.add_people(number)

    def get_cases(self):
        total_cases = self.vac1_ready.get_cases()
        total_cases += sum([subgroup.get_cases() for subgroup in list(self.in_process.values())])
        total_cases += sum([subgroup.get_cases() for subgroup in list(self.vac2_ready.values())])
        total_cases += sum([subgroup.get_cases() for subgroup in list(self.done.values())])
        return total_cases



class Subgroup:
    def __init__(self, parent, susceptible, infectious=0, recovered=0, severe_cases=0, vaccine=None):
        # variational variables
        self.susceptible= susceptible
        self.infectious = infectious
        self.recovered = recovered
        self.severe_cases = severe_cases
        self.deaths = 0

        #  specific to subgroup
        self.vaccine = vaccine  # todo create this object
        self.prob_transmission = (1 - vaccine.effectiveness) if vaccine else parent.prob_transmission_group
        self.prob_severe = vaccine.prob_severe if vaccine else parent.prob_severe_group
        self.prob_death = vaccine.prob_death if vaccine else parent.prob_death_group
        self.prob_recovery = parent.prob_recovery_group

        self.parent = parent

    def step(self, ratio_cases, *args, **kwargs):
        new_infectious = int(self.susceptible * self.parent.num_contacts * self.prob_transmission + ratio_cases)
        new_severe_cases = int(self.prob_severe * self.infectious)
        new_deaths = int(self.prob_death * self.severe_cases)
        new_recoveries_inf = int(self.prob_recovery * self.infectious)
        new_recoveries_sev = int(self.prob_recovery * self.severe_cases)


        self.susceptible -= new_infectious
        self.infectious = self.infectious + new_infectious - new_severe_cases - new_recoveries_inf
        self.severe_cases = self.severe_cases + new_severe_cases - new_deaths - new_recoveries_sev
        self.deaths += new_deaths
        self.recovered += new_recoveries_sev + new_recoveries_inf

    def get_cases(self):
        return self.infectious + self.severe_cases

    def add_people(self, num):
        self.susceptible += num


class Processing_Group(Subgroup):
    def __init__(self, parent, susceptible, infectious=0, recovered=0, severe_cases=0, vaccine=None):
        super().__init__(parent, susceptible, infectious, recovered, severe_cases, vaccine)
        self.history = [0] * vaccine.processing_time
        self.last_added = 0

    def add_people(self, num):
        super().add_people(num)
        self.last_added = num

    def step(self, ratio_cases):
        super().step(ratio_cases)
        # add the new number of vaccines
        num_new_vaccines = self.last_added
        self.history.insert(0, num_new_vaccines)
        self.last_added = 0
        return self.history.pop()




class Vaccine_Shipment:
    def __init__(self, vaccine, first_vaccines, second_vaccines):
        self.vaccine = vaccine
        self.first_vaccines = first_vaccines
        self.second_vaccines = second_vaccines

class Vaccine:
    def __init__(self, name, effectiveness, prob_severe, prob_death, processing_time):
        self.name = name
        self.effectiveness = effectiveness
        self.prob_severe = prob_severe
        self.prob_death = prob_death
        self.processing_time = processing_time
