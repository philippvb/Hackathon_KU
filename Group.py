class Group:
    def __init__(self, num_contacts, prob_transmission, prob_severe, prob_death, susceptible, infectious=0, recovered=0,
                 severe_cases=0):
        # immutable group variables
        self.prob_transmission_group = prob_transmission
        self.prob_severe_group = prob_severe
        self.prob_death_group = prob_death
        self.num_contacts = num_contacts

        # subgroups
        self.vac1_ready = Subgroup(self, susceptible, infectious, recovered, severe_cases, None)
        self.in_process = {}
        self.vac2_ready = {}
        self.done = {}

    def step(self, vaccine_shipments, ratio_cases):
        # vaccinate
        for shipment in vaccine_shipments:
            # distribute first vaccines
            if not shipment.vaccine.name in self.in_process:
                self.in_process[shipment.vaccine.name] = Processing_Group(self, 0, 0, 0, 0, shipment.vaccine)
            self.move(self.vac1_ready, self.in_process[shipment.vaccine.name], shipment.first_vaccines)

            # distribute the second vaccines
            if not shipment.vaccine.name in self.done:
                self.done[shipment.vaccine.name] = Subgroup(self, 0, 0, 0, 0, shipment.vaccine)
            self.move(self.vac2_ready, self.done[shipment.vaccine.name], shipment.second_vaccines)


        # manage subgroups
        self.vac1_ready.step()
        for subgroup in self.in_process:
            num_to_move = subgroup.step()
            if not subgroup.vaccine.name in self.vac2_ready:
                self.vac2_ready[subgroup.vaccine.name] = Subgroup(self, 0, 0, 0, 0, subgroup.vaccine)
            self.move(subgroup, self.vac2_ready[subgroup.vaccine.name], num_to_move)
        for subgroup in self.vac2_ready:
            subgroup.step()
        for subgroup in self.done:
            subgroup.step()
        # move within subgroup
        return

    def move(self, in_group, out_group, number):
        if number > in_group.susceptible:
            raise Exception
        in_group.susceptible -= number
        out_group.susceptible += number


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
        self.prob_transmission = vaccine.effectiveness if vaccine else parent.prob_transmission_group
        self.prob_severe = vaccine.prob_severe if vaccine else parent.prob_severe_group
        self.prob_death = vaccine.prob_death if vaccine else parent.prob_death_group

        self.parent=parent

    def step(self, ratio_cases):
        new_infectious = self.susceptible * self.parent.num_contacts * self.prob_transmission + ratio_cases
        new_severe_cases = self.prob_severe * self.infectious
        new_deaths = self.prob_death * self.severe_cases

        self.susceptible -= new_infectious
        self.infectious = self.infectious + new_infectious - new_severe_cases
        self.severe_cases = self.severe_cases + new_severe_cases - new_deaths
        self.deaths += new_deaths

    def get_cases(self):
        return self.infectious + self.severe_cases


class Processing_Group(Subgroup):
    def __init__(self, processing_time):
        super.__init__()
        self.history = [0] * processing_time


    def step(self, ratio_cases, num_new_vaccines):
        super.step()
        # add the new number of vaccines
        self.history.insert(0, num_new_vaccines)
        return self.history.pop()




class Vaccine_Shipment:
    def __init__(self, vaccine, first_vaccines, second_vaccines):
        self.vaccine = vaccine
        self.first_vaccines = first_vaccines
        self.second_vaccines = second_vaccines

class Vaccine:
    def __init__(self, name, effectiveness, prob_severe, prob_death):
        self.name = name
        self.effectiveness = effectiveness
        self.prob_severe = prob_severe
        self.prob_death = prob_death