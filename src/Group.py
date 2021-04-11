import random

class Group:
    """Represents a group of people.
    """
    def __init__(self, name, num_contacts, prob_transmission, prob_severe, prob_death, prob_recovery, susceptible,
                 infectious=0, recovered=0, severe_cases=0, possible_vaccines=[]):
        """Init method

        Args:
            name (string): Name of the group
            num_contacts (int): Number of contacts and individuum sees per timestep.
            prob_transmission (float): Probability of transmission from an infectious to susceptible person.
            prob_severe (float): Probability of a infection getting severe per timestep.
            prob_death (float): Probability of a severe case dying per timestep.
            prob_recovery (float): Probability of a severe case recovering per timestep.
            susceptible (int): Number of susceptible people.
            infectious (int, optional): Number of infectious people. Defaults to 0.
            recovered (int, optional): Number of recovered people. Defaults to 0.
            severe_cases (int, optional): Number of severe cases. Defaults to 0.
            possible_vaccines (list, optional): List of all possible vaccines that could be used.. Defaults to [].
        """
        # immutable group variables
        self.name = name
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
        for vaccine in possible_vaccines:
            self.in_process[vaccine.name] = Processing_Group(self, susceptible=0, vaccine=vaccine)
            self.vac2_ready[vaccine.name] = Subgroup(self, 0, 0, 0, 0, vaccine)
            self.done[vaccine.name] = Subgroup(self, 0, 0, 0, 0, vaccine)



    def step(self, ratio_cases, vaccination_plan=None):
        """Does one timestep with the given vaccine.

        Args:
            ratio_cases (float): The case ratio in the overall population
            vaccination_plan (list[Vaccination_Plan], optional): list of vaccines given at this timestep. Defaults to None.
        """
        # vaccinate
        if vaccination_plan:
            for shipment in vaccination_plan:
                # distribute first vaccines
                if shipment.first_vaccines > 0:
                    self.move(self.vac1_ready, self.in_process[shipment.vaccine.name], min(shipment.first_vaccines, self.vac1_ready.susceptible))

                # distribute the second vaccines
                if shipment.second_vaccines > 0:
                    self.move(self.vac2_ready[shipment.vaccine.name], self.done[shipment.vaccine.name], min(shipment.second_vaccines, self.vac2_ready[shipment.vaccine.name].susceptible))


        # manage subgroups
        self.vac1_ready.step(ratio_cases)
        for subgroup in self.in_process.values():
            num_to_move = subgroup.step(ratio_cases)
            self.move(subgroup, self.vac2_ready[subgroup.vaccine.name], min(num_to_move, subgroup.susceptible))
        for subgroup in self.vac2_ready.values():
            subgroup.step(ratio_cases)
        for subgroup in self.done.values():
            subgroup.step(ratio_cases)
        return

    def move(self, in_group, out_group, number):
        """Moves a given number of susceptible people from one to another group

        Args:
            in_group (Subgroup): Group to move from
            out_group (Subgroup): Group to move to
            number (int): Number of people

        Raises:
            Exception: If number > in_group.susceptible
        """
        if number > in_group.susceptible:
            raise Exception
        in_group.susceptible -= number
        out_group.add_people(number)

    
    def get_cases_vac1_ready(self):
        return self.vac1_ready.get_cases()

    def get_cases_vac2_ready(self):
        return [subgroup.get_cases() for subgroup in list(self.vac2_ready.values())]



    # get methods which collect infos of all subgroups

    def get_cases(self):
        total_cases = self.vac1_ready.get_cases()
        total_cases += sum([subgroup.get_cases() for subgroup in list(self.in_process.values())])
        total_cases += sum([subgroup.get_cases() for subgroup in list(self.vac2_ready.values())])
        total_cases += sum([subgroup.get_cases() for subgroup in list(self.done.values())])
        return total_cases

    def get_recovered(self):
        total_recovered = self.vac1_ready.recovered
        total_recovered += sum([subgroup.recovered for subgroup in list(self.in_process.values())])
        total_recovered += sum([subgroup.recovered for subgroup in list(self.vac2_ready.values())])
        total_recovered += sum([subgroup.recovered for subgroup in list(self.done.values())])
        return total_recovered

    def get_susceptible(self):
        total_susceptible = self.vac1_ready.susceptible
        total_susceptible += sum([subgroup.susceptible for subgroup in list(self.in_process.values())])
        total_susceptible += sum([subgroup.susceptible for subgroup in list(self.vac2_ready.values())])
        total_susceptible += sum([subgroup.susceptible for subgroup in list(self.done.values())])
        return total_susceptible

    def get_deaths(self):
        total_deaths = self.vac1_ready.deaths
        total_deaths += sum([subgroup.deaths for subgroup in list(self.in_process.values())])
        total_deaths += sum([subgroup.deaths for subgroup in list(self.vac2_ready.values())])
        total_deaths += sum([subgroup.deaths for subgroup in list(self.done.values())])
        return total_deaths



    def get_vaccinatable(self):
        vac2_ready_dict = {}
        vac1_ready_susceptible = self.vac1_ready.susceptible
        for subgroup_vac_2 in self.vac2_ready.values():
            vac2_ready_dict[subgroup_vac_2.vaccine.name] = subgroup_vac_2.susceptible
        return [vac1_ready_susceptible, vac2_ready_dict]

class Subgroup:
    """Represents a subgroup of a group, for example all people who are vaccinated once
    """
    def __init__(self, parent, susceptible, infectious=0, recovered=0, severe_cases=0, vaccine=None):
        """Init method

        Args:
            parent (group): The underlying group, used to get group variables like prob_transmission.
            susceptible (int): Number of susceptible people.
            infectious (int, optional): Number of infectious people. Defaults to 0.
            recovered (int, optional): Number of recovered people. Defaults to 0.
            severe_cases (int, optional): Number of severe cases. Defaults to 0.
            vaccine ([type], optional): The vaccine to use. Defaults to None.
        """
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

    def step(self, ratio_cases):
        """Does one environment step

        Args:
            ratio_cases (float): The ratio of cases in the whole population.
        """
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
    """Represents a group of people who have recently been vaccinated and are in
    incubation phase until the vaccine fully works.
    Inherits from Subgroup
    """
    def __init__(self, parent, susceptible, vaccine, infectious=0, recovered=0, severe_cases=0, processing_discount=1.2):
        """Init method. For not specified arguments, see subgroup

        Args:
            vaccine (Vaccine): The vaccine to get
            processing_discount (float, optional): The penalty factor for the incubation time,. Defaults to 1.2.
        """
        super().__init__(parent, susceptible, infectious, recovered, severe_cases, vaccine)
        self.history = [0] * vaccine.processing_time
        self.prob_transmission = processing_discount * self.prob_transmission
        self.prob_severe = processing_discount * self.prob_severe
        self.prob_death = processing_discount * self.prob_death
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




class Vaccination_Plan:
    """A vaccination plan or a timestep, contains info about how many vaccines
are handed to first and second time vaccined people in the group it is given to.
    """
    def __init__(self, vaccine, first_vaccines, second_vaccines):
        self.vaccine = vaccine
        self.first_vaccines = first_vaccines
        self.second_vaccines = second_vaccines

class Vaccine_Shipment:
    """A vaccine shipment that arrives and can be distributed
    """
    def __init__(self, vaccine, num):
        self.num = num
        self.vaccine = vaccine

class Vaccine:
    """Instance of one vaccine
    """
    def __init__(self, name, effectiveness, prob_severe, prob_death, processing_time):
        self.name = name
        self.effectiveness = effectiveness
        self.prob_severe = prob_severe
        self.prob_death = prob_death
        self.processing_time = processing_time
