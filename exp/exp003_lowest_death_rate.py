import Hackathon_KU.src.Group
import Hackathon_KU.src.env


def get_prob(elt):
    return elt[1]
# ------------------------------------------------------------------------------------------------------------
# initialize the environment


groups = {
    "groups": ["20s", "30s", "40s"],
    "starting_cases": [100, 100, 100],
    "group_sizes": [1000, 1000, 1000],
    "num_contacts": [4, 3, 2],
    "prob_transmission": [0.1, 0.3, 0.2],
    "prob_severe": [0.1, 0.2, 0.3],
    "prob_death": [0.05, 0.1, 0.2],
    "prob_recovery": [0.4, 0.3, 0.2],
    "max_time_steps": 365,
}
vaccination_schedule = {
    "Johnson": [10]*1000,
    "Biontech": [0]*100 + [15]*900
}
env = Hackathon_KU.src.env.VaccinationEnvironment(groups, vaccination_schedule)
# ------------------------------------------------------------------------------------------------------------
# make the agent
for a in range(100):
    new_vaccines = env.get_vaccines()
    people = env.get_vaccinatable()
    plan = {}
    for group in groups["groups"]:
        group_plans = {}
        for vaccine in vaccination_schedule.keys():
            group_plans[vaccine] = Hackathon_KU.src.Group.Vaccination_Plan(env.get_vaccine_options(vaccine), 0, 0)
        plan[group] = group_plans
    # first distribute second vaccines
    for i in range(len(new_vaccines)):
        # if nothing left, done
        for g in people.keys():
            shipment = new_vaccines[i]
            if shipment.num == 0:
                break
            in_process = people[g][1]
            temp = shipment.vaccine
            if in_process[shipment.vaccine.name] > shipment.num:
                to_move = shipment.num
            else:
                to_move = in_process[shipment.vaccine.name]
            # add into plan
            fv = plan[g][shipment.vaccine.name].first_vaccines
            sv = plan[g][shipment.vaccine.name].second_vaccines
            vac = plan[g][shipment.vaccine.name].vaccine
            sv += to_move
            plan[g][shipment.vaccine.name] = Hackathon_KU.src.Group.Vaccination_Plan(vac, fv, sv)
            # subtract from shipment
            num = shipment.num - to_move
            vaccine = shipment.vaccine
            new_vaccines[i] = Hackathon_KU.src.Group.Vaccine_Shipment(vaccine, num)
    # then give to groups with highest death rate
    rates = []
    for i in range(len(groups["groups"])):
        rates.append([groups["groups"][i], groups["prob_death"][i]])
    rates.sort(key=get_prob)
    for i in range(len(new_vaccines)):
        for group in rates:
            shipment = new_vaccines[i]
            if new_vaccines[i].num == 0:
                break
            group_name = group[0]
            vaccine_name = new_vaccines[i].vaccine.name
            if people[group_name][0] < shipment.num:
                to_move = people[group_name][0]
            else:
                to_move = shipment.num
            # move into plan
            fv = plan[group_name][vaccine_name].first_vaccines
            sv = plan[group_name][vaccine_name].second_vaccines
            vac = plan[group_name][vaccine_name].vaccine
            sv += to_move
            plan[group_name][vaccine_name] = Hackathon_KU.src.Group.Vaccination_Plan(vac, fv, sv)
            # subtract from shipment
            num = shipment.num - to_move
            vaccine = shipment.vaccine
            new_vaccines[i] = Hackathon_KU.src.Group.Vaccine_Shipment(vaccine, num)

    # reformat
    plan = {name: list(group_vaccines.values()) for name, group_vaccines in plan.items()}
    # ------------------------------------------------------------------------------------------------------------
    # run
    env.step(plan, match_keyword=True)
    # ------------------------------------------------------------------------------------------------------------

