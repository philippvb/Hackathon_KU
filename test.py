from src.Group import Vaccination_Plan
from src.env import VaccinationEnvironment, JOHNSON, BIONTECH


config = {
    "groups": ["20s", "30s", "40s"],
    "starting_cases": [100, 100, 100],
    "group_sizes": [1000, 1000, 1000],
    "num_contacts": 4,
    "prob_transmission": 0.1,
    "prob_severe": 0.1,
    "prob_death": 0.1,
    "prob_recovery": 0.1,
}


vacination_schedule = {
    "Johnson": [10]*1000,
    "Biontech": [0]*100 + [15]*900
}


env = VaccinationEnvironment(config, vacination_schedule)
vaccines = env.get_vaccines()

vaccination_plan = (Vaccination_Plan(JOHNSON, 10, 0))
