from agents.dqn import DQNAgent
from agents.alegaatr import AlegAATr
from agents.aleqgaatr import AleqgAATr
from agents.madqn import MADQN
from agents.qalegaatr import QAlegAATr
from agents.ralegaatr import RAlegAATr
from agents.rawo import RawO
from agents.rdqn import RDQN
from agents.smalegaatr import SMAlegAATr
from agents.soaleqgaatr import SOAleqgAATr
from copy import deepcopy
from game.main import run_with_specified_agents
from game.prisoners_dilemma import PrisonersDilemma
import numpy as np
from typing import List


# Variables
N_AGENTS = 10
N_ITERATIONS = 20
N_ROUNDS = 30
progress_percentage_chunk = int(0.05 * N_ITERATIONS)

# Frequencies and population details
agent_frequencies = [1 / N_AGENTS] * N_AGENTS
algorithms = [
    DQNAgent(PrisonersDilemma(), 0),
    MADQN(PrisonersDilemma(), 0),
    RDQN(PrisonersDilemma(), 0),
    AleqgAATr(PrisonersDilemma(), 0),
    RAlegAATr(PrisonersDilemma(), 0),
    SOAleqgAATr(PrisonersDilemma(), 0),
    AlegAATr(PrisonersDilemma(), 0, lmbda=0.0, ml_model_type='knn', enhanced=True),
    SMAlegAATr(PrisonersDilemma(), 0, enhanced=False),
    QAlegAATr(PrisonersDilemma(), 0, enhanced=False),
    RawO(PrisonersDilemma(), 0, enhanced=False)
]
population_selection = [
    DQNAgent(PrisonersDilemma(), 1),
    MADQN(PrisonersDilemma(), 1),
    RDQN(PrisonersDilemma(), 1),
    AleqgAATr(PrisonersDilemma(), 1),
    RAlegAATr(PrisonersDilemma(), 1),
    SOAleqgAATr(PrisonersDilemma(), 1),
    AlegAATr(PrisonersDilemma(), 1, lmbda=0.0, ml_model_type='knn', enhanced=True),
    SMAlegAATr(PrisonersDilemma(), 1, enhanced=False),
    QAlegAATr(PrisonersDilemma(), 1, enhanced=False),
    RawO(PrisonersDilemma(), 1, enhanced=False)
]
population = [population_selection[i] for i in range(N_AGENTS)]
population_types = [type(alg) for alg in population]


# Used for ensuring that the agent frequencies compose a valid probability distribution
def convert_to_probs(values: List[float]) -> List[float]:
    shift_values = values - np.min(values)
    exp_values = np.exp(shift_values)
    probabilities = exp_values / np.sum(exp_values)

    return list(probabilities)


for iteration in range(N_ITERATIONS):
    # Progress report
    curr_iteration = iteration + 1
    if curr_iteration != 0 and progress_percentage_chunk != 0 and curr_iteration % progress_percentage_chunk == 0:
        print(f'{100 * (curr_iteration / N_ITERATIONS)}%')

    # Rewards for this iteration
    agent_rewards, population_rewards = [0] * N_AGENTS, []

    # Test each algorithm against every algorithm in the population (allows algorithms to re-enter the population)
    for i, alg in enumerate(algorithms):
        for opponent in population:
            # Values needed for the simulation - copy each agent to make sure none of the parameters get messed up
            agent = deepcopy(alg)
            opp = deepcopy(opponent)
            opp.name = f'{agent.name}2'
            players = [agent, opp]
            player_indices = [0, 1]

            # Run the simulation, extract the rewards
            final_rewards = run_with_specified_agents(players, player_indices, N_ROUNDS)

            # Update the agent's reward
            agent_reward = final_rewards[0]
            agent_rewards[i] += agent_reward

            # If the agent is in the population, update the population rewards
            if type(agent) in population_types:
                population_rewards.append(final_rewards[0])

    # Update each algorithm's frequency/probability of being added to the population
    avg_pop_reward = sum(population_rewards) / len(population_rewards)
    for i in range(len(agent_frequencies)):
        avg_agent_reward = agent_rewards[i] / len(population)
        freq_update = agent_frequencies[i] * (avg_agent_reward - avg_pop_reward)
        agent_frequencies[i] += freq_update
    agent_frequencies = convert_to_probs(agent_frequencies)  # Ensure the frequencies are a valid distribution
    population_indices = np.random.choice(N_AGENTS, N_AGENTS, p=agent_frequencies)
    population = [population_selection[i] for i in population_indices]
    population_types = [type(alg) for alg in population]

    # Status update on the population
    print([alg.name for alg in population])
