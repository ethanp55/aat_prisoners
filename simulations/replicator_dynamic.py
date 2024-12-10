from agents.dqn import DQNAgent
from agents.alegaatr import AlegAATr
from agents.aleqgaatr import AleqgAATr
from agents.madqn import MADQN
from agents.ppo import PPO
from agents.qalegaatr import QAlegAATr
from agents.raat import RAAT
from agents.ralegaatr import RAlegAATr
from agents.rawo import RawO
from agents.rdqn import RDQN
from agents.smalegaatr import SMAlegAATr
from agents.soaleqgaatr import SOAleqgAATr
from copy import deepcopy
from game.main import run_with_specified_agents
from game.prisoners_dilemma import PrisonersDilemma
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import List


# Variables
N_ITERATIONS = 100
N_ROUNDS = 30
progress_percentage_chunk = int(0.05 * N_ITERATIONS)

# Frequencies and population details
algorithms = [
    DQNAgent(PrisonersDilemma(), 0),
    # MADQN(PrisonersDilemma(), 0),
    # RDQN(PrisonersDilemma(), 0),
    AleqgAATr(PrisonersDilemma(), 0),
    RAlegAATr(PrisonersDilemma(), 0),
    # SOAleqgAATr(PrisonersDilemma(), 0),
    AlegAATr(PrisonersDilemma(), 0, lmbda=0.0, ml_model_type='knn', enhanced=True),
    # SMAlegAATr(PrisonersDilemma(), 0, enhanced=False),
    QAlegAATr(PrisonersDilemma(), 0, enhanced=False),
    # RawO(PrisonersDilemma(), 0, enhanced=False),
    # PPO(PrisonersDilemma(), 0),
    # RAAT(PrisonersDilemma(), 0, enhanced=False)
]
population_selection = [
    DQNAgent(PrisonersDilemma(), 1),
    # MADQN(PrisonersDilemma(), 1),
    # RDQN(PrisonersDilemma(), 1),
    AleqgAATr(PrisonersDilemma(), 1),
    RAlegAATr(PrisonersDilemma(), 1),
    # SOAleqgAATr(PrisonersDilemma(), 1),
    AlegAATr(PrisonersDilemma(), 1, lmbda=0.0, ml_model_type='knn', enhanced=True),
    # SMAlegAATr(PrisonersDilemma(), 1, enhanced=False),
    QAlegAATr(PrisonersDilemma(), 1, enhanced=False),
    # RawO(PrisonersDilemma(), 1, enhanced=False),
    # PPO(PrisonersDilemma(), 1),
    # RAAT(PrisonersDilemma(), 1, enhanced=False)
]
N_AGENTS = len(algorithms)
agent_frequencies = [1 / N_AGENTS] * N_AGENTS
agent_representation_over_time = {agent.name: [1 / N_AGENTS] for agent in algorithms}

R = np.zeros((N_AGENTS, N_AGENTS))
for i, alg1 in enumerate(algorithms):
    print(f'Agent {i}')
    for j, alg2 in enumerate(population_selection):
        rewards_against_j = []
        for _ in range(1):
            agent = deepcopy(alg1)
            opp = deepcopy(alg2)
            opp.name = f'{agent.name}2'
            rewards = run_with_specified_agents([agent, opp], [0, 1], N_ROUNDS)
            rewards_against_j.append(rewards[0])
        R[i][j] = sum(rewards_against_j) / len(rewards_against_j)
min_r, max_r = R.min(), R.max()
R = (R - min_r) / (max_r - min_r)
# R = MinMaxScaler().fit_transform(R)
assert R.min() == 0 and R.max() == 1

for i in range(len(R)):
    print(R[i, :])

for iteration in range(N_ITERATIONS):
    fitnesses = []
    for i, alg in enumerate(algorithms):
        fitness = sum([agent_frequencies[j] * R[i][j] for j in range(len(algorithms))])
        fitnesses.append(fitness)

    avg_fitness = sum([agent_frequencies[j] * fitnesses[j] for j in range(len(algorithms))])
    agent_frequencies = [agent_frequencies[i] + agent_frequencies[i] * (fitnesses[i] - avg_fitness) for i in
                         range(len(algorithms))]
    assert round(sum(agent_frequencies), 3) == 1

    # Update data for plot
    for i, alg in enumerate(algorithms):
        new_proportion = agent_frequencies[i]
        agent_representation_over_time[alg.name] += [new_proportion]

# Plot agent representations over time
colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'lime', 'pink', 'yellow', 'brown', 'black']
name_conversions = {
    'DQN': 'DQN',
    'MADQN': 'MADQN',
    'RDQN': 'RDQN',
    'AleqgAATr': 'TRawAAT',
    'RAlegAATr': 'TAAT',
    'SOAleqgAATr': 'STRawAAT',
    'AlegAATr': 'AlegAATr',
    'SMAlegAATr': 'SRRawAAT',
    'QAlegAATr': 'RRawAAT',
    'RawO': 'RawR',
    'PPO': 'PPO',
    'RAAT': 'RAAT'
}
plt.figure(figsize=(10, 3))
plt.grid()
for i, agent in enumerate(agent_representation_over_time.keys()):
    proportions, color = agent_representation_over_time[agent], colors[i]
    plt.plot(proportions, label=name_conversions[agent], color=color)
plt.xlabel('Iteration', fontsize=18, fontweight='bold')
plt.ylabel('Proportion', fontsize=18, fontweight='bold')
plt.legend(loc='best', fontsize=10)
plt.savefig(f'../simulations/replicator_dynamic.png', bbox_inches='tight')
plt.clf()
