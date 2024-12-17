from agents.dqn import DQNAgent
from agents.alegaatr import AlegAATr
from agents.aleqgaatr import AleqgAATr
from agents.generator import BullyPunish
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
import os

N_EPOCHS = 30
N_ROUNDS = 30

# names = ['DQN', 'MADQN', 'RDQN', 'AleqgAATr', 'RAlegAATr', 'SOAleqgAATr', 'AlegAATr', 'SMAlegAATr', 'QAlegAATr',
#          'RawO', 'PPO', 'RAAT']
#
# # Reset any existing simulation files (opening a file in write mode will truncate it)
# for file in os.listdir('../simulations/adaptability_results/'):
#     name = file.split('_')[0]
#     if name in names:
#         with open(f'../simulations/adaptability_results/{file}', 'w', newline='') as _:
#             pass
#
# # Run the training process
# for epoch in range(N_EPOCHS):
#     print(f'Epoch {epoch + 1}')
#     player_idx = 1
#     opp_idx = 0
#
#     game = PrisonersDilemma()
#
#     list_of_opponents = []
#     list_of_opponents.append((BullyPunish(game, opp_idx), 'bullypunish'))
#     list_of_opponents.append((None, 'selfplay'))
#
#     for opponent, opponent_label in list_of_opponents:
#         agents_to_test = []
#         agents_to_test.append(DQNAgent(PrisonersDilemma(), player_idx))
#         agents_to_test.append(MADQN(PrisonersDilemma(), player_idx))
#         agents_to_test.append(RDQN(PrisonersDilemma(), player_idx))
#         agents_to_test.append(AleqgAATr(PrisonersDilemma(), player_idx))
#         agents_to_test.append(RAlegAATr(PrisonersDilemma(), player_idx))
#         agents_to_test.append(SOAleqgAATr(PrisonersDilemma(), player_idx))
#         agents_to_test.append(AlegAATr(PrisonersDilemma(), player_idx, lmbda=0.0, ml_model_type='knn',
#                                        enhanced=True))
#         agents_to_test.append(SMAlegAATr(PrisonersDilemma(), player_idx))
#         agents_to_test.append(QAlegAATr(PrisonersDilemma(), player_idx))
#         agents_to_test.append(RawO(PrisonersDilemma(), player_idx))
#         agents_to_test.append(PPO(PrisonersDilemma(), player_idx))
#         agents_to_test.append(RAAT(PrisonersDilemma(), player_idx))
#
#         self_play_agents = []
#         self_play_agents.append(DQNAgent(PrisonersDilemma(), opp_idx))
#         self_play_agents.append(MADQN(PrisonersDilemma(), opp_idx))
#         self_play_agents.append(RDQN(PrisonersDilemma(), opp_idx))
#         self_play_agents.append(AleqgAATr(PrisonersDilemma(), opp_idx))
#         self_play_agents.append(RAlegAATr(PrisonersDilemma(), opp_idx))
#         self_play_agents.append(SOAleqgAATr(PrisonersDilemma(), opp_idx))
#         self_play_agents.append(AlegAATr(PrisonersDilemma(), opp_idx, lmbda=0.0, ml_model_type='knn', enhanced=True))
#         self_play_agents.append(SMAlegAATr(PrisonersDilemma(), opp_idx))
#         self_play_agents.append(QAlegAATr(PrisonersDilemma(), opp_idx))
#         self_play_agents.append(RawO(PrisonersDilemma(), opp_idx))
#         self_play_agents.append(PPO(PrisonersDilemma(), opp_idx))
#         self_play_agents.append(RAAT(PrisonersDilemma(), opp_idx))
#
#         for i, agent_to_test in enumerate(agents_to_test):
#             if opponent_label == 'selfplay':
#                 assert opponent is None
#                 opp = self_play_agents[i]
#                 assert isinstance(opp, type(agent_to_test))
#                 opp.name = f'{opp.name}copy'
#             else:
#                 opp = deepcopy(opponent)
#             players = [opp, agent_to_test]
#             player_indices = [opp_idx, player_idx]
#             sim_label = f'{agent_to_test.name}_{opponent_label}_r={N_ROUNDS}'
#             run_with_specified_agents(players, player_indices, N_ROUNDS,
#                                       results_file=f'../simulations/adaptability_results/{sim_label}.csv')

# Run the training process
for epoch in range(N_EPOCHS):
    print(f'Epoch {epoch + 1}')
    player_idx = 1
    opp_idx = 0

    cooperators = [RawO(PrisonersDilemma(), opp_idx),
                   RAAT(PrisonersDilemma(), opp_idx),
                   QAlegAATr(PrisonersDilemma(), opp_idx),
                   AlegAATr(PrisonersDilemma(), opp_idx, lmbda=0.0, ml_model_type='knn', enhanced=True)]

    for opponent in cooperators:
        agents_to_test = []
        agents_to_test.append(DQNAgent(PrisonersDilemma(), player_idx))
        agents_to_test.append(MADQN(PrisonersDilemma(), player_idx))
        agents_to_test.append(RDQN(PrisonersDilemma(), player_idx))
        agents_to_test.append(AleqgAATr(PrisonersDilemma(), player_idx))
        agents_to_test.append(RAlegAATr(PrisonersDilemma(), player_idx))
        agents_to_test.append(SOAleqgAATr(PrisonersDilemma(), player_idx))
        agents_to_test.append(AlegAATr(PrisonersDilemma(), player_idx, lmbda=0.0, ml_model_type='knn',
                                       enhanced=True))
        agents_to_test.append(SMAlegAATr(PrisonersDilemma(), player_idx))
        agents_to_test.append(QAlegAATr(PrisonersDilemma(), player_idx))
        agents_to_test.append(RawO(PrisonersDilemma(), player_idx))
        agents_to_test.append(PPO(PrisonersDilemma(), player_idx))
        agents_to_test.append(RAAT(PrisonersDilemma(), player_idx))

        for agent_to_test in agents_to_test:
            if isinstance(opponent, type(agent_to_test)):
                continue
            opp = deepcopy(opponent)
            opp.name = f'{opp.name}copy'
            players = [opp, agent_to_test]
            player_indices = [opp_idx, player_idx]
            sim_label = f'{agent_to_test.name}_coop_r={N_ROUNDS}'
            run_with_specified_agents(players, player_indices, N_ROUNDS,
                                      results_file=f'../simulations/adaptability_results/{sim_label}.csv')
