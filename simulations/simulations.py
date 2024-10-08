from agents.bbl import BBL
from agents.dqn import DQNAgent
from agents.eee import EEE
from agents.alegaatr import AlegAATr
from agents.aleqgaatr import AleqgAATr
from agents.generator import Bullied, BullyPunish, CFR, Coop, CoopPunish, Minimax
from agents.madqn import MADQN
from agents.prisoners_dilemma_specific_agents import CoopOrGreedy, GreedyUntilNegative, Random, RoundNum, RoundNum2
from agents.ralegaatr import RAlegAATr
from agents.rdqn import RDQN
from agents.smalegaatr import SMAlegAATr
from agents.soaleqgaatr import SOAleqgAATr
from agents.spp import SPP
from copy import deepcopy
from game.main import run_with_specified_agents
from game.prisoners_dilemma import PrisonersDilemma
import os


N_EPOCHS = 5
N_ROUNDS = [10, 20, 30, 40, 50, 60]
n_training_iterations = N_EPOCHS * len(N_ROUNDS)
progress_percentage_chunk = int(0.05 * n_training_iterations)
curr_iteration = 0
print(n_training_iterations, progress_percentage_chunk)


names = ['SMAlegAATr']

# Reset any existing simulation files (opening a file in write mode will truncate it)
for file in os.listdir('../simulations/results/'):
    name = file.split('_')[0]
    if name in names:
        # if True:
        with open(f'../simulations/results/{file}', 'w', newline='') as _:
            pass

# Run the training process
for epoch in range(N_EPOCHS):
    print(f'Epoch {epoch + 1}')
    player_idx = 1
    opp_idx = 0

    for n_rounds in N_ROUNDS:
        if curr_iteration != 0 and progress_percentage_chunk != 0 and curr_iteration % progress_percentage_chunk == 0:
            print(f'{100 * (curr_iteration / n_training_iterations)}%')

        game = PrisonersDilemma()

        list_of_opponents = []
        list_of_opponents.append((Bullied(game, opp_idx), 'bullied'))
        list_of_opponents.append((BullyPunish(game, opp_idx), 'bullypunish'))
        list_of_opponents.append((CFR(game, opp_idx), 'cfr'))
        list_of_opponents.append((Coop(game, opp_idx), 'coop'))
        list_of_opponents.append((CoopPunish(game, opp_idx), 'cooppunish'))
        list_of_opponents.append((Minimax(game, opp_idx), 'minimax'))
        list_of_opponents.append((CoopOrGreedy('CoopOrGreedy', opp_idx), 'coopgreedy'))
        list_of_opponents.append((GreedyUntilNegative('GreedyUntilNegative', opp_idx), 'greedyneg'))
        list_of_opponents.append((Random('Random', opp_idx), 'random'))
        list_of_opponents.append((RoundNum('RoundNum', opp_idx), 'roundnum'))
        list_of_opponents.append((RoundNum2('RoundNum2', opp_idx), 'roundnum2'))
        list_of_opponents.append((BBL('BBL', game, opp_idx), 'bbl'))
        list_of_opponents.append((EEE('EEE', game, opp_idx), 'eee'))
        list_of_opponents.append((SPP('SPP', game, opp_idx), 'spp'))
        list_of_opponents.append((None, 'selfplay'))

        for opponent, opponent_label in list_of_opponents:
            agents_to_test = []
            # agents_to_test.append(DQNAgent(PrisonersDilemma(), player_idx))
            # agents_to_test.append(MADQN(PrisonersDilemma(), player_idx))
            # agents_to_test.append(RDQN(PrisonersDilemma(), player_idx))
            # agents_to_test.append(AleqgAATr(PrisonersDilemma(), player_idx))
            # agents_to_test.append(RAlegAATr(PrisonersDilemma(), player_idx))
            # agents_to_test.append(SOAleqgAATr(PrisonersDilemma(), player_idx))
            # agents_to_test.append(AlegAATr(PrisonersDilemma(), player_idx, lmbda=0.0, ml_model_type='knn', enhanced=True))
            agents_to_test.append(SMAlegAATr(PrisonersDilemma(), player_idx, enhanced=True))

            for agent_to_test in agents_to_test:
                if opponent_label == 'selfplay':
                    assert opponent is None
                    opp = deepcopy(agent_to_test)
                    opp.player = opp_idx
                    opp.name = f'{opp.name}copy'
                else:
                    opp = deepcopy(opponent)
                players = [opp, agent_to_test]
                player_indices = [opp_idx, player_idx]
                sim_label = f'{agent_to_test.name}_{opponent_label}_r={n_rounds}'
                run_with_specified_agents(players, player_indices, n_rounds,
                                          results_file=f'../simulations/results/{sim_label}.csv',
                                          generator_file=f'../simulations/generator_usage/{sim_label}.csv'
                                          # vector_file=f'../simulations/vectors/{sim_label}.csv'
                                          )

        curr_iteration += 1
