from agents.ppo import PPO
from agents.spp import SPP
from agents.bbl import BBL
from agents.eee import EEE
from game.prisoners_dilemma import PrisonersDilemma
from game.main import run_with_specified_agents


N_EPOCHS = 500
N_ROUNDS = [20, 30, 40, 50]
n_training_iterations = N_EPOCHS * len(N_ROUNDS)
progress_percentage_chunk = int(0.05 * n_training_iterations)
curr_iteration = 0
print(n_training_iterations, progress_percentage_chunk)

ppo = PPO(PrisonersDilemma(), 1, train_network=True)

# Run the training process
for epoch in range(N_EPOCHS):
    print(f'Epoch {epoch + 1}')
    player_idx = 1
    opp_idx = 0

    for n_rounds in N_ROUNDS:
        if curr_iteration != 0 and curr_iteration % progress_percentage_chunk == 0:
            print(f'{100 * (curr_iteration / n_training_iterations)}%')

        game = PrisonersDilemma()

        list_of_opponents = []
        list_of_opponents.append(SPP('SPP', game, opp_idx))
        list_of_opponents.append(BBL('BBL', game, opp_idx))
        list_of_opponents.append(EEE('EEE', game, opp_idx))

        for opponent in list_of_opponents:
            players = [opponent, ppo]
            player_indices = [opp_idx, player_idx]
            run_with_specified_agents(players, player_indices, n_rounds)

            ppo.reset()

        curr_iteration += 1

    ppo.train()
    ppo.clear_buffer()
