from aat.train_generators import N_EPOCHS, N_ROUNDS
from agents.aleqgaatr import AleqgAATr
from agents.generator import Bullied, BullyPunish, CFR, Coop, CoopPunish, Minimax
from game.prisoners_dilemma import PrisonersDilemma
from game.main import run_with_specified_agents


n_training_iterations = N_EPOCHS * len(N_ROUNDS)
progress_percentage_chunk = int(0.05 * n_training_iterations)
curr_iteration = 0
print(n_training_iterations, progress_percentage_chunk)

aleqgaatr = AleqgAATr(PrisonersDilemma(), 1, train_network=True)

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
        list_of_opponents.append(Bullied(game, opp_idx))
        list_of_opponents.append(BullyPunish(game, opp_idx))
        list_of_opponents.append(CFR(game, opp_idx))
        list_of_opponents.append(Coop(game, opp_idx))
        list_of_opponents.append(CoopPunish(game, opp_idx))
        list_of_opponents.append(Minimax(game, opp_idx))

        for opponent in list_of_opponents:
            players = [opponent, aleqgaatr]
            player_indices = [opp_idx, player_idx]
            run_with_specified_agents(players, player_indices, n_rounds)

            aleqgaatr.reset()

        curr_iteration += 1

    aleqgaatr.train()
    aleqgaatr.update_epsilon()
    aleqgaatr.update_networks()
    aleqgaatr.clear_buffer()
