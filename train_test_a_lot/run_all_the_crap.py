from agents.alegaatr import AlegAATr
from agents.generator_pool import GeneratorPool
from agents.spp import SPP
from agents.bbl import BBL
from agents.eee import EEE
from copy import deepcopy
from game.main import run_with_specified_agents
from game.prisoners_dilemma import PrisonersDilemma
import numpy as np
import os
from simple_rl.agents.AgentClass import Agent
from simple_rl.mdp.markov_game.MarkovGameMDPClass import MarkovGameMDP
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from aat.train_rawo import train_raw
from aat.train_qalegaatr import train_qalegaatr
from aat.train_raat import train_raat
from agents.rawo import RawO
from agents.qalegaatr import QAlegAATr
from agents.raat import RAAT
from agents.train_dqn import train_dqn
from agents.train_ralegaatr import train_ralegaatr
from agents.train_aleqgaatr import train_aleqgaatr
from simulations.adaptability_sims import adaptability


# Agent that just randomly (uniform) chooses a generator to use
class UniformSelector(Agent):
    def __init__(self, game: MarkovGameMDP, player: int, check_assumptions: bool = False,
                 no_baseline: bool = False) -> None:
        Agent.__init__(self, name='UniformSelector', actions=[])
        self.player = player
        self.generator_pool = GeneratorPool('pool', game, player, check_assumptions=check_assumptions,
                                            no_baseline_labels=no_baseline)
        self.check_assumptions = check_assumptions
        self.generator_indices = [i for i in range(len(self.generator_pool.generators))]
        self.generator_to_use_idx = None

    def store_terminal_state(self, state, reward) -> None:
        if self.check_assumptions:
            self.generator_pool.store_terminal_state(state, reward, self.generator_to_use_idx)

    def record_final_results(self, state, agent_reward) -> None:
        if self.check_assumptions:
            self.generator_pool.train_aat(state, agent_reward, self.generator_to_use_idx)

    def act(self, state, reward, round_num):
        # Get the actions of every generator
        generator_to_token_allocs = self.generator_pool.act(state, reward, round_num)

        # Randomly (uniform) choose a generator to use
        self.generator_to_use_idx = np.random.choice(self.generator_indices)

        token_allocations = generator_to_token_allocs[self.generator_to_use_idx]

        return token_allocations


# Agent that favors generators that have been used more recently
class FavorMoreRecent(Agent):
    def __init__(self, game: MarkovGameMDP, player: int, check_assumptions: bool = False,
                 no_baseline: bool = False) -> None:
        Agent.__init__(self, name='FavorMoreRecent', actions=[])
        self.player = player
        self.generator_pool = GeneratorPool('pool', game, player, check_assumptions=check_assumptions,
                                            no_baseline_labels=no_baseline)
        self.check_assumptions = check_assumptions
        self.generator_indices = [i for i in range(len(self.generator_pool.generators))]
        self.generator_to_use_idx, self.prev_generator_idx = None, None
        self.n_rounds_since_last_use = {}
        self.max_in_a_row = 5
        self.n_rounds_used = 0

    def store_terminal_state(self, state, reward) -> None:
        if self.check_assumptions:
            self.generator_pool.store_terminal_state(state, reward, self.generator_to_use_idx)

    def record_final_results(self, state, agent_reward) -> None:
        if self.check_assumptions:
            self.generator_pool.train_aat(state, agent_reward, self.generator_to_use_idx)

    def act(self, state, reward, round_num):
        # Get the actions of every generator
        generator_to_token_allocs = self.generator_pool.act(state, reward, round_num)

        # Randomly choose a generator, but favor those that have been used most recently
        rounds_since_used = [1 / self.n_rounds_since_last_use.get(i, 1) for i in self.generator_indices]
        if self.prev_generator_idx is not None and self.prev_generator_idx == self.generator_to_use_idx and \
                self.n_rounds_used >= self.max_in_a_row:
            rounds_since_used[self.generator_to_use_idx] = 0
            self.n_rounds_used = 0
        sum_val = sum(rounds_since_used)

        probabilities = [x / sum_val for x in rounds_since_used]
        self.prev_generator_idx = self.generator_to_use_idx
        self.generator_to_use_idx = np.random.choice(self.generator_indices, p=probabilities)

        # Update the number of rounds since each generator was used
        for i in self.generator_indices:
            self.n_rounds_since_last_use[i] = (
                    self.n_rounds_since_last_use.get(i, 1) + 1) if i != self.generator_to_use_idx else 1

        self.n_rounds_used += 1

        token_allocations = generator_to_token_allocs[self.generator_to_use_idx]

        return token_allocations


# Reset any existing simulation files (opening a file in write mode will truncate it)
for file in os.listdir('../simulations/adaptability_results/'):
    with open(f'../simulations/adaptability_results/{file}', 'w', newline='') as _:
        pass


N_TRAIN_TEST_RUNS = 30

for run_num in range(N_TRAIN_TEST_RUNS):
    print(f'RUN NUM = {run_num + 1}')
    N_EPOCHS = 10
    N_ROUNDS = [20, 30, 40, 50]

    print('Training REGAETune agents...')

    # Train AlegAATr
    NO_BASELINE = False

    for file in os.listdir('../aat/training_data/'):
        if (NO_BASELINE and 'sin_c' in file) or (not NO_BASELINE and 'sin_c' not in file):
            with open(f'../aat/training_data/{file}', 'w', newline='') as _:
                pass

    for epoch in range(N_EPOCHS):
        print(f'Epoch {epoch + 1}')
        player_idx = 1
        opp_idx = 0

        for n_rounds in N_ROUNDS:
            game = PrisonersDilemma()

            list_of_opponents = []
            list_of_opponents.append(SPP('SPP', game, opp_idx))
            list_of_opponents.append(BBL('BBL', game, opp_idx))
            list_of_opponents.append(EEE('EEE', game, opp_idx))

            for opponent in list_of_opponents:
                agents_to_train_on = []
                agents_to_train_on.append(UniformSelector(game, player_idx, check_assumptions=True,
                                                          no_baseline=NO_BASELINE))
                agents_to_train_on.append(FavorMoreRecent(game, player_idx, check_assumptions=True,
                                                          no_baseline=NO_BASELINE))

                for agent_to_train_on in agents_to_train_on:
                    players = [deepcopy(opponent), agent_to_train_on]
                    player_indices = [opp_idx, player_idx]
                    run_with_specified_agents(players, player_indices, n_rounds)

    generator_to_alignment_vectors, generator_to_correction_terms = {}, {}
    training_data_folder = '../aat/training_data/'
    enhanced = False
    adjustment = '_enh' if enhanced else ''

    for file in os.listdir(training_data_folder):
        if (enhanced and '_enh' not in file) or (not enhanced and '_enh' in file) or 'sin_c' in file:
            continue

        generator_idx = file.split('_')[1]
        data = np.genfromtxt(f'{training_data_folder}{file}', delimiter=',', skip_header=0)
        if data.shape[0] == 0:
            continue
        is_alignment_vectors = 'vectors' in file
        map_to_add_to = generator_to_alignment_vectors if is_alignment_vectors else generator_to_correction_terms
        map_to_add_to[generator_idx] = data

    for generator_idx, vectors in generator_to_alignment_vectors.items():
        correction_terms = generator_to_correction_terms[generator_idx]

        assert len(vectors) == len(correction_terms)

    for generator_idx, x in generator_to_alignment_vectors.items():
        y = generator_to_correction_terms[generator_idx]

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)

        print(f'X and Y data for generator {generator_idx}')
        print('X train shape: ' + str(x_scaled.shape))
        print('Y train shape: ' + str(y.shape))

        k_values, cv_scores = range(1, int(len(x_scaled) ** 0.5) + 1), []
        for k in k_values:
            knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
            scores = cross_val_score(knn, x_scaled, y, cv=5, scoring='neg_mean_squared_error')
            cv_scores.append(scores.mean())
        n_neighbors = k_values[np.argmax(cv_scores)]

        knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance')
        knn.fit(x_scaled, y)

        with open(f'../aat/knn_models/generator_{generator_idx}_knn{adjustment}.pickle', 'wb') as f:
            pickle.dump(knn, f)

        with open(f'../aat/knn_models/generator_{generator_idx}_scaler{adjustment}.pickle', 'wb') as f:
            pickle.dump(scaler, f)

        print(f'Best MSE: {-cv_scores[np.argmax(cv_scores)]}')
        print(f'Best R-squared: {r2_score(y, knn.predict(x_scaled))}')
        print(f'N neighbors: {n_neighbors}\n')

    for file in os.listdir('../aat/training_data/'):
        if (NO_BASELINE and 'sin_c' in file) or (not NO_BASELINE and 'sin_c' not in file):
            with open(f'../aat/training_data/{file}', 'w', newline='') as _:
                pass

    for epoch in range(N_EPOCHS):
        print(f'Epoch {epoch + 1}')
        player_idx = 1
        opp_idx = 0

        for n_rounds in N_ROUNDS:
            game = PrisonersDilemma()

            list_of_opponents = []
            list_of_opponents.append(SPP('SPP', game, opp_idx))
            list_of_opponents.append(BBL('BBL', game, opp_idx))
            list_of_opponents.append(EEE('EEE', game, opp_idx))

            for opponent in list_of_opponents:
                agents_to_train_on = []
                agents_to_train_on.append(AlegAATr(game, player_idx, lmbda=0.0, ml_model_type='knn', train=True))

                for agent_to_train_on in agents_to_train_on:
                    players = [deepcopy(opponent), agent_to_train_on]
                    player_indices = [opp_idx, player_idx]
                    run_with_specified_agents(players, player_indices, n_rounds)

    generator_to_alignment_vectors, generator_to_correction_terms = {}, {}
    training_data_folder = '../aat/training_data/'
    enhanced = True
    adjustment = '_enh' if enhanced else ''

    for file in os.listdir(training_data_folder):
        if (enhanced and '_enh' not in file) or (not enhanced and '_enh' in file) or 'sin_c' in file:
            continue

        generator_idx = file.split('_')[1]
        data = np.genfromtxt(f'{training_data_folder}{file}', delimiter=',', skip_header=0)
        if data.shape[0] == 0:
            continue
        is_alignment_vectors = 'vectors' in file
        map_to_add_to = generator_to_alignment_vectors if is_alignment_vectors else generator_to_correction_terms
        map_to_add_to[generator_idx] = data

    for generator_idx, vectors in generator_to_alignment_vectors.items():
        correction_terms = generator_to_correction_terms[generator_idx]

        assert len(vectors) == len(correction_terms)

    for generator_idx, x in generator_to_alignment_vectors.items():
        y = generator_to_correction_terms[generator_idx]

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)

        print(f'X and Y data for generator {generator_idx}')
        print('X train shape: ' + str(x_scaled.shape))
        print('Y train shape: ' + str(y.shape))

        k_values, cv_scores = range(1, int(len(x_scaled) ** 0.5) + 1), []
        for k in k_values:
            knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
            scores = cross_val_score(knn, x_scaled, y, cv=5, scoring='neg_mean_squared_error')
            cv_scores.append(scores.mean())
        n_neighbors = k_values[np.argmax(cv_scores)]

        knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance')
        knn.fit(x_scaled, y)

        with open(f'../aat/knn_models/generator_{generator_idx}_knn{adjustment}.pickle', 'wb') as f:
            pickle.dump(knn, f)

        with open(f'../aat/knn_models/generator_{generator_idx}_scaler{adjustment}.pickle', 'wb') as f:
            pickle.dump(scaler, f)

        # Print metrics and best number of neighbors
        print(f'Best MSE: {-cv_scores[np.argmax(cv_scores)]}')
        print(f'Best R-squared: {r2_score(y, knn.predict(x_scaled))}')
        print(f'N neighbors: {n_neighbors}\n')

    # Train RawR, RRawAAT, RAAT
    NO_BASELINE = True

    for file in os.listdir('../aat/training_data/'):
        if (NO_BASELINE and 'sin_c' in file) or (not NO_BASELINE and 'sin_c' not in file):
            with open(f'../aat/training_data/{file}', 'w', newline='') as _:
                pass

    for epoch in range(N_EPOCHS):
        print(f'Epoch {epoch + 1}')
        player_idx = 1
        opp_idx = 0

        for n_rounds in N_ROUNDS:
            game = PrisonersDilemma()

            list_of_opponents = []
            list_of_opponents.append(SPP('SPP', game, opp_idx))
            list_of_opponents.append(BBL('BBL', game, opp_idx))
            list_of_opponents.append(EEE('EEE', game, opp_idx))

            for opponent in list_of_opponents:
                agents_to_train_on = []
                agents_to_train_on.append(UniformSelector(game, player_idx, check_assumptions=True,
                                                          no_baseline=NO_BASELINE))
                agents_to_train_on.append(FavorMoreRecent(game, player_idx, check_assumptions=True,
                                                          no_baseline=NO_BASELINE))

                for agent_to_train_on in agents_to_train_on:
                    players = [deepcopy(opponent), agent_to_train_on]
                    player_indices = [opp_idx, player_idx]
                    run_with_specified_agents(players, player_indices, n_rounds)

    train_raw(ENHANCED=False)
    train_qalegaatr(ENHANCED=False)
    train_raat(ENHANCED=False)

    for file in os.listdir('../aat/training_data/'):
        if (NO_BASELINE and 'sin_c' in file) or (not NO_BASELINE and 'sin_c' not in file):
            with open(f'../aat/training_data/{file}', 'w', newline='') as _:
                pass

    for epoch in range(N_EPOCHS):
        print(f'Epoch {epoch + 1}')
        player_idx = 1
        opp_idx = 0

        for n_rounds in N_ROUNDS:
            game = PrisonersDilemma()

            list_of_opponents = []
            list_of_opponents.append(SPP('SPP', game, opp_idx))
            list_of_opponents.append(BBL('BBL', game, opp_idx))
            list_of_opponents.append(EEE('EEE', game, opp_idx))

            for opponent in list_of_opponents:
                agents_to_train_on = []
                agents_to_train_on.append(RawO(game, player_idx, train=True))

                for agent_to_train_on in agents_to_train_on:
                    players = [deepcopy(opponent), agent_to_train_on]
                    player_indices = [opp_idx, player_idx]
                    run_with_specified_agents(players, player_indices, n_rounds)

    train_raw(ENHANCED=True)

    for file in os.listdir('../aat/training_data/'):
        if (NO_BASELINE and 'sin_c' in file) or (not NO_BASELINE and 'sin_c' not in file):
            with open(f'../aat/training_data/{file}', 'w', newline='') as _:
                pass

    for epoch in range(N_EPOCHS):
        print(f'Epoch {epoch + 1}')
        player_idx = 1
        opp_idx = 0

        for n_rounds in N_ROUNDS:
            game = PrisonersDilemma()

            list_of_opponents = []
            list_of_opponents.append(SPP('SPP', game, opp_idx))
            list_of_opponents.append(BBL('BBL', game, opp_idx))
            list_of_opponents.append(EEE('EEE', game, opp_idx))

            for opponent in list_of_opponents:
                agents_to_train_on = []
                agents_to_train_on.append(QAlegAATr(game, player_idx, train=True))

                for agent_to_train_on in agents_to_train_on:
                    players = [deepcopy(opponent), agent_to_train_on]
                    player_indices = [opp_idx, player_idx]
                    run_with_specified_agents(players, player_indices, n_rounds)

    train_qalegaatr(ENHANCED=True)

    for file in os.listdir('../aat/training_data/'):
        if (NO_BASELINE and 'sin_c' in file) or (not NO_BASELINE and 'sin_c' not in file):
            with open(f'../aat/training_data/{file}', 'w', newline='') as _:
                pass

    for epoch in range(N_EPOCHS):
        print(f'Epoch {epoch + 1}')
        player_idx = 1
        opp_idx = 0

        for n_rounds in N_ROUNDS:
            game = PrisonersDilemma()

            list_of_opponents = []
            list_of_opponents.append(SPP('SPP', game, opp_idx))
            list_of_opponents.append(BBL('BBL', game, opp_idx))
            list_of_opponents.append(EEE('EEE', game, opp_idx))

            for opponent in list_of_opponents:
                agents_to_train_on = []
                agents_to_train_on.append(RAAT(game, player_idx, train=True))

                for agent_to_train_on in agents_to_train_on:
                    players = [deepcopy(opponent), agent_to_train_on]
                    player_indices = [opp_idx, player_idx]
                    run_with_specified_agents(players, player_indices, n_rounds)

    train_raat(ENHANCED=True)

    # Train DQN, RAlegAATr, AleqgAATr
    print('Training EG agents...')
    train_dqn()
    train_ralegaatr()
    train_aleqgaatr()

    # Run the adaptability crap
    print('Generating new adaptability results...')
    adaptability(run_num)
