from agents.generator_pool import GeneratorPool
from game.prisoners_dilemma import ACTIONS
import keras
import numpy as np
import pickle
from simple_rl.agents.AgentClass import Agent
from simple_rl.mdp.markov_game.MarkovGameMDPClass import MarkovGameMDP
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model, Model


@keras.saving.register_keras_serializable()
class SingleGenModelRaw(Model):
    def __init__(self, state_dim: int) -> None:
        super(SingleGenModelRaw, self).__init__()
        self.state_dim = state_dim

        self.dense1 = Dense(self.state_dim, activation='relu')
        self.dense2 = Dense(32, activation='relu')
        self.output_layer = Dense(6, activation='linear')

    def get_config(self):
        return {'state_dim': self.state_dim}

    def call(self, state: np.array, return_transformed_state: bool = False) -> tf.Tensor:
        x = self.dense1(state)
        x = self.dense2(x)

        if return_transformed_state:
            return x

        return self.output_layer(x)


class RawO(Agent):
    def __init__(self, game: MarkovGameMDP, player: int, train: bool = False, enhanced: bool = False) -> None:
        Agent.__init__(self, name='RawO', actions=[])
        self.player = player
        self.generator_pool = GeneratorPool('pool', game, player, check_assumptions=True, no_baseline_labels=True)
        self.generator_indices = [i for i in range(len(self.generator_pool.generators))]
        self.generator_to_use_idx = None
        file_adj = '_enh' if enhanced else ''
        self.model = load_model(f'../aat/single_gen_model_raw/single_gen_model{file_adj}.keras')
        self.scaler = pickle.load(open(f'../aat/single_gen_model_raw/single_gen_scaler{file_adj}.pickle', 'rb'))
        self.train = train
        self.tracked_vector = None
        self.generators_used = set()
        self.state = None
        self.prev_reward = None

    def store_terminal_state(self, state, reward) -> None:
        self.generator_pool.store_terminal_state(state, reward, self.generator_to_use_idx)
        self.state = np.array([ACTIONS.index(state.actions[self.player]), ACTIONS.index(state.actions[1 - self.player]),
                               reward, self.prev_reward])

    def record_final_results(self, state, agent_reward) -> None:
        if self.train:
            self.generator_pool.train_aat(state, agent_reward, self.generator_to_use_idx, enhanced=True)
        # print(f'Generators used: {self.generators_used}')

    def act(self, state, reward, round_num):
        self.prev_reward = reward

        # Get the actions of every generator
        generator_to_token_allocs = self.generator_pool.act(state, reward, round_num)

        # State vector
        curr_state = self.state if self.state is not None else np.array([-1, -1, 0, 0])
        curr_state = self.scaler.transform(curr_state.reshape(1, -1))

        # Make predictions
        q_values = self.model(curr_state)
        self.generator_to_use_idx = np.argmax(q_values.numpy())
        self.tracked_vector = self.model(curr_state, return_transformed_state=True).numpy().reshape(-1, )
        self.generators_used.add(self.generator_to_use_idx)

        token_allocations = generator_to_token_allocs[self.generator_to_use_idx]

        return token_allocations
