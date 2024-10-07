from agents.generator_pool import GeneratorPool
from collections import deque
from copy import deepcopy
from game.prisoners_dilemma import ACTIONS
import numpy as np
import pickle
import random
from simple_rl.agents.AgentClass import Agent
from simple_rl.mdp.markov_game.MarkovGameMDPClass import MarkovGameMDP
import tensorflow as tf
from tensorflow import keras
from typing import Tuple


class DynamicMinMaxScaler:
    def __init__(self, num_features: int) -> None:
        self.num_features = num_features
        self.min_vals = np.inf * np.ones(num_features)
        self.max_vals = -np.inf * np.ones(num_features)

    def update(self, state: np.array) -> None:
        self.min_vals = np.minimum(self.min_vals, state)
        self.max_vals = np.maximum(self.max_vals, state)

    def scale(self, state: np.array) -> np.array:
        return (state - self.min_vals) / (self.max_vals - self.min_vals + 0.00001)


class DQN(keras.Model):
    def __init__(self, aat_dim: int, state_dim: int, action_dim: int) -> None:
        super(DQN, self).__init__()
        self.aat_dim = aat_dim
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.dense_aat_1 = keras.layers.Dense(self.aat_dim, activation='relu')
        self.dense_aat_2 = keras.layers.Dense(self.state_dim, activation='relu')

        self.dense_state_1 = keras.layers.Dense(self.state_dim, activation='relu')
        self.dense_state_2 = keras.layers.Dense(self.state_dim, activation='relu')

        self.dense_combined_1 = keras.layers.Dense(self.state_dim, activation='relu')
        self.dense_combined_2 = keras.layers.Dense(32, activation='relu')

        self.output_layer = keras.layers.Dense(self.action_dim, activation='linear')

    def get_config(self):
        return {'state_dim': self.state_dim, 'action_dim': self.action_dim, 'aat_dim': self.aat_dim}

    def call(self, states: Tuple[np.array, np.array], return_transformed_state: bool = False) -> tf.Tensor:
        aat_state, state = states

        x_aat = self.dense_aat_1(aat_state)
        x_aat = x_aat + aat_state
        x_aat = self.dense_aat_2(x_aat)

        x_state = self.dense_state_1(state)
        x_state = x_state + state
        x_state = self.dense_state_2(x_state)

        x = x_aat + x_state
        x = x + self.dense_combined_1(x)
        x = self.dense_combined_2(x)

        if return_transformed_state:
            return x

        return self.output_layer(x)


class AleqgAATr(Agent):
    def __init__(self, game: MarkovGameMDP, player: int, learning_rate: float = 0.001, discount_factor: float = 0.9,
                 epsilon: float = 0.1, epsilon_decay: float = 0.99, replay_buffer_size: int = 50000,
                 batch_size: int = 64, train_network: bool = False) -> None:
        Agent.__init__(self, name='AleqgAATr', actions=[])
        self.player = player
        self.game = deepcopy(game)

        # Variables used for training and/or action selection
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.state, self.aat_state = None, None
        self.train_network = train_network
        self.prev_reward = 0
        self.best_loss = np.inf

        # Generators
        self.generator_pool = GeneratorPool('pool', game, player, check_assumptions=True)
        self.generator_indices = [i for i in range(len(self.generator_pool.generators))]
        self.generator_to_use_idx = None

        # DQN model and target model
        self.aat_dim = 3 + 4 + 3 + 4 + 3 + 2
        self.state_dim = 4
        self.action_dim = len(self.generator_indices)
        self.model = DQN(self.aat_dim, self.state_dim, self.action_dim)
        self.target_model = DQN(self.aat_dim, self.state_dim, self.action_dim)
        self.target_model.set_weights(self.model.get_weights())

        # Optimizer
        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

        # Replay buffer
        self.replay_buffer = deque(maxlen=replay_buffer_size)

        # Episode experiences (to add to the replay buffer)
        self.current_episode_experiences = []

        # State scaler
        self.scaler = DynamicMinMaxScaler(self.state_dim)

        # If we're not in training mode, load the saved/trained model
        if not self.train_network:
            self.load_network()

        self.tracked_vector = None
        self.generators_used = set()
        self.next_state, self.next_aat_state = None, None

    def _create_aat_vec(self) -> np.array:
        aat_vec = []
        for i in self.generator_indices:
            aat_vec.extend(self.generator_pool.assumptions(i))
        aat_vec = np.array(aat_vec).reshape(1, -1)

        return aat_vec

    def update_epsilon(self) -> None:
        self.epsilon *= self.epsilon_decay

    def reset(self) -> None:
        self.generator_to_use_idx = None
        self.prev_reward = 0
        self.next_state, self.next_aat_state = None, None
        self.generator_pool = GeneratorPool('pool', self.game, self.player, check_assumptions=True)

    def record_final_results(self, state, reward) -> None:
        # Add a new experience to the replay buffer and update the network weights (if there are enough experiences)
        if self.train_network:
            assert self.next_state is not None
            assert self.next_aat_state is not None
            increase = reward - self.prev_reward
            self.add_experience(self.generator_to_use_idx, increase, self.next_aat_state, self.next_state, True)

        # print(f'Generators used: {self.generators_used}')

    def store_terminal_state(self, state, reward) -> None:
        player_idx, opp_idx = self.player, 1 - self.player
        self.next_state = np.array([ACTIONS.index(state.actions[player_idx]), ACTIONS.index(state.actions[opp_idx]),
                                    reward, self.prev_reward])
        self.generator_pool.store_terminal_state(state, reward, self.generator_to_use_idx)
        self.next_aat_state = self._create_aat_vec()

    def act(self, state, reward, round_num):
        # Add a new experience to the replay buffer and update the network weights (if there are enough experiences)
        if self.train_network and self.next_state is not None:
            assert self.next_aat_state is not None
            increase = reward - self.prev_reward
            self.add_experience(self.generator_to_use_idx, increase, self.next_aat_state, self.next_state, False)
        self.prev_reward = reward

        # Get the actions of every generator
        generator_to_token_allocs = self.generator_pool.act(state, reward, round_num)

        self.state = self.next_state if self.next_state is not None else np.array([-1, -1, 0, 0])
        self.aat_state = self.next_aat_state if self.next_aat_state is not None else self._create_aat_vec()

        # Epsilon-greedy policy for generator selection (only consider exploring if we're in training mode)
        if self.train_network and np.random.rand() < self.epsilon:
            self.generator_to_use_idx = np.random.choice(self.action_dim)

        else:
            scaled_state = self.scaler.scale(self.state)
            q_values = self.model((self.aat_state, np.expand_dims(scaled_state, 0)))
            self.generator_to_use_idx = np.argmax(q_values.numpy())

            network_state = self.model((self.aat_state, np.expand_dims(scaled_state, 0)), return_transformed_state=True)
            self.tracked_vector = network_state.numpy().reshape(-1, )

        self.generators_used.add(self.generator_to_use_idx)

        token_allocations = generator_to_token_allocs[self.generator_to_use_idx]

        return token_allocations

    def update_networks(self) -> None:
        # Update target network weights periodically
        self.target_model.set_weights(self.model.get_weights())

    def train(self) -> None:
        for _ in range(100):
            # Sample a batch of experiences from the replay buffer
            batch = random.sample(self.replay_buffer, self.batch_size)
            batch_aat_states, batch_states, batch_actions, batch_rewards, batch_next_aat_states, batch_next_states, batch_dones = map(
                np.array, zip(*batch))
            batch_states, batch_next_states = \
                batch_states.reshape(self.batch_size, -1), batch_next_states.reshape(self.batch_size, -1)
            batch_aat_states, batch_next_aat_states = \
                batch_aat_states.reshape(self.batch_size, -1), batch_next_aat_states.reshape(self.batch_size, -1)

            # Q-value update
            next_q_values = self.target_model((batch_next_aat_states, batch_next_states))
            max_next_q_values = np.max(next_q_values.numpy(), axis=1)
            targets = batch_rewards + (1 - batch_dones) * self.discount_factor * max_next_q_values

            with tf.GradientTape() as tape:
                q_values = self.model((batch_aat_states, batch_states))
                selected_action_values = tf.reduce_sum(tf.one_hot(batch_actions, self.action_dim) * q_values, axis=1)
                loss = tf.keras.losses.MSE(targets, selected_action_values)

            loss_val = loss.numpy()
            if loss_val < self.best_loss:
                print(f'Loss improved from {self.best_loss} to {loss_val}')
                self.best_loss = loss_val
                self.save_network()

            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    def add_experience(self, action: int, reward: float, next_state_aat: np.array, next_state: np.array, done: bool):
        # Accumulate experiences over multiple time steps
        scaled_state = self.scaler.scale(self.state)
        scaled_next_state = self.scaler.scale(next_state)
        self.scaler.update(next_state)

        self.current_episode_experiences.append((self.aat_state, scaled_state, action, reward, next_state_aat,
                                                 scaled_next_state, done))

        # If the episode is done, add the accumulated experiences to the replay buffer
        if done:
            self.replay_buffer.extend(self.current_episode_experiences)
            self.current_episode_experiences = []

    def clear_buffer(self) -> None:
        self.replay_buffer.clear()
        self.current_episode_experiences = []

    def save_network(self) -> None:
        # Save the network and scaler
        self.model.save('../agents/aleqgaatr_model/model.keras')

        with open('../agents/aleqgaatr_model/scaler.pickle', 'wb') as f:
            pickle.dump(self.scaler, f)

    def load_network(self) -> None:
        # Load the network and scaler
        self.model = keras.models.load_model('../agents/aleqgaatr_model/model.keras')
        self.scaler = pickle.load(open('../agents/aleqgaatr_model/scaler.pickle', 'rb'))
