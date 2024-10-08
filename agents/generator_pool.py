from agents.generator import Bullied, BullyPunish, CFR, Coop, CoopPunish, Minimax
import csv
import fcntl
from game.prisoners_dilemma import ACTIONS
import numpy as np
from simple_rl.agents.AgentClass import Agent
from simple_rl.mdp.markov_game.MarkovGameMDPClass import MarkovGameMDP
from typing import List


class GeneratorPool(Agent):
    def __init__(self, name: str, game: MarkovGameMDP, player: int, check_assumptions: bool = False,
                 no_baseline_labels: bool = False) -> None:
        Agent.__init__(self, name=name, actions=[])
        self.player = player
        self.generators = []
        self.generators.append(Bullied(game, player, check_assumptions=check_assumptions))
        self.generators.append(BullyPunish(game, player, check_assumptions=check_assumptions))
        self.generators.append(CFR(game, player, check_assumptions=check_assumptions))
        self.generators.append(Coop(game, player, check_assumptions=check_assumptions))
        self.generators.append(CoopPunish(game, player, check_assumptions=check_assumptions))
        self.generators.append(Minimax(game, player, check_assumptions=check_assumptions))
        self.check_assumptions = check_assumptions
        self.generator_to_assumption_estimates = {}
        self.generators_to_states = {}
        self.no_baseline_labels = no_baseline_labels
        self.reward_history = []
        self.generator_just_used_idx = None
        self.prev_round_num = 0
        self.state = None

    def act(self, state, reward, round_num):
        generator_to_action = {}

        for i, generator in enumerate(self.generators):
            generator_to_action[i] = generator.act(state, reward, round_num)

        if self.generator_just_used_idx is not None:
            generator_just_used = self.generators[self.generator_just_used_idx]

            # Grab the assumption estimates, if we're tracking them
            if self.check_assumptions:
                assumps = self.assumptions(self.generator_just_used_idx)
                tup = (assumps, round_num, generator_just_used.baseline, None)

                self.generator_to_assumption_estimates[self.generator_just_used_idx] = \
                    self.generator_to_assumption_estimates.get(self.generator_just_used_idx, []) + [tup]

                if self.no_baseline_labels:
                    curr_state = self.state

                    self.generators_to_states[self.generator_just_used_idx] = self.generators_to_states.get(
                        self.generator_just_used_idx, []) + [curr_state]

        self.reward_history.append(reward)
        self.prev_round_num = round_num

        return generator_to_action

    def train_aat(self, state, reward, generator_just_used_idx: int, discount_factor: float = 0.9,
                  enhanced: bool = False) -> None:
        # Calculate assumption estimates for the final round
        self.store_terminal_state(state, reward, generator_just_used_idx)
        self.act(state, reward, self.prev_round_num + 1)

        # Calculate discounted rewards
        discounted_rewards, running_sum = [0] * (len(self.reward_history) - 1), 0
        for i in reversed(range(len(self.reward_history))):
            if i == 0:
                break
            reward = self.reward_history[i] - self.reward_history[i - 1]
            running_sum = reward + discount_factor * running_sum
            discounted_rewards[i - 1] = running_sum

        # Store the training data
        for generator_idx in self.generator_to_assumption_estimates.keys():
            assumptions_history = self.generator_to_assumption_estimates[generator_idx]
            states_history = self.generators_to_states.get(generator_idx, [])

            for i in range(len(assumptions_history)):
                assumption_estimates, round_num, baseline, game_state = assumptions_history[i]
                state = states_history[i] if self.no_baseline_labels else None
                assert round_num > 0

                discounted_reward = discounted_rewards[round_num - 1]
                correction_term = discounted_reward / baseline
                alignment_vector = assumption_estimates

                if self.no_baseline_labels:
                    # Store the alignment vector
                    adjustment = '_enh' if enhanced else ''
                    file_path = f'../aat/training_data/generator_{generator_idx}_sin_c_vectors{adjustment}.csv'

                    with open(file_path, 'a', newline='') as file:
                        fcntl.flock(file.fileno(), fcntl.LOCK_EX)  # Lock the file (for write safety)
                        writer = csv.writer(file)
                        writer.writerow(alignment_vector)
                        fcntl.flock(file.fileno(), fcntl.LOCK_UN)  # Unlock the file

                    # Store the state
                    assert state is not None
                    file_path = f'../aat/training_data/generator_{generator_idx}_sin_c_states{adjustment}.csv'
                    with open(file_path, 'a', newline='') as file:
                        fcntl.flock(file.fileno(), fcntl.LOCK_EX)  # Lock the file (for write safety)
                        writer = csv.writer(file)
                        writer.writerow(state)
                        fcntl.flock(file.fileno(), fcntl.LOCK_UN)  # Unlock the file

                    # Store the discounted reward
                    file_path = f'../aat/training_data/generator_{generator_idx}_sin_c_correction_terms{adjustment}.csv'
                    with open(file_path, 'a', newline='') as file:
                        fcntl.flock(file.fileno(), fcntl.LOCK_EX)  # Lock the file (for write safety)
                        writer = csv.writer(file)
                        writer.writerow([discounted_reward])
                        fcntl.flock(file.fileno(), fcntl.LOCK_UN)  # Unlock the file

                else:
                    # Store the alignment vector
                    adjustment = '_enh' if enhanced else ''
                    file_path = f'../aat/training_data/generator_{generator_idx}_vectors{adjustment}.csv'

                    with open(file_path, 'a', newline='') as file:
                        fcntl.flock(file.fileno(), fcntl.LOCK_EX)  # Lock the file (for write safety)
                        writer = csv.writer(file)
                        writer.writerow(alignment_vector)
                        fcntl.flock(file.fileno(), fcntl.LOCK_UN)  # Unlock the file

                    # Store the correction term
                    file_path = f'../aat/training_data/generator_{generator_idx}_correction_terms{adjustment}.csv'
                    with open(file_path, 'a', newline='') as file:
                        fcntl.flock(file.fileno(), fcntl.LOCK_EX)  # Lock the file (for write safety)
                        writer = csv.writer(file)
                        writer.writerow([correction_term])
                        fcntl.flock(file.fileno(), fcntl.LOCK_UN)  # Unlock the file

    def store_terminal_state(self, state, reward, generator_just_used_idx: int) -> None:
        if self.check_assumptions:
            for i, generator in enumerate(self.generators):
                was_used = i == generator_just_used_idx
                generator.check_assumptions(state, reward, was_used)
            self.generator_just_used_idx = generator_just_used_idx
            self.state = np.array([ACTIONS.index(state.actions[self.player]),
                                   ACTIONS.index(state.actions[1 - self.player]),
                                   reward,
                                   self.reward_history[-1]])

    def assumptions(self, generator_idx: int) -> List[float]:
        return self.generators[generator_idx].assumptions()
