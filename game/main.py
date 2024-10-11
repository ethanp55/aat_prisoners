from copy import deepcopy
import csv
from game.prisoners_dilemma import PrisonersDilemma
import numpy as np
from simple_rl.agents.AgentClass import Agent
from typing import List, Optional
from utils.utils import P1


def run_with_specified_agents(players: List[Agent], player_indices: List[int], n_rounds: int,
                              results_file: Optional[str] = None, generator_file: Optional[str] = None,
                              vector_file: Optional[str] = None) -> List[float]:
    assert len(players) == len(player_indices) == 2
    assert n_rounds > 0

    # Reset any generator usage data and/or vector data
    if generator_file is not None:
        with open(generator_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['round', 'generator'])
    if vector_file is not None:
        with open(vector_file, 'w', newline='') as _:
            pass

    game = PrisonersDilemma()
    rewards = [0, 0]

    for round_num in range(n_rounds):
        game.reset()
        state = deepcopy(game.get_init_state())
        action_map = dict()

        while not state.is_terminal():
            for i, agent in enumerate(players):
                player_idx = player_indices[i]
                agent_reward = rewards[player_idx]
                agent_action1, agent_action2 = agent.act(state, agent_reward, round_num)
                action_map[agent.name] = agent_action1 if player_idx == P1 else agent_action2

            updated_rewards_map, next_state = game.execute_agent_action(action_map)

            # print(action_map, updated_rewards_map)

            for i, agent in enumerate(players):
                player_idx = player_indices[i]
                rewards[player_idx] += updated_rewards_map[agent.name]

            state = next_state

            # Write any generator usage data and/or vectors
            if generator_file is not None:
                with open(generator_file, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([round_num, players[-1].generator_to_use_idx])
            if vector_file is not None:
                with open(vector_file, 'a', newline='') as file:
                    writer = csv.writer(file)
                    row = np.concatenate([np.array([players[-1].generator_to_use_idx]), players[-1].tracked_vector])
                    writer.writerow(np.squeeze(row))

        # Some algorithms need to store the terminal state
        for i, agent in enumerate(players):
            player_idx = player_indices[i]
            agent_reward = rewards[player_idx]
            try:
                agent.store_terminal_state(state, agent_reward)
            except:
                continue

    # Some algorithms need to store final results
    for i, agent in enumerate(players):
        player_idx = player_indices[i]
        agent_reward = rewards[player_idx]
        try:
            agent.record_final_results(state, agent_reward)
        except:
            continue

    # Save data
    if results_file is not None:
        with open(results_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(rewards)

    # Return final rewards (for any caller that needs them)
    return rewards
