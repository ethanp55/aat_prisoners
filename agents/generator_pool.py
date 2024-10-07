from agents.generator import Bullied, BullyPunish, CFR, Coop, CoopPunish, Minimax
from simple_rl.agents.AgentClass import Agent
from simple_rl.mdp.markov_game.MarkovGameMDPClass import MarkovGameMDP
from typing import List


class GeneratorPool(Agent):
    def __init__(self, name: str, game: MarkovGameMDP, player: int, check_assumptions: bool = False) -> None:
        Agent.__init__(self, name=name, actions=[])
        self.generators = []
        self.generators.append(Bullied(game, player, check_assumptions=check_assumptions))
        self.generators.append(BullyPunish(game, player, check_assumptions=check_assumptions))
        self.generators.append(CFR(game, player, check_assumptions=check_assumptions))
        self.generators.append(Coop(game, player, check_assumptions=check_assumptions))
        self.generators.append(CoopPunish(game, player, check_assumptions=check_assumptions))
        self.generators.append(Minimax(game, player, check_assumptions=check_assumptions))
        self.check_assumptions = check_assumptions

    def act(self, state, reward, round_num):
        generator_to_action = {}

        for i, generator in enumerate(self.generators):
            generator_to_action[i] = generator.act(state, reward, round_num)

        return generator_to_action

    def store_terminal_state(self, state, reward, generator_just_used_idx: int) -> None:
        if self.check_assumptions:
            for i, generator in enumerate(self.generators):
                was_used = i == generator_just_used_idx
                generator.check_assumptions(state, reward, was_used)

    def assumptions(self, generator_idx: int) -> List[float]:
        return self.generators[generator_idx].assumptions()