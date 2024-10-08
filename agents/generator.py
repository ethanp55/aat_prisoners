from aat.checker import *
from agents.folk_egal import FolkEgalAgent, FolkEgalPunishAgent
from agents.minimax_q import MinimaxAgent
from agents.cfr import CFRAgent
from simple_rl.agents.AgentClass import Agent
from simple_rl.mdp.markov_game.MarkovGameMDPClass import MarkovGameMDP
from typing import List, Optional


class Generator(Agent):
    def __init__(self, name: str, generator: Agent, baseline: float,
                 checker: Optional[AssumptionChecker] = None) -> None:
        Agent.__init__(self, name=name, actions=[])
        self.generator = generator
        self.baseline = baseline
        self.checker = checker

    def act(self, state, reward, round_num):
        return self.generator.act(state, reward, round_num)

    def check_assumptions(self, state, reward, was_used) -> None:
        assert self.checker is not None
        self.checker.check_assumptions(state, reward, was_used)

    def assumptions(self) -> List[float]:
        assert self.checker is not None

        return self.checker.assumptions()


class Coop(Generator):
    def __init__(self, game: MarkovGameMDP, player: int, check_assumptions: bool = False) -> None:
        name = 'CoopGenerator'
        initial_state = game.get_init_state()
        game_name = str(game)
        generator = FolkEgalAgent(name, 1, 1, initial_state, game_name, read_from_file=True, player=player)
        checker = CoopChecker() if check_assumptions else None
        Generator.__init__(self, name=name, generator=generator, baseline=3, checker=checker)


class CoopPunish(Generator):
    def __init__(self, game: MarkovGameMDP, player: int, check_assumptions: bool = False) -> None:
        name = 'CoopPunishGenerator'
        initial_state = game.get_init_state()
        game_name = str(game)
        coop_agent = FolkEgalAgent('foo', 1, 1, initial_state, game_name, read_from_file=True, player=player)
        generator = FolkEgalPunishAgent(name, coop_agent, game_name, game)
        checker = CoopPunishChecker() if check_assumptions else None
        Generator.__init__(self, name=name, generator=generator, baseline=3, checker=checker)


class Bully(Generator):
    def __init__(self, game: MarkovGameMDP, player: int, check_assumptions: bool = False) -> None:
        name = 'BullyGenerator'
        initial_state = game.get_init_state()
        game_name = str(game)
        generator = FolkEgalAgent(name, 1, 1, initial_state, game_name + '_bully', read_from_file=True,
                                  specific_policy=True, p1_weight=1.0 - player, player=player)
        Generator.__init__(self, name=name, baseline=5, generator=generator)


class BullyPunish(Generator):
    def __init__(self, game: MarkovGameMDP, player: int, check_assumptions: bool = False) -> None:
        name = 'BullyPunishGenerator'
        initial_state = game.get_init_state()
        game_name = str(game)
        bully_agent = FolkEgalAgent('foo', 1, 1, initial_state, game_name + '_bully', read_from_file=True,
                                    specific_policy=True, p1_weight=1.0 - player, player=player)
        generator = FolkEgalPunishAgent(name, bully_agent, game_name, game)
        checker = BullyPunishChecker() if check_assumptions else None
        Generator.__init__(self, name=name, generator=generator, baseline=5, checker=checker)


class Bullied(Generator):
    def __init__(self, game: MarkovGameMDP, player: int, check_assumptions: bool = False) -> None:
        name = 'BulliedGenerator'
        initial_state = game.get_init_state()
        game_name = str(game)
        generator = FolkEgalAgent(name, 1, 1, initial_state, game_name + '_bullied', read_from_file=True,
                                  specific_policy=True, p1_weight=abs(0.2 - player), player=player)
        checker = BulliedChecker() if check_assumptions else None
        Generator.__init__(self, name=name, generator=generator, baseline=-3, checker=checker)


class Minimax(Generator):
    def __init__(self, game: MarkovGameMDP, player: int, check_assumptions: bool = False) -> None:
        name = 'MinimaxGenerator'
        initial_state = game.get_init_state()
        game_name = str(game)
        generator = MinimaxAgent(name, 1, initial_state, game_name, read_from_file=True, player=player)
        checker = MinimaxChecker() if check_assumptions else None
        Generator.__init__(self, name=name, generator=generator, baseline=-1, checker=checker)


class CFR(Generator):
    def __init__(self, game: MarkovGameMDP, player: int, check_assumptions: bool = False) -> None:
        name = 'CFRGenerator'
        initial_state = game.get_init_state()
        game_name = str(game)
        generator = CFRAgent(name=name, initial_game_state=initial_state, n_iterations=1, file_name=game_name,
                             read_from_file=True, player=player)
        checker = CFRChecker() if check_assumptions else None
        Generator.__init__(self, name=name, generator=generator, baseline=-1, checker=checker)
