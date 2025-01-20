from simple_rl.agents.AgentClass import Agent


class TitForTat(Agent):
    def __init__(self, player: int) -> None:
        Agent.__init__(self, name='TitForTat', actions=[])
        self.player = player
        self.defect = False  # Flag for choosing whether to defect - initially turned off in order to cooperate in round 1

    def store_terminal_state(self, state, reward) -> None:
        opp_action = state.actions[1 - self.player]
        self.defect = opp_action == 'defect'  # Defect the next round if the opponent defected this round

    def act(self, state, reward, round_num):
        action = 'defect' if self.defect else 'cooperate'

        return action, action
