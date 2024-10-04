from game.prisoners_dilemma import ACTIONS
from typing import List


class AssumptionChecker:
    def __init__(self) -> None:
        pass

    def check_assumptions(self, state, reward, round_num):
        pass

    def assumptions(self) -> List[float]:
        pass


class BullyPunishChecker(AssumptionChecker):
    def __init__(self) -> None:
        AssumptionChecker.__init__(self)

        # Assumption estimates
        self.willing_to_be_bullied_1 = 1.0
        self.willing_to_be_bullied_2 = 1.0
        self.responds_to_punishment = 1.0
        self.progress = 0.5

        # Previous values (used in estimate calculations)
        self.prev_willing_to_be_bullied_1 = 1.0
        self.prev_willing_to_be_bullied_2 = 1.0
        self.prev_responds_to_punishment = 1.0

        # Other values (used in estimate calculations)
        self.prev_reward = 0
        self.prev_rewards = []

    def check_assumptions(self, state, reward, round_num, was_used):
        our_action, their_action = state.actions[1], state.actions[0]
        assert our_action in ACTIONS and their_action in ACTIONS

        # Willing to be bullied (every round)
        self.willing_to_be_bullied_1 = ((0.7 * self.prev_willing_to_be_bullied_1) + 0.3) \
            if their_action == 'cooperate' else (0.3 * self.prev_willing_to_be_bullied_1)
        self.prev_willing_to_be_bullied_1 = self.willing_to_be_bullied_1

        # Willing to be bullied (rounds where bully punish is used)
        if was_used:
            self.willing_to_be_bullied_2 = ((0.7 * self.prev_willing_to_be_bullied_2) + 0.3) \
                if their_action == 'cooperate' else (0.3 * self.prev_willing_to_be_bullied_2)
            self.prev_willing_to_be_bullied_2 = self.willing_to_be_bullied_2
        else:
            self.willing_to_be_bullied_2 = self.prev_willing_to_be_bullied_2

        # Punishment was effective
        self.responds_to_punishment = ((0.5 * self.prev_responds_to_punishment) + 0.5) \
            if their_action == 'cooperate' else (0.5 * self.prev_responds_to_punishment)
        self.prev_responds_to_punishment = self.responds_to_punishment

        # Progress check
        increase = reward - self.prev_reward
        self.prev_reward = reward
        if was_used:
            self.prev_rewards.append(increase)
        optimistic_avg = 5
        r_max, r_min = 5, -3
        historical_avg = sum(self.prev_rewards) / len(self.prev_rewards)
        self.progress = (0.5 + ((0.5 * (historical_avg - optimistic_avg)) / (r_max - optimistic_avg))) \
            if optimistic_avg < historical_avg else ((0.5 * (historical_avg - r_min)) / (optimistic_avg - r_min))

    def assumptions(self) -> List[float]:
        return [self.willing_to_be_bullied_1, self.willing_to_be_bullied_2, self.responds_to_punishment, self.progress]
