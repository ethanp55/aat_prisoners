from game.prisoners_dilemma import ACTIONS
from typing import List


class AssumptionChecker:
    def __init__(self) -> None:
        pass

    def check_assumptions(self, state, reward, was_used) -> None:
        pass

    def assumptions(self) -> List[float]:
        pass


class BulliedChecker(AssumptionChecker):
    def __init__(self) -> None:
        AssumptionChecker.__init__(self)

        # Assumption estimates
        self.insists = 1.0
        self.wont_harm = 1.0
        self.progress = 0.5

        # Previous values (used in estimate calculations)
        self.prev_insists = 1.0
        self.prev_wont_harm = 1.0

        # Other values (used in estimate calculations)
        self.prev_reward = 0
        self.prev_rewards = []
        self.all_rewards = []

    def check_assumptions(self, state, reward, was_used) -> None:
        our_action, their_action = state.actions[1], state.actions[0]
        assert our_action in ACTIONS and their_action in ACTIONS

        increase = reward - self.prev_reward
        self.prev_reward = reward
        self.all_rewards.append(increase)

        # Other player insists that our payoff is less than or equal to the bullied average reward
        bullied_average_reward = -3
        avg_reward = sum(self.all_rewards) / len(self.all_rewards)
        self.insists = ((0.8 * self.prev_insists) + 0.2) if avg_reward <= bullied_average_reward \
            else (0.8 * self.prev_insists)
        self.prev_insists = self.insists

        # Other player won't harm us if we conform to the bullied outcome
        if was_used:
            self.wont_harm = ((0.8 * self.prev_wont_harm) + 0.2) if increase >= bullied_average_reward \
                else (0.7 * self.prev_wont_harm)
        else:
            self.wont_harm = self.prev_wont_harm
        self.prev_wont_harm = self.wont_harm

        # Progress check
        if was_used:
            self.prev_rewards.append(increase)
        optimistic_avg = -3
        r_max, r_min = 5, -3
        historical_avg = (sum(self.prev_rewards) / len(self.prev_rewards)) if len(self.prev_rewards) > 0 else 0
        self.progress = (0.5 + ((0.5 * (historical_avg - optimistic_avg)) / (r_max - optimistic_avg))) \
            if optimistic_avg < historical_avg else ((0.5 * (historical_avg - r_min)) / (optimistic_avg - r_min))

    def assumptions(self) -> List[float]:
        return [self.insists, self.wont_harm, self.progress]


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

    def check_assumptions(self, state, reward, was_used) -> None:
        our_action, their_action = state.actions[1], state.actions[0]
        assert our_action in ACTIONS and their_action in ACTIONS

        # Willing to be bullied (every round)
        self.willing_to_be_bullied_1 = ((0.7 * self.prev_willing_to_be_bullied_1) + 0.3) \
            if their_action == 'cooperate' else (0.3 * self.prev_willing_to_be_bullied_1)
        self.prev_willing_to_be_bullied_1 = self.willing_to_be_bullied_1

        # Willing to be bullied (rounds where bully punish is used)
        if was_used:
            self.willing_to_be_bullied_2 = ((0.7 * self.prev_willing_to_be_bullied_2) + 0.3) \
                if their_action == 'cooperate' else (0.7 * self.prev_willing_to_be_bullied_2)
        else:
            self.willing_to_be_bullied_2 = self.prev_willing_to_be_bullied_2
        self.prev_willing_to_be_bullied_2 = self.willing_to_be_bullied_2

        # Punishment was effective
        if was_used:
            self.responds_to_punishment = ((0.5 * self.prev_responds_to_punishment) + 0.5) \
                if their_action == 'cooperate' else (0.5 * self.prev_responds_to_punishment)
        else:
            self.responds_to_punishment = self.prev_responds_to_punishment
        self.prev_responds_to_punishment = self.responds_to_punishment

        # Progress check
        increase = reward - self.prev_reward
        self.prev_reward = reward
        if was_used:
            self.prev_rewards.append(increase)
        optimistic_avg = 5
        r_max, r_min = 5, -3
        historical_avg = (sum(self.prev_rewards) / len(self.prev_rewards)) if len(self.prev_rewards) > 0 else 0
        self.progress = (0.5 + ((0.5 * (historical_avg - optimistic_avg)) / (r_max - optimistic_avg))) \
            if optimistic_avg < historical_avg else ((0.5 * (historical_avg - r_min)) / (optimistic_avg - r_min))

    def assumptions(self) -> List[float]:
        return [self.willing_to_be_bullied_1, self.willing_to_be_bullied_2, self.responds_to_punishment, self.progress]


class CFRChecker(AssumptionChecker):
    def __init__(self) -> None:
        AssumptionChecker.__init__(self)

        # Assumption estimates
        self.played_best_response_1 = 1.0
        self.played_best_response_2 = 1.0
        self.progress = 0.5

        # Previous values (used in estimate calculations)
        self.prev_played_best_response_1 = 1.0
        self.prev_played_best_response_2 = 1.0

        # Other values (used in estimate calculations)
        self.prev_reward = 0
        self.prev_rewards = []

    def check_assumptions(self, state, reward, was_used) -> None:
        our_action, their_action = state.actions[1], state.actions[0]
        assert our_action in ACTIONS and their_action in ACTIONS

        # The other player played a best response (every round)
        k = 1 if their_action == 'defect' else 0
        self.played_best_response_1 = (0.8 * self.prev_played_best_response_1) + (0.2 * k)
        self.prev_played_best_response_1 = self.played_best_response_1

        # The other player played a best response (rounds where CFR is used)
        self.played_best_response_2 = ((0.8 * self.prev_played_best_response_2) + (0.2 * k)) if was_used \
            else self.prev_played_best_response_2
        self.prev_played_best_response_2 = self.played_best_response_2

        # Progress check
        increase = reward - self.prev_reward
        self.prev_reward = reward
        if was_used:
            self.prev_rewards.append(increase)
        optimistic_avg = -1
        r_max, r_min = 5, -3
        historical_avg = (sum(self.prev_rewards) / len(self.prev_rewards)) if len(self.prev_rewards) > 0 else 0
        self.progress = (0.5 + ((0.5 * (historical_avg - optimistic_avg)) / (r_max - optimistic_avg))) \
            if optimistic_avg < historical_avg else ((0.5 * (historical_avg - r_min)) / (optimistic_avg - r_min))

    def assumptions(self) -> List[float]:
        return [self.played_best_response_1, self.played_best_response_2, self.progress]


class CoopChecker(AssumptionChecker):
    def __init__(self) -> None:
        AssumptionChecker.__init__(self)

        # Assumption estimates
        self.wants_to_cooperate = 1.0
        self.reciprocates_cooperation = 1.0
        self.exploits_when_cooperate = 1.0
        self.progress = 0.5

        # Previous values (used in estimate calculations)
        self.prev_wants_to_cooperate = 1.0
        self.prev_reciprocates_cooperation = 1.0
        self.prev_exploits_when_cooperate = 1.0

        # Other values (used in estimate calculations)
        self.prev_reward = 0
        self.prev_rewards = []

    def check_assumptions(self, state, reward, was_used) -> None:
        our_action, their_action = state.actions[1], state.actions[0]
        assert our_action in ACTIONS and their_action in ACTIONS

        # The other player wants to cooperate
        nice = their_action == 'cooperate' and our_action == 'defect'
        cooperated = their_action == 'cooperate'
        if nice:
            self.wants_to_cooperate = (0.2 * self.prev_wants_to_cooperate) + 0.8
        elif cooperated:
            self.wants_to_cooperate = (0.9 * self.prev_wants_to_cooperate) + 0.1
        else:
            self.wants_to_cooperate = 0.8 * self.prev_wants_to_cooperate
        self.prev_wants_to_cooperate = self.wants_to_cooperate

        # The other player cooperates when we cooperate
        if was_used:
            self.reciprocates_cooperation = ((0.7 * self.prev_reciprocates_cooperation) + 0.3) if cooperated \
                else (0.7 * self.prev_reciprocates_cooperation)
        else:
            self.reciprocates_cooperation = self.prev_reciprocates_cooperation
        self.prev_reciprocates_cooperation = self.reciprocates_cooperation

        # The other player exploits us when we cooperate
        coop_reward = 3
        their_reward = 5 if their_action == 'defect' else 3
        if was_used:
            self.exploits_when_cooperate = (0.7 * self.prev_exploits_when_cooperate) if their_reward > coop_reward \
                else ((0.8 * self.prev_exploits_when_cooperate) + 0.2)
        else:
            self.exploits_when_cooperate = self.prev_exploits_when_cooperate
        self.prev_exploits_when_cooperate = self.exploits_when_cooperate

        # Progress check
        increase = reward - self.prev_reward
        self.prev_reward = reward
        if was_used:
            self.prev_rewards.append(increase)
        optimistic_avg = 3
        r_max, r_min = 5, -3
        historical_avg = (sum(self.prev_rewards) / len(self.prev_rewards)) if len(self.prev_rewards) > 0 else 0
        self.progress = (0.5 + ((0.5 * (historical_avg - optimistic_avg)) / (r_max - optimistic_avg))) \
            if optimistic_avg < historical_avg else ((0.5 * (historical_avg - r_min)) / (optimistic_avg - r_min))

    def assumptions(self) -> List[float]:
        return [self.wants_to_cooperate, self.reciprocates_cooperation, self.exploits_when_cooperate, self.progress]


class CoopPunishChecker(AssumptionChecker):
    def __init__(self) -> None:
        AssumptionChecker.__init__(self)

        # Assumption estimates
        self.they_cooperate = 1.0
        self.responds_to_punishment = 1.0
        self.progress = 0.5

        # Previous values (used in estimate calculations)
        self.prev_they_cooperate = 1.0
        self.prev_responds_to_punishment = 1.0

        # Other values (used in estimate calculations)
        self.prev_reward = 0
        self.prev_rewards = []

    def check_assumptions(self, state, reward, was_used) -> None:
        our_action, their_action = state.actions[1], state.actions[0]
        assert our_action in ACTIONS and their_action in ACTIONS

        # The other player cooperates
        cooperated = their_action == 'cooperate'
        if was_used:
            self.they_cooperate = ((0.7 * self.prev_they_cooperate) + 0.3) if cooperated \
                else (0.7 * self.prev_they_cooperate)
        else:
            self.they_cooperate = self.prev_they_cooperate
        self.prev_they_cooperate = self.they_cooperate

        # Punishment was effective
        if was_used and our_action == 'defect':
            self.responds_to_punishment = ((0.5 * self.prev_responds_to_punishment) + 0.5) \
                if their_action == 'cooperate' else (0.5 * self.prev_responds_to_punishment)
        else:
            self.responds_to_punishment = self.prev_responds_to_punishment
        self.prev_responds_to_punishment = self.responds_to_punishment

        # Progress check
        increase = reward - self.prev_reward
        self.prev_reward = reward
        if was_used:
            self.prev_rewards.append(increase)
        optimistic_avg = 3
        r_max, r_min = 5, -3
        historical_avg = (sum(self.prev_rewards) / len(self.prev_rewards)) if len(self.prev_rewards) > 0 else 0
        self.progress = (0.5 + ((0.5 * (historical_avg - optimistic_avg)) / (r_max - optimistic_avg))) \
            if optimistic_avg < historical_avg else ((0.5 * (historical_avg - r_min)) / (optimistic_avg - r_min))

    def assumptions(self) -> List[float]:
        return [self.they_cooperate, self.responds_to_punishment, self.progress]


class MinimaxChecker(AssumptionChecker):
    def __init__(self) -> None:
        AssumptionChecker.__init__(self)

        # Assumption estimates
        self.they_attack = 1.0
        self.progress = 0.5

        # Previous values (used in estimate calculations)
        self.prev_they_attack = 1.0

        # Other values (used in estimate calculations)
        self.prev_reward = 0
        self.prev_rewards = []

    def check_assumptions(self, state, reward, was_used) -> None:
        our_action, their_action = state.actions[1], state.actions[0]
        assert our_action in ACTIONS and their_action in ACTIONS

        increase = reward - self.prev_reward
        self.prev_reward = reward

        # They play the attack strategy
        minimax_val = -1
        self.they_attack = ((0.8 * self.prev_they_attack) + 0.2) if increase < minimax_val \
            else (0.8 * self.prev_they_attack)

        # Progress check
        if was_used:
            self.prev_rewards.append(increase)
        optimistic_avg = -1
        r_max, r_min = 5, -3
        historical_avg = (sum(self.prev_rewards) / len(self.prev_rewards)) if len(self.prev_rewards) > 0 else 0
        self.progress = (0.5 + ((0.5 * (historical_avg - optimistic_avg)) / (r_max - optimistic_avg))) \
            if optimistic_avg < historical_avg else ((0.5 * (historical_avg - r_min)) / (optimistic_avg - r_min))

    def assumptions(self) -> List[float]:
        return [self.they_attack, self.progress]
