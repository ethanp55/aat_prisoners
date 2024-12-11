import numpy as np
import os

baselines = {'bullypunish': -1, 'selfplay': 3, 'coop': 3}
results, folder = {}, '../simulations/adaptability_results/'

for file in os.listdir(folder):
    agent_name = file.split('_')[0]
    if agent_name not in results:
        results[agent_name] = {}
    opp_type, n_rounds = file.split('_')[1], int(file.split('_')[2][2:4])
    comparison = baselines[opp_type] * n_rounds
    minimax_val, lowest_reward = -1 * n_rounds, -3 * n_rounds
    data = np.genfromtxt(f'{folder}{file}', delimiter=',', skip_header=0)

    if opp_type == 'bullypunish':
        avg_reward = sum([row[-1] for row in data]) / len(data)
        regret = (comparison - lowest_reward) - (avg_reward - lowest_reward)
        val = 1 - max((regret / (comparison - lowest_reward)), 0)

    elif opp_type == 'selfplay':
        row_avgs = [sum(row) / len(row) for row in data]
        avg_reward = sum(row_avgs) / len(row_avgs)
        regret = (comparison - minimax_val) - (avg_reward - minimax_val)
        val = 1 - min((regret / (comparison - minimax_val)), 1)

    elif opp_type == 'coop':
        avg_reward = sum([row[-1] for row in data]) / len(data)
        regret = (comparison - minimax_val) - (avg_reward - minimax_val)
        val = 1 - min((regret / (comparison - minimax_val)), 1)

    else:
        raise Exception(f'{opp_type} is not a defined opponent type')

    assert 0 <= val <= 1
    results[agent_name][opp_type] = val

rc_scores = []
for agent, res, in results.items():
    print(agent)
    defect_score, self_play_score, coop_score = res['bullypunish'], res['selfplay'], res['coop']
    robust_coop_score = min([defect_score, self_play_score, coop_score])
    print(f'Defect score: {defect_score}')
    print(f'Self-play score: {self_play_score}')
    print(f'Coop score: {coop_score}')
    print(f'Robust coop score: {robust_coop_score}\n')
    rc_scores.append((agent, robust_coop_score))
rc_scores.sort(key=lambda x: x[1], reverse=True)
print(rc_scores)
