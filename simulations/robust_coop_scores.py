import numpy as np
import os
from scipy.stats import hmean

baselines = {'bullypunish': -1, 'cooppunish': 3}
results, folder = {}, '../simulations/results/'
N_ROUNDS = 30

for file in os.listdir(folder):
    agent_name = file.split('_')[0]
    if agent_name not in results:
        results[agent_name] = {}
    opp_type, n_rounds = file.split('_')[1], int(file.split('_')[2][2:4])
    if opp_type not in baselines or n_rounds != N_ROUNDS:
        continue
    comparison = baselines[opp_type] * n_rounds
    data = np.genfromtxt(f'{folder}{file}', delimiter=',', skip_header=0)

    if opp_type == 'bullypunish':
        avg_final_reward = sum([row[-1] for row in data]) / len(data)
        assert avg_final_reward <= comparison and avg_final_reward < 0
        val = comparison / avg_final_reward

    elif opp_type == 'cooppunish':
        worst = -3 * n_rounds
        denom = comparison - worst
        assert denom > 0
        all_deviations = []
        for row in data:
            deviations = [1 - ((comparison - min(reward, comparison)) / denom) for reward in row]
            all_deviations.append(sum(deviations) / len(deviations))
        val = sum(all_deviations) / len(all_deviations)

    else:
        raise Exception(f'{opp_type} is not a defined opponent type')

    assert 0 <= val <= 1
    # results[agent_name][opp_type] = results[agent_name].get(opp_type, []) + [val]
    results[agent_name][opp_type] = val

rc_scores = []
for agent, res, in results.items():
    print(agent)
    # defect_scores, coop_scores = res['bullypunish'], res['cooppunish']
    # defect_score = sum(defect_scores) / len(defect_scores)
    # coop_score = hmean(coop_scores)
    defect_score, coop_score = res['bullypunish'], res['cooppunish']
    robust_coop_score = min(defect_score, coop_score)
    print(f'Defect score: {defect_score}')
    print(f'Coop score: {coop_score}')
    print(f'Robust coop score: {robust_coop_score}\n')
    rc_scores.append((agent, robust_coop_score))
rc_scores.sort(key=lambda x: x[1], reverse=True)
print(rc_scores)
