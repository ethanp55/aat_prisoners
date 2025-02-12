import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

N_TRAIN_TEST_RUNS = 30
baselines = {'bullypunish': -1, 'selfplay': 3, 'coop': 3}
folder = '../simulations/adaptability_results/'

# for file in os.listdir(folder):
#     agent_name = file.split('_')[0]
#     if agent_name not in results:
#         results[agent_name] = {}
#     opp_type, n_rounds = file.split('_')[1], int(file.split('_')[2][2:4])
#     comparison = baselines[opp_type] * n_rounds
#     minimax_val, lowest_reward = -1 * n_rounds, -3 * n_rounds
#     data = np.genfromtxt(f'{folder}{file}', delimiter=',', skip_header=0)
#
#     if opp_type == 'bullypunish':
#         avg_reward = sum([row[-1] for row in data]) / len(data)
#         regret = (comparison - lowest_reward) - (avg_reward - lowest_reward)
#         val = 1 - max((regret / (comparison - lowest_reward)), 0)
#
#     elif opp_type == 'selfplay':
#         row_avgs = [sum(row) / len(row) for row in data]
#         avg_reward = sum(row_avgs) / len(row_avgs)
#         regret = (comparison - minimax_val) - (avg_reward - minimax_val)
#         val = 1 - min((regret / (comparison - minimax_val)), 1)
#
#     elif opp_type == 'coop':
#         avg_reward = sum([row[-1] for row in data]) / len(data)
#         regret = (comparison - minimax_val) - (avg_reward - minimax_val)
#         val = 1 - min((regret / (comparison - minimax_val)), 1)
#
#     else:
#         raise Exception(f'{opp_type} is not a defined opponent type')
#
#     assert 0 <= val <= 1
#     results[agent_name][opp_type] = val
#
# rc_scores = []
# for agent, res, in results.items():
#     print(agent)
#     defect_score, self_play_score, coop_score = res['bullypunish'], res['selfplay'], res['coop']
#     robust_coop_score = min([defect_score, self_play_score, coop_score])
#     print(f'Defect score: {defect_score}')
#     print(f'Self-play score: {self_play_score}')
#     print(f'Coop score: {coop_score}')
#     print(f'Robust coop score: {robust_coop_score}\n')
#     rc_scores.append((agent, robust_coop_score))
# rc_scores.sort(key=lambda x: x[1], reverse=True)
# print(rc_scores)

results_from_every_epoch = {}

for run_num in range(N_TRAIN_TEST_RUNS):
    results = {}
    for file in os.listdir(folder):
        if f'epoch={run_num}' not in file:
            continue
        agent_name = file.split('_')[0]
        if agent_name not in results_from_every_epoch:
            results_from_every_epoch[agent_name] = {}
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

        val = max(val, 0)
        val = min(val, 1)
        assert 0 <= val <= 1
        results[agent_name][opp_type] = val

    for agent, res, in results.items():
        defect_score, self_play_score, coop_score = res['bullypunish'], res['selfplay'], res['coop']
        avg_coop_score = (self_play_score + coop_score) / 2
        adapt_score = min([defect_score, avg_coop_score])
        results_from_every_epoch[agent]['d'] = results_from_every_epoch[agent].get('d', []) + [defect_score]
        results_from_every_epoch[agent]['c'] = results_from_every_epoch[agent].get('c', []) + [avg_coop_score]
        results_from_every_epoch[agent]['a'] = results_from_every_epoch[agent].get('a', []) + [adapt_score]


conditions = ['Defect', 'Self-Play', 'Cooperate', 'Adaptability']
alg_names = ['DQN', 'RawO', 'RAlegAATr', 'RAAT', 'AleqgAATr', 'QAlegAATr', 'AlegAATr']
alg_plot_names = ['EG-Raw', 'REGAE-Raw', 'EG-AAT', 'REGAE-AAT', 'EG-RawAAT', 'REGAE-RawAAT', 'AlegAATr']
colors = ['#ef8a62', '#67a9cf', '#ef8a62', '#67a9cf', '#ef8a62', '#67a9cf', '#999999']
scores, score_types, learning_algs, features, domain = [], [], [], [], []
for cond in ['d', 'c', 'a']:
    avgs, ses = [], []
    for alg in alg_names:
        alg_data = results_from_every_epoch[alg][cond]
        avgs.append(np.mean(alg_data))
        ses.append(np.std(alg_data, ddof=1) / np.sqrt(len(alg_data)))

        n_samples = len(alg_data)
        scores.extend(alg_data)
        score_types.extend([cond] * n_samples)
        name = alg_plot_names[alg_names.index(alg)]
        learning_alg = name.split('-')[0] if alg != 'AlegAATr' else 'REGAEKNN'
        feature_set = name.split('-')[1] if alg != 'AlegAATr' else 'AATKNN'
        learning_algs.extend([learning_alg] * n_samples)
        features.extend([feature_set] * n_samples)
        domain.extend(['prisoners'] * n_samples)

    plt.figure(figsize=(10, 3))
    plt.grid()
    plt.bar(alg_plot_names, avgs, yerr=ses, capsize=5, color=colors)
    plt.xlabel('Algorithm', fontsize=18, fontweight='bold')
    plt.ylabel('Score', fontsize=18, fontweight='bold')
    plt.savefig(f'../simulations/{cond}.png', bbox_inches='tight')
    plt.clf()

df = pd.DataFrame({
    'score': scores,
    'score_type': score_types,
    'learning_alg': learning_algs,
    'feature_set': features,
    'domain': domain
})
df.to_csv('./prisoners_adaptability_results.csv', index=False)
