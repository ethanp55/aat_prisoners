import numpy as np
import pandas as pd
import os
from statsmodels.stats.multicomp import pairwise_tukeyhsd

MAX_N_PLAYERS = 20
SAVE_DATA = False
FINAL_POPS, PERCENTILES, WINS, POP_SUMS = True, False, False, False


def _calculate_percentile(array, value):
    sorted_array = np.sort(array)
    rank = np.searchsorted(sorted_array, value, side='right')

    return rank / len(array)


results, folder = {}, '../simulations/results/'
results['overall'], results['banditsociety'] = {}, {}
names, pop_conditions, num_players, num_rounds, num_cats, opponent_types, initial_pop_classes = \
    [], [], [], [], [], [], []
pop_sums, final_pops, percentiles, wins = [], [], [], []

for file in os.listdir(folder):
    agent_name = file.split('_')[0]
    if 'generator' in agent_name:
        # Don't save formatted results with the best generator
        if SAVE_DATA:
            continue
        agent_name = agent_name + '_' + file.split('_')[1]
        if agent_name != 'generator_10':
            continue

    pop_condition = file.split('pop=')[1].split('_')[0]
    n_players = file.split('p=')[2].split('_')[0]
    n_rounds = file.split('r=')[1].split('_')[0]
    n_cats = file.split('c=')[1].split('_')[0][0]
    opp_type = file[len(agent_name) + 1:].split('_')[0]
    opp_type = 'banditsociety' if opp_type == 'basicbandits' else opp_type
    data = np.genfromtxt(f'{folder}{file}', delimiter=',', skip_header=0)
    if data.shape[0] == 0:
        continue
    if len(data.shape) == 1:
        data = data.reshape(1, -1)

    if SAVE_DATA:
        for row in data:
            # Add the condition info
            names.append(agent_name)
            pop_conditions.append(pop_condition)
            num_players.append(n_players)
            num_rounds.append(n_rounds)
            num_cats.append(n_cats)
            opponent_types.append(opp_type)

            # Calculate and add the results
            agent_pop = row[-1]
            pop_sums.append(sum(row))
            final_pops.append(agent_pop)
            if opp_type == 'selfplay':
                percentile = 1.0
                won = 1
            else:
                percentile = _calculate_percentile(row, agent_pop)
                won = 1 if agent_pop == row.max() else 0
            percentiles.append(percentile)
            wins.append(won)

        if pop_condition == 'random':
            initial_pop_classes_data = pd.read_csv(f'../simulations/initial_pops/{file}', header=None)

            for _, row in initial_pop_classes_data.iterrows():
                initial_pop_classes.append(row[0])

        else:
            initial_pop_classes.extend(['equal' for _ in range(len(data))])

    if f'pop={pop_condition}' not in results:
        results[f'pop={pop_condition}'] = {}
    if f'p={n_players}' not in results:
        results[f'p={n_players}'] = {}
    if f'r={n_rounds}' not in results:
        results[f'r={n_rounds}'] = {}
    if f'c={n_cats}' not in results:
        results[f'c={n_cats}'] = {}

    n_rows, n_cols = data.shape
    if n_cols < MAX_N_PLAYERS:
        n_zeroes = MAX_N_PLAYERS - n_cols
        zeroes = np.zeros((n_rows, n_zeroes))
        data = np.concatenate((zeroes, data), axis=1)

    results[f'pop={pop_condition}'][agent_name] = np.concatenate(
        (results[f'pop={pop_condition}'][agent_name], data)) if agent_name in results[
        f'pop={pop_condition}'] else data
    results[f'p={n_players}'][agent_name] = np.concatenate(
        (results[f'p={n_players}'][agent_name], data)) if agent_name in results[f'p={n_players}'] else data
    results[f'r={n_rounds}'][agent_name] = np.concatenate(
        (results[f'r={n_rounds}'][agent_name], data)) if agent_name in results[f'r={n_rounds}'] else data
    results[f'c={n_cats}'][agent_name] = np.concatenate(
        (results[f'c={n_cats}'][agent_name], data)) if agent_name in results[f'c={n_cats}'] else data
    results['overall'][agent_name] = np.concatenate(
        (results['overall'][agent_name], data)) if agent_name in results['overall'] else data
    if opp_type == 'banditsociety':
        results['banditsociety'][agent_name] = np.concatenate((results['banditsociety'][agent_name], data)) \
            if agent_name in results['banditsociety'] else data

# Store in a csv file for analysis in Google Sheets (or MS Excel)
if SAVE_DATA:
    df = pd.DataFrame(
        {
            'algorithm': names,
            'pop_condition': pop_conditions,
            'n_players': num_players,
            'n_rounds': num_rounds,
            'n_cats': num_cats,
            'opponent_type': opponent_types,
            'society_pop_sum': pop_sums,
            'agent_final_pop': final_pops,
            'agent_percentile': percentiles,
            'agent_won': wins,
            'initial_class': initial_pop_classes
        }
    )
    df.to_csv('../simulations/formatted_results_hand_picked_fewer_cats.csv', index=False)


# Effect sizes for final popularities (overall)
def _cohens_d(group1, group2):
    mean_diff = np.mean(group1) - np.mean(group2)
    pooled_std = np.sqrt((np.std(group1, ddof=1) ** 2 + np.std(group2, ddof=1) ** 2) / 2)

    return mean_diff / pooled_std


overall_results = results['overall']
alegaatr_final_pops, latex_df = overall_results['AlegAATr'], []

for name, pops in overall_results.items():
    if name == 'AlegAATr':
        continue

    d = _cohens_d(alegaatr_final_pops[:, -1], pops[:, -1])

    # print(f'Cohen\'s d AlegAATr vs. {name}: {d}')
    latex_df.append((f'AlegAATr vs. {name}', round(d, 3)))

latex_df = pd.DataFrame(latex_df, columns=['Comparison', 'Effect Size'])
# print(latex_df.to_latex(index=False))
print(latex_df)

# Run comparison tests
for scenario_str, scenario_results in results.items():
    names, pop_sums, final_pops, percentiles, wins = [], [], [], [], []

    for agent_name, agent_results in scenario_results.items():
        for row in agent_results:
            agent_pop = row[-1]
            names.append(agent_name)
            pop_sums.append(sum(row))
            final_pops.append(agent_pop)
            percentiles.append(_calculate_percentile(row, agent_pop))
            wins.append(1 if agent_pop == row.max() else 0)

    print('-----------------------------------------------')
    print(f'STATISTICAL TESTS FOR {scenario_str} SCENARIOS')
    print('-----------------------------------------------')

    if FINAL_POPS:
        print('\nFINAL POPS:')
        df = pd.DataFrame({'algorithm': names, 'results': final_pops})
        print(pairwise_tukeyhsd(endog=df['results'], groups=df['algorithm'], alpha=0.05))

    if PERCENTILES:
        print('\nPERCENTILES:')
        df = pd.DataFrame({'algorithm': names, 'results': percentiles})
        print(pairwise_tukeyhsd(endog=df['results'], groups=df['algorithm'], alpha=0.05))

    if WINS:
        print('\nWINS:')
        df = pd.DataFrame({'algorithm': names, 'results': wins})
        print(pairwise_tukeyhsd(endog=df['results'], groups=df['algorithm'], alpha=0.05))

    if POP_SUMS:
        print('\nPOP SUMS:')
        df = pd.DataFrame({'algorithm': names, 'results': pop_sums})
        print(pairwise_tukeyhsd(endog=df['results'], groups=df['algorithm'], alpha=0.05))

    print('-----------------------------------------------')
    print('-----------------------------------------------')
    print('-----------------------------------------------\n')
