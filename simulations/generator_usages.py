import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

folder = '../simulations/generator_usage/'
files = os.listdir(folder)
agent_names = set()
N_GENERATORS = 6

for file in files:
    agent_name = file.split('_')[0]
    agent_names.add(agent_name)

for agent_name in agent_names:
    num_rounds, opponent_types = [], []
    list_of_generator_counts = [[] for _ in range(N_GENERATORS)]

    for file in files:
        name = file.split('_')[0]
        if name != agent_name:
            continue
        n_rounds = file.split('r=')[1].split('_')[0]
        opp_type = file[len(agent_name) + 1:].split('_')[0]
        data = np.genfromtxt(f'{folder}{file}', delimiter=',', skip_header=1)
        curr_generator_counts = {}

        for row in data:
            generator = row[-1]
            curr_generator_counts[generator] = curr_generator_counts.get(generator, 0) + 1

        for i in range(N_GENERATORS):
            gen_count = curr_generator_counts.get(i, 0)
            list_of_generator_counts[i].append(gen_count)

        num_rounds.append(n_rounds)
        opponent_types.append(opp_type)

    data_dict = {
        'n_rounds': num_rounds,
        'opponent_type': opponent_types,
    }
    for i in range(N_GENERATORS):
        data_dict[f'generator_{i}'] = list_of_generator_counts[i]

    df = pd.DataFrame(data_dict)

    # Plot generator usages by number of rounds
    df_melted = pd.melt(df, id_vars=['n_rounds'], value_vars=[f'generator_{i}' for i in range(N_GENERATORS)],
                        var_name='generator', value_name='usage_count')
    df_grouped = df_melted.groupby(['n_rounds', 'generator']).sum().reset_index()
    unique_num_rounds = df_grouped['n_rounds'].unique()
    bar_positions = np.arange(len(df_grouped['generator'].unique()))
    bar_width = 0.2
    fig, ax = plt.subplots()
    for i, num_rounds in enumerate(unique_num_rounds):
        subset = df_grouped[df_grouped['n_rounds'] == num_rounds]
        ax.bar(bar_positions + i * bar_width, subset['usage_count'], bar_width, label=f'rounds={num_rounds}')
    ax.set_xticks(bar_positions + bar_width * (len(unique_num_rounds) - 1) / 2)
    ax.set_xticklabels([i for i in range(N_GENERATORS)])
    ax.set_xlabel('Generator')
    ax.set_ylabel('Usage Count')
    ax.legend()
    plt.grid()
    plt.savefig(f'../simulations/generator_usage_plots/{agent_name}_rounds.png', bbox_inches='tight')
    plt.clf()

    # Overall
    generator_usage = df[[f'generator_{i}' for i in range(N_GENERATORS)]].sum()
    plt.figure(figsize=(10, 3))
    plt.grid()
    plt.bar(generator_usage.index, generator_usage.values)
    plt.xticks(ticks=range(N_GENERATORS), labels=['Bullied', 'Bully', 'CFR', 'Coop', 'Coop-Punish', 'Minimax'])
    plt.xlabel('Generator', fontsize=18, fontweight='bold')
    plt.ylabel('Counts', fontsize=18, fontweight='bold')
    plt.savefig(f'../simulations/generator_usage_plots/{agent_name}_overall.png', bbox_inches='tight')
    plt.clf()
