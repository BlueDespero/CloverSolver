import matplotlib.pyplot as plt
import numpy as np
import pickle
from pathlib import Path


def compress_from_file(path):
    # compress dicts with equal parameters from a file to few distinct dicts
    # Results are also compressed, by taking its mean
    infile = open(path, "rb")
    new_dict = pickle.load(infile)
    infile.close()

    list_of_dicts = []
    for dict_result in new_dict:
        founded = False
        for previously_saved_dict in list_of_dicts:
            equivalent_dict = True
            for key in previously_saved_dict.keys():
                if key not in ['best_fitness', 'fitness_record', 'best_result', 'initial_state', 'worst_record',
                               'best_record', 'mean_record']:
                    if dict_result[key] != previously_saved_dict[key]:
                        equivalent_dict = False
            if equivalent_dict:
                founded_dict = previously_saved_dict
                founded = True
                break

        if founded:
            founded_dict['best_fitness'].append(dict_result['best_fitness'])
            founded_dict['worst_record'] = np.vstack(
                [founded_dict['worst_record'], np.array(dict_result['fitness_record'])[:, 0]])
            founded_dict['best_record'] = np.vstack(
                [founded_dict['best_record'], np.array(dict_result['fitness_record'])[:, 1]])
            founded_dict['mean_record'] = np.vstack(
                [founded_dict['mean_record'], np.array(dict_result['fitness_record'])[:, 2]])

        else:
            dict_result['best_fitness'] = [dict_result['best_fitness']]
            dict_result['worst_record'] = np.array(dict_result['fitness_record'])[:, 0]
            dict_result['best_record'] = np.array(dict_result['fitness_record'])[:, 1]
            dict_result['mean_record'] = np.array(dict_result['fitness_record'])[:, 2]
            del dict_result['fitness_record']
            del dict_result['initial_state']
            del dict_result['best_result']

            list_of_dicts.append(dict_result)

    for dict_result in list_of_dicts:
        dict_result['best_fitness'] = np.mean(np.array(dict_result['best_fitness']))
        dict_result['worst_record'] = np.mean(dict_result['worst_record'], axis=0)
        dict_result['best_record'] = np.mean(dict_result['best_record'], axis=0)
        dict_result['mean_record'] = np.mean(dict_result['mean_record'], axis=0)

    return list_of_dicts


def compress_list_of_dicts(list_of_dicts):
    new_dict = list_of_dicts

    list_of_dicts = []
    for dict_result in new_dict:
        founded = False
        for previously_saved_dict in list_of_dicts:
            equivalent_dict = True
            for key in previously_saved_dict.keys():
                if key not in ['best_fitness', 'worst_record', 'best_record', 'mean_record']:
                    if dict_result[key] != previously_saved_dict[key]:
                        equivalent_dict = False
            if equivalent_dict:
                founded_dict = previously_saved_dict
                founded = True
                break

        if founded:
            founded_dict['best_fitness'].append(dict_result['best_fitness'])
            founded_dict['worst_record'] = np.vstack([founded_dict['worst_record'], dict_result['worst_record']])
            founded_dict['best_record'] = np.vstack([founded_dict['best_record'], dict_result['best_record']])
            founded_dict['mean_record'] = np.vstack([founded_dict['mean_record'], dict_result['mean_record']])

        else:
            dict_result['best_fitness'] = [dict_result['best_fitness']]
            list_of_dicts.append(dict_result)

    for dict_result in list_of_dicts:
        if dict_result['worst_record'].size == max(dict_result['worst_record'].shape):
            dict_result['worst_record'] = np.vstack(
                [dict_result['worst_record'], np.zeros(dict_result['worst_record'].shape[0])])
            dict_result['best_record'] = np.vstack(
                [dict_result['best_record'], np.zeros(dict_result['best_record'].shape[0])])
            dict_result['mean_record'] = np.vstack(
                [dict_result['mean_record'], np.zeros(dict_result['mean_record'].shape[0])])

        dict_result['best_fitness'] = np.mean(np.array(dict_result['best_fitness']))
        dict_result['worst_record'] = np.mean(dict_result['worst_record'], axis=0)
        dict_result['best_record'] = np.mean(dict_result['best_record'], axis=0)
        dict_result['mean_record'] = np.mean(dict_result['mean_record'], axis=0)

    return list_of_dicts


def plot_SGA(paths):
    new_dict = []
    for path in paths:
        new_dict += compress_from_file(path)
    new_dict = compress_list_of_dicts(new_dict)
    for d in new_dict:
        fig, axs = plt.subplots(2, figsize=(10, 7))

        axs[0].plot(np.arange(d['iterations']), d['worst_record'], label="worst")
        axs[0].plot(np.arange(d['iterations']), d['best_record'], label="best")
        axs[0].plot(np.arange(d['iterations']), d['mean_record'], label="mean")
        axs[0].legend()
        axs[0].set_title(d['algorithm'])

        val1 = ["Parameters"]
        val2 = [key for key in d.keys() if key not in ['worst_record', 'best_record', 'mean_record', 'algorithm']]
        val3 = [[d[key]] for key in d.keys() if key not in ['worst_record', 'best_record', 'mean_record', 'algorithm']]

        axs[1].set_axis_off()
        table = axs[1].table(
            cellText=val3,
            rowLabels=val2,
            colLabels=val1,
            colWidths=[0.7, 0.03],
            rowColours=["palegreen"] * (len(d.keys()) - 4),
            colColours=["palegreen"] * 1,
            cellLoc='center',
            loc='upper right')
        print(d.keys())
        break

        plt.show()


if __name__ == "__main__":
    # gets paths to all files in directory with extension .pickle
    pathlist = Path(r"C:\Users\user\PycharmProjects\CloverSolver\tests\test_effectiveness").rglob('*.pickle')
    pathlist = [path for path in pathlist]
    plot_SGA(pathlist)
