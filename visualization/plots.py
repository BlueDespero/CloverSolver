import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import dash_core_components as dcc
import dash_html_components as html
from tests.test_effectiveness.run_tests import runsingle

from genetic.common import lambda_plus_mu, lambda_coma_mu
from genetic.plugin_algorithms.crossover import exchange_two_rows_crossover, exchange_two_columns_crossover, \
    exchange_two_boxes_crossover, single_point_crossover, double_point_crossover
from genetic.plugin_algorithms.fitness import quadratic_fitness, linear_fitness
from genetic.plugin_algorithms.ga_base import SGA
from genetic.plugin_algorithms.initial_pop import uniform_initial_population
from genetic.plugin_algorithms.mutation import shuffle_column_mutation, shuffle_row_mutation, shuffle_box_mutation, \
    reverse_bit_mutation
from tests.test_common import default_termination_condition


def translate_operator(name):
    d = {"exchange_two_rows_crossover": exchange_two_rows_crossover,
         "exchange_two_columns_crossover": exchange_two_columns_crossover,
         "exchange_two_boxes_crossover": exchange_two_boxes_crossover,
         "single_point_crossover": single_point_crossover,
         "double_point_crossover": double_point_crossover,

         "quadratic_fitness": quadratic_fitness,
         "linear_fitness": linear_fitness,

         "SGA": SGA,

         "uniform_initial_population": uniform_initial_population,

         "shuffle_column_mutation": shuffle_column_mutation,
         "shuffle_row_mutation": shuffle_row_mutation,
         "shuffle_box_mutation": shuffle_box_mutation,
         "reverse_bit_mutation": reverse_bit_mutation,

         "default_termination_condition": default_termination_condition,

         "lambda_plus_mu": lambda_plus_mu,
         "lambda_coma_mu": lambda_coma_mu
         }
    return d[name]


from dash.dependencies import Input, Output
from jupyter_dash import JupyterDash


def visualize(df):
    # Build App
    algorithm = df['algorithm'].iloc[0]
    initial_population_generation = df['initial_population_generation'].iloc[0]
    fitness_function = df['fitness_function'].iloc[0]
    mutation_operator = df['mutation_operator'].iloc[0]
    mutation_rate = df['mutation_rate'].iloc[0]
    crossover_operator = df['crossover_operator'].iloc[0]
    termination_condition = df['termination_condition'].iloc[0]
    population_merge_function = df['population_merge_function'].iloc[0]
    iterations = df['iterations'].iloc[0]
    population_size = df['population_size'].iloc[0]
    number_of_children = df['number_of_children'].iloc[0]

    app = JupyterDash(__name__)
    app.layout = html.Div([
        dcc.Graph(id='graph'),
        html.Div([
            html.Div([
                html.Label([
                    "Mutatuin rate",
                    dcc.Slider(
                        id="mutation_rate_slider",
                        min=df['mutation_rate'].min(),
                        max=df['mutation_rate'].max(),
                        step=None,
                        marks={float(val): str(val) for val in df.mutation_rate.unique()},
                        value=df['mutation_rate'].min()
                    )
                ])
            ],
                style={'width': '33.3%', 'float': 'left', 'display': 'inline-block'}),

            html.Div([
                html.Label([
                    "Number of children",
                    dcc.Slider(
                        id="number_of_children_slider",
                        min=df['number_of_children'].min(),
                        max=df['number_of_children'].max(),
                        step=None,
                        marks={int(val): str(val) for val in df.number_of_children.unique()},
                        value=df['number_of_children'].min()
                    )
                ])
            ],
                style={'width': '33.3%', 'display': 'inline-block'}),

            html.Div([
                html.Label([
                    "Population size",
                    dcc.Slider(
                        id="population_size_slider",
                        min=df['population_size'].min(),
                        max=df['population_size'].max(),
                        step=None,
                        marks={int(val): str(val) for val in df.population_size.unique()},
                        value=df['population_size'].min()
                    )
                ])
            ],
                style={'width': '33.3%', 'float': 'right', 'display': 'inline-block'})
        ]),

        html.Div([
            html.Div([
                html.Label([
                    "Mutation operator",
                    dcc.Dropdown(
                        id="mutation_operator_dropdown",
                        value=df['mutation_operator'].iloc[0],
                        options=[{'label': ''.join(word + ' ' for word in name.split("_")), 'value': name}
                                 for name in df.mutation_operator.unique()
                                 ],
                        searchable=False,
                        clearable=False
                    )
                ])
            ],
                style={'width': '33.3%', 'float': 'left', 'display': 'inline-block'}),

            html.Div([
                html.Label([
                    "Initial population generation",
                    dcc.Dropdown(
                        id="initial_population_generation_dropdown",
                        value=df["initial_population_generation"].iloc[0],
                        options=[{'label': ''.join(word + ' ' for word in name.split("_")), 'value': name}
                                 for name in df.initial_population_generation.unique()
                                 ],
                        searchable=False,
                        clearable=False
                    )
                ])
            ],
                style={'width': '33.3%', 'display': 'inline-block'}),

            html.Div([
                html.Label([
                    "Fitness function",
                    dcc.Dropdown(
                        id="fitness_function_dropdown",
                        value=df["fitness_function"].iloc[0],
                        options=[{'label': ''.join(word + ' ' for word in name.split("_")), 'value': name}
                                 for name in df.fitness_function.unique()
                                 ],
                        searchable=False,
                        clearable=False
                    )
                ])
            ],
                style={'width': '33.3%', 'float': 'right', 'display': 'inline-block'})
        ]),

        html.Div([
            html.Div([
                html.Label([
                    "Crossover operator",
                    dcc.Dropdown(
                        id="crossover_operator_dropdown",
                        value=df["crossover_operator"].iloc[0],
                        options=[{'label': ''.join(word + ' ' for word in name.split("_")), 'value': name}
                                 for name in df.crossover_operator.unique()
                                 ],
                        searchable=False,
                        clearable=False
                    )
                ])
            ],
                style={'width': '33.3%', 'float': 'left', 'display': 'inline-block'}),
            html.Div([
                html.Label([
                    "Population merge function",
                    dcc.Dropdown(
                        id="population_merge_function_dropdown",
                        value=df["population_merge_function"].iloc[0],
                        options=[{'label': ''.join(word + ' ' for word in name.split("_")), 'value': name}
                                 for name in df.population_merge_function.unique()
                                 ],
                        searchable=False,
                        clearable=False
                    )
                ])
            ],
                style={'width': '33.3%', 'display': 'inline-block'}),
            html.Div([
                html.Label([
                    "Termination condition",
                    dcc.Dropdown(
                        id="termination_condition_dropdown",
                        value=df["termination_condition"].iloc[0],
                        options=[{'label': ''.join(word + ' ' for word in name.split("_")), 'value': name}
                                 for name in df.termination_condition.unique()
                                 ],
                        searchable=False,
                        clearable=False
                    )
                ])
            ],
                style={'width': '33.3%', 'float': 'right', 'display': 'inline-block'})
        ])
    ])

    @app.callback(
        Output('graph', 'figure'),
        Input("mutation_rate_slider", 'value'),
        Input("population_size_slider", 'value'),
        Input("number_of_children_slider", 'value'),
        Input("mutation_operator_dropdown", 'value'),
        Input("initial_population_generation_dropdown", 'value'),
        Input("fitness_function_dropdown", 'value'),
        Input("crossover_operator_dropdown", 'value'),
        Input("population_merge_function_dropdown", 'value'),
        Input("termination_condition_dropdown", 'value'))
    def update_graph(mutation_val, population_size_val, number_of_children_val,
                     mutation_operator_name, initial_population_generation_name,
                     fitness_function_name, crossover_operator_name,
                     population_merge_function_name, termination_condition_name):
        mutation_rate = mutation_val
        population_size = population_size_val
        mutation_operator = mutation_operator_name
        initial_population_generation = initial_population_generation_name
        fitness_function = fitness_function_name
        crossover_operator = crossover_operator_name
        population_merge_function = population_merge_function_name
        termination_condition = termination_condition_name
        number_of_children = number_of_children_val
        filtered_df = filter_df(df, df.columns, [algorithm,
                                                 initial_population_generation,
                                                 fitness_function,
                                                 mutation_operator,
                                                 mutation_rate,
                                                 crossover_operator,
                                                 termination_condition,
                                                 population_merge_function,
                                                 iterations,
                                                 population_size,
                                                 number_of_children])
        return px.scatter(
            stretch_df(filtered_df), range_y=[0, 200],
            x="Iteration", y='result', color=stretch_df(filtered_df)["record_type"],
            render_mode="webgl", title="Best fitness:" + str(filtered_df["best_fitness"].iloc[0]),
            labels={'worst_record': "Worst record", 'best_record': "Best record", 'mean_record': "Mean record"}
        )

    # Run app and display result inline in the notebook
    app.run_server(mode='inline')


def fill(arr, expected_length):
    return np.hstack([arr, np.zeros(expected_length - arr.shape[0])])


def compress_from_file(path=None, l=None):
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
                [founded_dict['worst_record'],
                 fill(np.array(dict_result['fitness_record'])[:, 0], dict_result['iterations'])])
            founded_dict['best_record'] = np.vstack(
                [founded_dict['best_record'],
                 fill(np.array(dict_result['fitness_record'])[:, 1], int(dict_result['iterations']))])
            founded_dict['mean_record'] = np.vstack(
                [founded_dict['mean_record'],
                 fill(np.array(dict_result['fitness_record'])[:, 2], int(dict_result['iterations']))])

        else:
            if 'algorithm' not in dict_result.keys():
                dict_result['algorithm'] = 'SGA'
            dict_result['best_fitness'] = [dict_result['best_fitness']]
            dict_result['worst_record'] = fill(np.array(dict_result['fitness_record'])[:, 0],
                                               int(dict_result['iterations']))
            dict_result['best_record'] = fill(np.array(dict_result['fitness_record'])[:, 1],
                                              int(dict_result['iterations']))
            dict_result['mean_record'] = fill(np.array(dict_result['fitness_record'])[:, 2],
                                              int(dict_result['iterations']))
            # dict_result['size'] = int(np.sqrt(2))
            del dict_result['fitness_record']
            del dict_result['initial_state']
            del dict_result['best_result']

            list_of_dicts.append(dict_result)

    for dict_result in list_of_dicts:
        if dict_result['worst_record'].size == max(dict_result['worst_record'].shape):
            dict_result['worst_record'] = np.vstack(
                [dict_result['worst_record'], np.zeros(dict_result['worst_record'].shape[0])])
            dict_result['best_record'] = np.vstack(
                [dict_result['best_record'], np.zeros(dict_result['best_record'].shape[0])])
            dict_result['mean_record'] = np.vstack(
                [dict_result['mean_record'], np.zeros(dict_result['mean_record'].shape[0])])
        dict_result['best_fitness'] = np.min(np.array(dict_result['best_fitness']))
        dict_result['worst_record'] = np.mean(dict_result['worst_record'], axis=0)
        dict_result['best_record'] = np.mean(dict_result['best_record'], axis=0)
        dict_result['mean_record'] = np.mean(dict_result['mean_record'], axis=0)

    return list_of_dicts


def compress_list_of_dicts(list_of_dicts):
    new_dict = list_of_dicts

    list_of_dicts = []
    for dict_result in new_dict:
        founded = False

        # Extension for old version of dicts
        if 'algorithm' not in dict_result.keys():
            dict_result['algorithm'] = "SGA"

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

        dict_result['best_fitness'] = np.min(np.array(dict_result['best_fitness']))
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

        plt.show()


def dict_to_df(d):
    new_dict = {}
    for key in d.keys():
        if key not in ['mean_record', 'best_record', 'worst_record']:
            new_dict[key] = np.array([d[key]] * (d['best_record'].shape[0]))
    new_dict['mean_record'] = d['mean_record']
    new_dict['best_record'] = d['best_record']
    new_dict['worst_record'] = d['worst_record']
    new_dict['Iteration'] = np.arange(d['worst_record'].shape[0])
    df = pd.DataFrame(new_dict)
    return df


def compress_list_of_df(l):
    if l:
        df = l[0]
        for df_temp in l[1:]:
            df = pd.concat([df, df_temp], ignore_index=True)
        return df
    else:
        return pd.DataFrame()


def df_from_google_drive_file(pathlist):
    new_dict = []
    for path in pathlist:
        new_dict += compress_from_file(path)
    new_dict = compress_list_of_dicts(new_dict)

    l = [dict_to_df(temp_dict) for temp_dict in new_dict]
    df = compress_list_of_df(l)
    return df


def filter_df(df, columns, keys):
    filter_mask = np.ones(df.shape[0]).astype(bool)
    for column_name, key in zip(columns, keys):
        filter_mask = np.logical_and(filter_mask, df[column_name] == key)
    return df[filter_mask]


def stretch_df(df):
    first = df.drop(['mean_record', 'best_record'], axis=1)
    second = df.drop(['worst_record', 'best_record'], axis=1)
    third = df.drop(['worst_record', 'mean_record'], axis=1)

    unified_column_indexes = [element if element not in ['worst_record', 'mean_record', 'best_record'] else 'result' for
                              element in first.columns]

    first.columns = unified_column_indexes
    second.columns = unified_column_indexes
    third.columns = unified_column_indexes

    first['record_type'] = ['worst_record'] * first.shape[0]
    second['record_type'] = ['mean_record'] * second.shape[0]
    third['record_type'] = ['best_record'] * third.shape[0]

    result = pd.concat([first, second, third], ignore_index=True)

    return result


def visualize_SGA(pathlist):
    l = []
    for path in pathlist:
        l += compress_from_file(path)
    l = compress_list_of_dicts(l)
    l_of_dfs = [dict_to_df(d) for d in l]

    df = compress_list_of_df(l_of_dfs)
    if df.empty:
        print("All files are empty.")
    else:
        visualize(df)
