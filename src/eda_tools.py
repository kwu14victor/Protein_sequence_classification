'''
01.08.2024 ver 0.0.1 by KLW

This is the script that process the protein sequence dataset spreadsheets.

EDA or exploratory data analysis is an essential step for any data science project.
The tool provided in this scripts allow EDA and any subsequent analysis. 

'''

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

matplotlib.use('agg')

unrelated_columns = ['residueCount_y', 'macromoleculeType_y',\
                      'publicationYear','pdbxDetails','chainId']
#['macromoleculeType','experimentalTechnique',\
#                      'structureId','publicationYear','pdbxDetails',\
#                      'structureMolecularWeight','densityMatthews']
target_columns = ['crystallizationMethod','classification']
unique_columns = ['structureId','sequence']

def plot_distribution(counts_arr, fig_type = 'dist',\
    output_name='test.png', save_fig=False, x_ticks=False):
    '''
    This function visualizes the distribution of categorical features.
    Data is in the format of pandas.DataFrame.value_counts.
    x ticks are optional because the dataset has too many rows.
    Resulting graph will be saved as a separate figure.
    
    Arguments:
      counts_arr (series): data from pandas.DataFrame.value_counts
      fig_type (str): type of figure to plot
      output_name (str): file name for saved image
      save_fig (bool): whether to save the figure
      x_ticks (bool): whether to show the tickers for x axis
    
    '''
    fig_types = ['dist','bar']
    assert fig_type in fig_types,\
    f'only pick figure types from {fig_types}'
    if fig_type=='dist':
        plt.figure()
        sns.displot(counts_arr)
        plt.ylabel('ratio of records')
    elif fig_type=='bar':
        plt.figure(figsize=(15,20))
        counts_arr.plot(kind='bar')
        plt.ylabel('number of records')
        plt.rcParams['xtick.labelsize']=6
        if not x_ticks:
            plt.xticks([],[])
    if save_fig:
        plt.savefig(output_name)


def plot_data_from_dataframe(data_frame, y_axis, x_axis='classification',\
    output_name='test.png', save_fig=False, fig_type='hist'):
    '''
    This function visualizes the distribution of categorical features.
    Data has to be in the format of a pandas dataframe.
    Resulting histogram is plotted in different color/hue map.
    This figure will also be saved as a separate figure.
    
    Arguments:
      data_frame (pandas.DataFrane): dataset 
      y_axis (str): column index to get values for y axis
      x_axis (str): column index to get values for x axis
      output_name (str): file name for saved image
      save_fig (bool): whether to save the figure
      fig_type (str): type of figure to plot
    
    '''
    fig_types = ['hist','box']
    assert fig_type in fig_types,\
    f'only pick figure types from {fig_types}'
    if fig_type=='hist':
        plt.figure()
        sns.histplot(data=data_frame, x=x_axis,hue=y_axis,multiple="dodge", shrink=.8,legend=False)
    elif fig_type=='box':
        plt.figure()
        sns.boxplot(x=data_frame[x_axis],y=data_frame[y_axis])
    if save_fig:
        plt.savefig(output_name)

def preprocess_df2(data_frame, class_num=3,\
    col_to_drop=unrelated_columns, col_to_encode=target_columns,\
    col_to_be_unique=unique_columns):
    '''
    This function preprocess the protein sequence dataset.
    The preprocessing starts with picking the classes with the most counts.
    The unrelated columns are dropped.
    
    Arguments:
        data_frame (pandas.DataFrane): dataset
        class_num (int): how many classes to keep based on counts
        col_to_drop (list of str): which columns are not important
        col_to_encode (list of str): which columns have to be encoded
    
    Returns:
        data_cleaned (pandas.DataFrane): processed data for ML projects
    '''
    counts_val = sorted(data_frame.classification.value_counts())[::-1]
    counts = data_frame.classification.value_counts()
    types = np.asarray(counts[(counts > counts_val[class_num])].index)
    data = data_frame[data_frame.classification.isin(types)]
    data_cleaned = data.drop(col_to_drop, axis=1)
    data_cleaned = data_cleaned.drop_duplicates(subset=col_to_be_unique)
    data_cleaned = data_cleaned.dropna()
    for col in col_to_encode:
        data_cleaned[col] =\
        LabelEncoder().fit_transform(data_cleaned[col])

    return data_cleaned

