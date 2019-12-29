import os
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from torch.utils.data import Dataset
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, KFold

    
from decimal import Decimal
import json 
from os import listdir
from os.path import isfile, join
import pandas as pd
import hashlib

class Writer():
    
    def __init__(self, prior_keyword=[], dir='./results'):
        self.prior_keyword = prior_keyword
        self.dir = dir
        
    def generate_hash(self, args):
        str_as_bytes = str.encode(str(args))
        hashed = hashlib.sha256(str_as_bytes).hexdigest()[:24]
        return hashed

    def write(self, args, prior_keyword=None):
        dict_args = vars(args)
        if 'bar' in dict_args:
            #del dict_args['bar']
            pass
        
        if prior_keyword:
            self.prior_keyword = prior_keyword
        filename = 'exp_{}'.format(args.exp_name)
        for keyword in self.prior_keyword:
            value = str(dict_args[keyword])
            if value.isdigit():
                filename += keyword + ':{:.2E}_'.format(Decimal(dict_args[keyword]))
            else:
                filename += keyword + ':{}_'.format(value)
#         hashcode = self.generate_hash(args)
#         filename += hashcode
        filename += '.json'
        
        with open(self.dir+'/'+filename, 'w') as outfile:
            json.dump(dict_args, outfile)
            
    def read(self, exp_name=''):
        list_result = list()
        filenames = [f for f in listdir(self.dir) if isfile(join(self.dir, f))]
        for filename in filenames:
            with open(join(self.dir, filename), 'r') as infile:
                result = json.load(infile)
                if len(exp_name) > 0:
                    if result['exp_name'] == exp_name:
                        list_result.append(result)
                else:
                    list_result.append(result)
                        
        return pd.DataFrame(list_result)
    
    def clear(self, exp_name=''):
        filenames = [f for f in listdir(self.dir) if isfile(join(self.dir, f))]
        for filename in filenames:
            if len(exp_name) > 0:
                result = json.load(open(join(self.dir, filename), 'r'))
                if result['exp_name'] == exp_name:
                    os.remove(join(self.dir, filename))
            else:
                os.remove(join(self.dir, filename))
                
                
                
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.font_manager import FontProperties
import seaborn as sns


def generate_setting(args, var1, var2):
    dict_args = vars(args)
    output = '{:92}'.format('[Exp Settings]') + '\n'
    output += '-'*91 + '\n'

    num_var = 3
    cnt_var = 0
    for keyword, value in dict_args.items():
        if keyword != var1 and keyword != var2 and type(value) != list and not 'best' in keyword and keyword != 'elapsed':
            str_value = str(value)
            if str_value.isdigit():
                if type(value) == float:
                    temp = '| {}={:.2E}'.format(keyword, Decimal(dict_args[keyword]))
                if type(value) == int:
                    temp = '| {}={}'.format(keyword, str_value[:15])

            else:
                temp = '| {}={}'.format(keyword, str_value[:15])
            output += '{:<30}'.format(temp[:30])
            cnt_var += 1
            if cnt_var % num_var == 0:
                cnt_var = 0
                output += '|\n'
                output += '-'*91 + '\n'
    return output

def plot_performance(results, variable1, variable2, args, title='', filename=''):
    fig, ax = plt.subplots(1, 2)

    fig.set_size_inches(15, 6)
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})
    sns.barplot(x=variable1, y='best_mae', hue=variable2, data=results, ax=ax[0])
    sns.barplot(x=variable1, y='best_std', hue=variable2, data=results, ax=ax[1])

    font = FontProperties()
    font.set_family('monospace')
    font.set_size('large')
    alignment = {'horizontalalignment': 'center', 'verticalalignment': 'baseline'}
    fig.text(0.5, -0.6, generate_setting(args, variable1, variable2), fontproperties=font, **alignment)
    
    fig.suptitle(title)
    filename = filename if len(filename) > 0 else title
    plt.savefig('./images/{}.png'.format(filename))
    
    
def plot_distribution(results, variable1, variable2, x='true_y', y='pred_y', title='', filename='', **kwargs):
    list_v1 = results[variable1].unique()
    list_v2 = results[variable2].unique()
    list_data = list()
    for value1 in list_v1:
        for value2 in list_v2:
            row = results.loc[results[variable1]==value1]
            row = row.loc[results[variable2]==value2]

            best_true_y = list(row.best_true_y)[0]
            best_pred_y = list(row.best_pred_y)[0]
            for i in range(len(best_true_y)):
                list_data.append({x:best_true_y[i], y:best_pred_y[i], variable1:value1, variable2:value2})
    df = pd.DataFrame(list_data)

    g = sns.FacetGrid(df, row=variable2, col=variable1, margin_titles=True)
    g.map(plt.scatter, x, y, alpha=0.3)

    def identity(**kwargs):
        plt.plot(np.linspace(-1.5,4,50), np.linspace(-1.5,4,50),'k',linestyle='dashed')
    g.map(identity)
    g.set_axis_labels(x, y)
    g.fig.suptitle(title) # can also get the figure from plt.gcf()
    plt.subplots_adjust(top=kwargs.get('top',0.93))
    filename = filename if len(filename) > 0 else title
    plt.savefig('./images/{}.png'.format(filename))
    
    
def plot_loss(results, variable1, variable2, x='true_y', y='pred_y', title='', filename='', **kwargs):
    list_v1 = results[variable1].unique()
    list_v2 = results[variable2].unique()
    list_data = list()
    for value1 in list_v1:
        for value2 in list_v2:
            row = results.loc[results[variable1]==value1]
            row = row.loc[results[variable2]==value2]

            train_losses = list(row.train_losses)[0]
            val_losses = list(row.val_losses)[0]
            maes = list(row.maes)[0]
            
            for item in train_losses:
                item.update({'type':'train', 'loss':item['train_loss'], variable1:value1, variable2:value2})
                
            for item in val_losses:
                item.update({'type':'val', 'loss':item['val_loss'], variable1:value1, variable2:value2})
            
            for item in maes:
                item.update({'type':'mae', variable1:value1, variable2:value2})
            list_data += train_losses + val_losses + maes

    df = pd.DataFrame(list_data)
    temp_mae = df.loc[df['mae'] < df['mae'].quantile(0.98)]
    ymax = temp_mae['mae'].max()
    ymin = temp_mae['mae'].min()
    
    temp_loss = df.loc[df['loss'] < df['loss'].quantile(0.98)]
    lossmax = temp_loss['loss'].max()
    lossmin = temp_loss['loss'].min()
    
    g = sns.FacetGrid(df, row=variable2, col=variable1, hue='type', margin_titles=False)
    axes = g.axes
    for i in range(len(axes)):
        for j in range(len(axes[0])):
            if i==0:
                g.axes[i][j].yaxis.set_label_coords(1.1,0.9)
                
    def mae_line(x, y, **kwargs):
        ax2 = plt.gca().twinx()
        ax2.plot(x, y,'g--')
        ax2.set_ylim(kwargs['ymax']*1.05, kwargs['ymin']*0.95)
        ax2.grid(False)

    g.map(plt.plot, x, y)
    g.map(mae_line, 'epoch', 'mae', ymin=ymin, ymax=ymax)
    g.set_axis_labels(x, y)
    g.fig.suptitle(title) # can also get the figure from plt.gcf()
    g.add_legend()
    
    for ax in g.axes.flatten():
        ax.set_ylim(lossmin, lossmax)
        
    plt.subplots_adjust(top=kwargs.get('top', 0.93))
    filename = filename if len(filename) > 0 else title
    plt.savefig('./images/{}.png'.format(filename))