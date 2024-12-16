
from scipy.cluster.hierarchy import dendrogram, linkage, to_tree
from scipy.spatial.distance import squareform
import seaborn as sns
from highcharts import Highchart
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
import matplotlib as mpl

from feamap.utils.logtools import print_info
from feamap.preparation import factor_int



def plot_scatter(molmap, htmlpath = './', htmlname = None, radius = 3):
    '''
    molmap: the object of molmap
    htmlpath: the figure path, not include the prefix of 'html'
    htmlname: the name 
    radius: int, defaut:3, the radius of scatter dot
    '''
    
    title = '2D emmbedding of %s based on %s method' % (molmap.ftype, molmap.method)
    subtitle = 'number of %s: %s, metric method: %s' % (molmap.ftype, len(molmap.flist), molmap.metric)
    name = '%s_%s_%s_%s_%s' % (molmap.ftype,len(molmap.flist), molmap.metric, molmap.method, 'scatter')
    
    if not os.path.exists(htmlpath):
        os.makedirs(htmlpath)
    
    if htmlname:
        name = htmlname + '_' + name 
        
    filename = os.path.join(htmlpath, name)
    print_info('generate file: %s' % filename)
        
    
    xy = molmap.embedded.embedding_
    # colormaps = molmap.extract.colormaps
    colormaps = molmap.colormaps
    
    df = pd.DataFrame(xy, columns = ['x', 'y'])
    # bitsinfo = molmap.extract.bitsinfo.set_index('IDs')
    bitsinfo = molmap.bitsinfo.set_index('IDs')
    df = df.join(bitsinfo.loc[molmap.flist].reset_index())
    # df['colors'] = df['Subtypes'].map(colormaps)

    channels = molmap.channel_col
    df_grid = df.sort_values(['y', 'x']).reset_index(drop=True)
    mp_colors = colormaps
    
    figsize_m, figsize_n = factor_int(len(channels))
    fig, axes =  plt.subplots(figsize_m, figsize_n, figsize=(figsize_m, figsize_n))
    axe = axes.ravel()
    for i, channel in enumerate(channels):
        color = mp_colors[channel]
        data = df_grid.loc[df_grid[channel]==1]
        sns.scatterplot(x='x', y='y', data=data, color = color, ax=axe[i])# cbar_kws = dict(use_gridspec=False,location="top")
        axe[i].axhline(y=0, color='grey',lw=0.2, ls =  '--')
        axe[i].axvline(x=data.shape[1], color='grey',lw=0.2, ls =  '--')
        axe[i].axhline(y=data.shape[0], color='grey',lw=0.2, ls =  '--')
        axe[i].axvline(x=0, color='grey',lw=0.2, ls =  '--')

    for i, e_c in enumerate(range((figsize_m*figsize_n)-len(channels))):
        axe[(figsize_m*figsize_n)-1-i].remove()

    patches = [plt.plot([],[], marker="s", ms=8, ls="", mec=None, color=mp_colors[k], label=k)[0]  for k in channels]    
    l = 1.32
    # if mp_d.ftype == 'fingerprint':
    #     l += 0.05
    # plt.legend(handles=patches, bbox_to_anchor=(l,1.01), loc='upper right', ncol=1, facecolor="w", numpoints=1,fontsize=1 )    
    
    #plt.tight_layout()
    # plt.savefig(f'{filename}.tif', bbox_inches='tight', dpi = 600 ,format='tif')
    # print_info('save file to %s' % filename)
    plt.close()

    return df, fig



def plot_grid(molmap, htmlpath = './', htmlname = None):
    '''
    molmap: the object of molmap
    htmlpath: the figure path
    '''    

    if not os.path.exists(htmlpath):
        os.makedirs(htmlpath)    
    
    title = 'Assignment of %s by %s emmbedding result' % (molmap.ftype, molmap.method)
    subtitle = 'number of %s: %s, metric method: %s' % (molmap.ftype, len(molmap.flist), molmap.metric)    

    name = '%s_%s_%s_%s_%s' % (molmap.ftype,len(molmap.flist), molmap.metric, molmap.method, 'molmap')
    
    if htmlname:
        name = name = htmlname + '_' + name   
    
    filename = os.path.join(htmlpath, name)
    print_info('generate file: %s' % filename)
    
    
    
    m,n = molmap.fmap_shape
    # colormaps = molmap.extract.colormaps
    colormaps = molmap.colormaps
    position = np.zeros(molmap.fmap_shape, dtype='O').reshape(m*n,)
    position[molmap._S.col_asses] = molmap.flist
    position = position.reshape(m, n)
    

    
    x = []
    for i in range(n):
        x.extend([i]*m)
        
    y = list(range(m))*n
        
        
    v = position.reshape(m*n, order = 'f')

    df = pd.DataFrame(list(zip(x,y, v)), columns = ['x', 'y', 'v'])
    # bitsinfo = molmap.extract.bitsinfo
    bitsinfo = molmap.bitsinfo
    subtypedict = bitsinfo.set_index('IDs')['Subtypes'].to_dict()
    subtypedict.update({0:'NaN'})
    df['Subtypes'] = df.v.map(subtypedict)
    # df['colors'] = df['Subtypes'].map(colormaps)
    # df['colors'] = pd.Series([i.str.cat(sep=',') for idx,i in pd.DataFrame.from_dict({col:s.map(colormaps) for col, s in bitsinfo['Subtypes'].str.split(',', expand=True).iteritems()}).iterrows()])

    
    channels = molmap.channel_col
    df = df.join(bitsinfo.set_index('IDs')[channels], on='v')
    df_grid = df.sort_values(['y', 'x']).reset_index(drop=True)
    mp_colors = colormaps
    
    figsize_m, figsize_n = factor_int(len(channels))
    
    fig, axes =  plt.subplots(figsize_m, figsize_n, figsize=(figsize_m, figsize_n))
    try: axe = axes.ravel()
    except: axe = [axes]
    cbars = {}
    for i, channel in enumerate(channels):
        color = mp_colors[channel]
        ys_singlechannel = df_grid.loc[df_grid['Subtypes']==channel].y
        xs_singlechannel = df_grid.loc[df_grid['Subtypes']==channel].x
        ys_multichannel = df_grid.loc[(df_grid[channel]==1) & ~(df_grid['Subtypes']==channel)].y
        xs_multichannel = df_grid.loc[(df_grid[channel]==1) & ~(df_grid['Subtypes']==channel)].x
        data = -1 * np.ones((m,n))
        data[ys_singlechannel,xs_singlechannel] = np.ones((m,n))[ys_singlechannel,xs_singlechannel]
        # data[ys_multichannel,xs_multichannel] = np.zeros((m,n))[ys_multichannel,xs_multichannel]
        data[ys_multichannel,xs_multichannel] = np.ones((m,n))[ys_multichannel,xs_multichannel]
        # data[ys,xs] = np.ones((m,n))[ys,xs]
        cmap = sns.light_palette(color, n_colors = 3, reverse=False)
        cbars[channel]=cmap
        sns.heatmap(np.where(data !=-1, data, np.nan), cmap = cmap, vmax=1, vmin=0, yticklabels=False, xticklabels=False, cbar=False, linewidths=0, ax=axe[i])# cbar_kws = dict(use_gridspec=False,location="top")
        axe[i].axhline(y=0, color='grey',lw=0.2, ls = '--')
        axe[i].axvline(x=data.shape[1], color='grey',lw=0.2, ls = '--')
        axe[i].axhline(y=data.shape[0], color='grey',lw=0.2, ls = '--')
        axe[i].axvline(x=0, color='grey',lw=0.2, ls = '--')
    
    for i, e_c in enumerate(range((figsize_m*figsize_n)-len(channels))):
        axe[(figsize_m*figsize_n)-1-i].remove()


    patches = [plt.plot([],[], marker="s", ms=8, ls="", mec=None, color=mp_colors[k], label=k)[0]  for k in channels]    
    l = 1.32
    # if mp_d.ftype == 'fingerprint':
    #     l += 0.05
    # plt.legend(handles=patches, bbox_to_anchor=(l,1.01), loc='upper right', ncol=1, facecolor="w", numpoints=1,fontsize=6 )    
    
    #plt.tight_layout()
    # plt.savefig(f'{filename}.tif', bbox_inches='tight', dpi = 600 ,format='tif')
    # print_info('save file to %s' % filename)
    plt.close()

    return df, fig



def _getNewick(node, newick, parentdist, leaf_names):
    if node.is_leaf():
        return "%s:%.2f%s" % (leaf_names[node.id], parentdist - node.dist, newick)
    else:
        if len(newick) > 0:
            newick = "):%.2f%s" % (parentdist - node.dist, newick)
        else:
            newick = ");"
        newick = _getNewick(node.get_left(), newick, node.dist, leaf_names)
        newick = _getNewick(node.get_right(), ",%s" % (newick), node.dist, leaf_names)
        newick = "(%s" % (newick)
        return newick
    
def _mp2newick(molmap, treefile = 'mytree'):

    dist_matrix = molmap.dist_matrix
    leaf_names = molmap.flist
    df = molmap.df_embedding[['colors','Subtypes']]
    
    dists = squareform(dist_matrix)
    linkage_matrix = linkage(dists, 'complete')
    tree = to_tree(linkage_matrix, rd=False)
    newick = getNewick(tree, "", tree.dist, leaf_names = leaf_names)
    
    with open(treefile + '.nwk', 'w') as f:
        f.write(newick)
    df.to_excel(treefile + '.xlsx')
    
        
def plot_tree(molmap, htmlpath = './', htmlname = None):
    pass