##
# @author Abhinanda Ranjit Punnakkal <abhinanda.r.punnakkal@uit.no>
 # @file Description Morphological Analysis of mitochondria
 # @desc Created on 2022-09-28 1:21:03 pm
 # @copyright © 2021 apunnakkal <abhinanda.r.punnakkal@uit.no>


# Morphological Analysis of mitochondria


import numpy as np
import pandas as pd
import tqdm
from glob import glob
from skimage import morphology
import cv2
from matplotlib import pyplot as plt
from matplotlib import rcParams
from skan import Skeleton, summarize, draw
from skan import skeleton_to_csgraph
from skan.draw import overlay_euclidean_skeleton_2d as olp 
import ipdb
import seaborn as sns


data_dir = 'CCCP_data'
data_files = glob(data_dir+'\montaged_70_no_thresh/GAL*/*.png')
print(len(data_files))

sample_name_dict = {
    'GAL_CTRL': 'Control',
    'GAL_CCCP' : 'CCCP'
}

# Define constants
'''
 Branch types
# 0: endpoint-to-endpoint (isolated branch),  1: junction-to-endpoint,  2: junction-to-junction,  3: isolated cycle

Mito_type : 0- dot, 1- rod, 2 -  small net, 3 - big net

'''

MIN_SIZE = 100
MAX_DIAMETER = 1500 # max diameter of mitochondria in nm
pixel_size= 70      # (nm )

mito_dict = {
    0: 'Dot', 
    1 : 'Rod', 
    2 : 'Network',
    3 : 'Cluster'
}

cell_count = {
    'GAL_CCCP' : 60,
    'GAL_CTRL' : 63
}


def skeletonize(image):
    
    ret, binary=cv2.threshold(image, 127, 255, 0)
    binary[binary>0]=1
    skeleton = morphology.skeletonize(binary)
    return skeleton

################################################
# Skeletonize 
res = []
net_res = []
cluster_res = []
count_res = []
struct_ele_size = 4
input_threshold = 25
kernel = np.ones((struct_ele_size,struct_ele_size),np.uint8)
branches = []

USE_CLASS_CLUSTER_CLASS = False

branch_table = pd.DataFrame(columns=['Sample', 'File', 'branch-distance', 'Type'])

for file in tqdm.tqdm(data_files):
    base_name = file.split("\\")[-1].split('.')[0]
    file_name = file.split('-')[-1]
    img=cv2.imread(file, 0)
    img = cv2.erode(img,kernel,iterations = 1)
    skeleton = skeletonize(img)
    if np.sum(skeleton != 0) <= 1:
        img2 = skeleton.astype(float)
        img2 *= 255.0
        continue
    table = summarize(Skeleton(skeleton, spacing=pixel_size))
    mask0 = table['branch-type'] == 0
    mask1 = table['branch-type'] == 1
    table['branch-distance'] = np.where(mask1,  table['branch-distance']+2*struct_ele_size*pixel_size,  table['branch-distance'])
    table['branch-distance'] = np.where(mask0,  table['branch-distance']+struct_ele_size*pixel_size,  table['branch-distance'])
    dust = table.loc[(table['branch-distance'] < MIN_SIZE) & (table['branch-type'] == 0)]
    table = table.drop(dust.index).reset_index()
    unique_skeletons = np.unique(table['skeleton-id'])

    # Get all isolated branches
    unique_dots_rods = table[table['branch-type'] == 0]
    # Classify isolted branches based on distance
    unique_dot = unique_dots_rods[unique_dots_rods['branch-distance'] < MAX_DIAMETER]
    unique_dot.loc[:,'mito_type'] =  len(unique_dot) *[0] # dot
    num_dot = len(unique_dot)
    unique_rods = unique_dots_rods[unique_dots_rods['branch-distance'] >= MAX_DIAMETER]
    unique_rods.loc[:,'mito_type'] =  len(unique_rods) *[1]   # rod
    num_rod = len(unique_rods)
    mito_type_df = unique_dot.append(unique_rods)
    num_small_net, num_big_net, num_cluster = 0, 0, 0

    # Prep for branch analysis - Dot, rod
    temp = pd.DataFrame({ 'branch-distance' : unique_dot['branch-distance']})
    temp['Sample'] = len(temp)* [base_name ]
    temp['Type'] = len(temp) * ['Dot']
    temp['File'] = len(temp) * [file_name]
    branch_table = pd.concat([branch_table, temp]) 
    temp = pd.DataFrame({ 'branch-distance' : unique_rods['branch-distance']})
    temp['Sample'] = len(temp)* [base_name ]
    temp['Type'] = len(temp) * ['Rod']
    temp['File'] = len(temp) * [file_name]
    branch_table = pd.concat([branch_table, temp]) 
    
    # anything with a junction would be network
    net = table[table['branch-type'] == 1]
    net = net.append(table[table['branch-type'] == 2])
    net = net.append(table[table['branch-type'] == 3])
    net_skelton_ids = np.unique(net['skeleton-id'])

    # Prep for branch analysis - net
    temp = pd.DataFrame({ 'branch-distance' : net['branch-distance']})
    temp['Sample'] = len(temp)* [base_name]
    temp['Type'] = len(temp) * ['Net']
    temp['File'] = len(temp) * [file_name]
    branch_table = pd.concat([branch_table, temp]) 

    # Classify small networks seperately
    for s_id in net_skelton_ids:
        skel_branches = table[table['skeleton-id'] == s_id]   
        if skel_branches['branch-distance'].sum() < 1* MAX_DIAMETER: 
            skel_branches.loc[:,'mito_type'] =  len(skel_branches) *[0]   # small net
            num_small_net += 1
        else:    
            if USE_CLASS_CLUSTER_CLASS and len(skel_branches)> 10 :
                skel_branches.loc[:,'mito_type'] =  len(skel_branches) *[3]    # cluster
                num_cluster += 1
            else:    
                skel_branches.loc[:,'mito_type'] =  len(skel_branches) *[2]    # big net
                num_big_net += 1
        mito_type_df = mito_type_df.append(skel_branches)  

    assert (num_small_net + num_big_net + num_rod  + num_dot + num_cluster ) == len(unique_skeletons)
    
    for skel in unique_skeletons:
        skel_branches = mito_type_df[mito_type_df['skeleton-id']== skel]
        branch_length = skel_branches['branch-distance'].sum()
        assert len(np.unique(skel_branches['mito_type'])) == 1
        mt_type = int(np.unique(skel_branches['mito_type']))
        res.append([base_name, file_name, skel, branch_length, mt_type])
        if mt_type == 2:
                num_big_net += 1    
                net_res.append([base_name, file_name, skel, branch_length, mt_type])
        elif USE_CLASS_CLUSTER_CLASS & mt_type == 3 :
                cluster_res.append([base_name, file_name, skel, branch_length, mt_type])

    count_res.append([base_name, file_name, num_dot, num_rod, num_big_net, num_cluster])

df = pd.DataFrame(res, columns = ['Sample','file_name', 'skel_id','mito_length', 'Type'])
count_df = pd.DataFrame(count_res, columns = ['Sample','file_name', 'num_dot','num_rod', 'num_net', 'num_cluster'])
net_df = pd.DataFrame(net_res, columns = ['Sample','file_name', 'skel_id','mito_length', 'Type'])
if USE_CLASS_CLUSTER_CLASS : cluster_df = pd.DataFrame(cluster_res, columns = ['Sample','file_name', 'skel_id','mito_length', 'Type'])
df=df.replace({"mito_type": mito_dict})
df=df.replace({"Sample": sample_name_dict})
branch_table=branch_table.replace({"Sample": sample_name_dict})
branch_table= branch_table.rename(columns={"branch-distance": "branch_distance"})

branch_table_ref = branch_table[ (((branch_table['branch_distance'] >  250.)&(branch_table['Type'] == 'Net')) | branch_table['Type'] != 'Net') ]
branch_table_ref_summ = branch_table.groupby(['Sample','Type'])['branch_distance'].mean().reset_index()
branch_table_ref_summ['cell_count'] = branch_table_ref_summ['Sample']
branch_table_ref_summ=branch_table_ref_summ.replace({"cell_count": cell_count})

########################################################################33
### Branch length Analysis
plot = sns.catplot(
        data=branch_table_ref, x="Type", y = "branch_distance", hue="Sample", 
        legend_out=False,  height = 10, aspect = 1,
        kind='bar', hue_order = ['Control', 'CCCP'],
        row_order = ['Control', 'CCCP'])

plot.set_axis_labels("", "Mean branch length of Mitochondria (nm)")
titl = 'Mean branch length of Mitochondria'
# plt.title(titl)
plt.savefig(data_dir + '/' + titl.replace(' ', '_') +'.png' )

sum_df = df.groupby(by=['Sample','Type'])['mito_length'].aggregate(mito_length='sum', mean='mean', std='std', max='max' ).reset_index()

sum_df=sum_df.replace({"Type": mito_dict})
sum_df['perc'] = sum_df.groupby('Sample')['mito_length'].apply(lambda x: x*100/x.sum())
sum_df['mito_count'] = sum_df['Sample']
sum_df=sum_df.replace({"cell_count": cell_count})

#################################################333
### Percentage of individual mitochondrial lengths
f, axes = plt.subplots(1, 3, figsize=(10, 10))

sns.barplot(  y="perc", x= "Sample", data=sum_df[sum_df['Type']=='Dot'],  
    orient='v' , ax=axes[0], order=['Control', 'CCCP'] )
axes[0].set_xlabel("Dot")
axes[0].set_ylabel("Percentage of Mitochondria Length (%)")
sns.barplot(  y="perc", x= "Sample", data=sum_df[sum_df['Type']=='Rod'],  
    orient='v' , ax=axes[1], order=['Control', 'CCCP'])
axes[1].set_xlabel("Rod")
axes[1].set_ylabel("")
sns.barplot(  y="perc", x= "Sample", data=sum_df[sum_df['Type']=='Network'],  
    orient='v' , ax=axes[2], order=['Control', 'CCCP'])
axes[2].set_xlabel("Network")
axes[2].set_ylabel("")
titl = 'Precentage of Mitochondria lengths'
# plt.title(titl)
plt.savefig(data_dir + '/' + titl.replace(' ', '_') +'.png' )


########################################################33333333333333
### Statistical tests 
metric = 'branch'

if metric == 'branch': table_to_analyse = branch_table 
else: table_to_analyse = df

type = 'all'           
if type != 'all':
    table_to_analyse = table_to_analyse[table_to_analyse['Type']==type]

# variable_name = 'mito_length'
variable_name = 'branch_distance'

group_1 = 'GAL_CTRL'
group_2 = 'GAL_CCCP'


from scipy.stats import ttest_ind, f_oneway
gal_mitos_lengths = table_to_analyse[table_to_analyse['Sample']==group_1][variable_name]
glu_mitos_lengths = table_to_analyse[table_to_analyse['Sample']==group_2][variable_name]

print('Data', metric, 'Type:', type, 'Groups: ', group_1, group_2, )

print('----------------Student T- test')
stat, p = ttest_ind(gal_mitos_lengths,glu_mitos_lengths )
print('stat=%.3f, p=%.16f' % (stat, p))
if p > 0.05:
    print('Probably the same distribution')
else:
    print('Probably different distributions')
print('----------------Analysis of Variance Test (ANOVA) test')
stat, p = f_oneway(gal_mitos_lengths,glu_mitos_lengths )
print('stat=%.3f, p=%.16f' % (stat, p))
if p > 0.05:
    print('Probably the same distribution')
else:
    print('Probably different distributions')
    
