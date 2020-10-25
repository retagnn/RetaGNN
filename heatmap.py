import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import seaborn as sns
import random

def HeatMap(square_array,name):
    square_array = square_array
    fig = plt.figure(name[0])
    value_list = list()
    new_square_array = list()
    for i in range(square_array.shape[0]):
        if i != 0:
            new_square_array.append([])
            for j in range(square_array.shape[1]-1):
                if i > j:
                    new_square_array[i-1].append(square_array[i,j])
                    value_list.append(square_array[i,j])
                    ##
                    # if i-1 == j :
                    #     noise = random.choice([0.05,0.1,0.15,0.2])
                    # else:
                    #     noise = random.choice([0.05,0.045,0.05,0.08])
                    # ss = square_array[i,j] + noise
                    # new_square_array[i-1].append(ss)
                    # value_list.append(ss)

                else:
                    new_square_array[i-1].append(0)
    max_value = max(value_list)
    new_square_array = np.array(new_square_array)
    row_col = new_square_array.shape[0]

    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['font.family']='sans-serif'
    matplotlib.rcParams['axes.unicode_minus'] = False

    Correlation = row_col
    plt.title('Amazon-Books')
    colormap = plt.cm.viridis

    sns.heatmap(new_square_array,vmax=max_value-0.2, square=True, cmap="YlGnBu",yticklabels=False,xticklabels=False)
    plt.savefig('/home/hsucheng/DRS/code/RS_2/heatmap/heatmap-'+str(name[0])+'-'+str(name[1])+'.png')
    plt.close(name[0])
    # plt.show()











