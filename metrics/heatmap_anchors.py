import os 
import matplotlib.pyplot as plt 
import numpy as np 
import json 
import seaborn as sns

#folder = '/media/lucap/big_data/datasets/repair/eccv_exp/quantitative_evaluation'
folder = 'output/ACCV24/maps_quantitative_evaluation'

heatmap_D = np.zeros((8,8))
heatmap_N = np.zeros((8,8))

for i in range(64):
    json_file = f"evaluation_anchor{i}_rot1.json"
    with open(os.path.join(folder, json_file), 'r') as jf:
        eval_i = json.load(jf)
    heatmap_D[i % 8, i // 8] = eval_i['correct']
    heatmap_N[i % 8, i // 8] = eval_i['neighbours']

plt.figure(figsize=(32,16))
plt.suptitle("Image 00003 (Barcelona), 64 Squared Pieces", fontsize=26)
plt.subplot(121)
plt.title("Direct Metric", fontsize=22)
sns.heatmap(heatmap_D, annot=True, square=True, cbar=False, cmap='RdYlGn', vmin=0)
#plt.axis('equal')
plt.subplot(122)
plt.title("Neighbours Metric", fontsize=22)
sns.heatmap(heatmap_N, annot=True, square=True, cbar=False, cmap='RdYlGn', vmin=0)
#plt.axis('equal')
plt.show()
