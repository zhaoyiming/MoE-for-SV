import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 


cc = pd.read_csv("./show/01.csv")
cc=np.array(cc)
# sns.heatmap(cc, vmin=0, vmax=0.5)
# plt.savefig("./show/narrow_01.jpg")
# plt.show()

# for i in range(len(cc)):
#     for j in range(len(cc[0])):
#         if cc[i][j]>0 :
#             cc[i][j]=1
# sns.heatmap(cc, vmin=0, vmax=0.5)
# plt.savefig("./show/wide_bottom_01_gray.jpg")
# plt.show()


pos=0
neg=0
for i in range(len(cc)):
    for j in range(len(cc[0])):
        if cc[i][j]>0 :
            pos+=1
        else:
            neg+=1
print(neg/(pos+neg))
