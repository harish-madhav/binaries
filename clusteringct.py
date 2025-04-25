import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage,dendrogram,fcluster
import matplotlib.pyplot as plt
df=pd.read_csv(r"C:\Users\subik\Documents\Datascience\parkinsons_disease_data_with_doctor_notes.csv")
num=df.select_dtypes(include='number')
scale=StandardScaler()
df_num=scale.fit_transform(num)
linked=linkage(df_num,method='ward')
plt.figure(figsize=(16,10))
dendrogram(linked,truncate_mode='lastp',p=30,leaf_rotation=90.,leaf_font_size=12.,show_contracted=True)
plt.title("dendrogram")
plt.xlabel("cluster size")
plt.ylabel("distance")
plt.show()