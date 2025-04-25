import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
df=pd.read_csv(r"C:\Users\subik\Documents\Datascience\parkinsons_disease_data_with_doctor_notes.csv")
num=df.select_dtypes(include='number')
scale=StandardScaler()
df_num=scale.fit_transform(num)
km=KMeans(n_clusters=2,random_state=42)
km_scale=km.fit_predict(df_num)
inertia=km.inertia_
sh=silhouette_score(df_num,km_scale)
print(f"{inertia}")
print(f"{sh}")
plt.scatter(df_num[:,0],df_num[:,1],c=km_scale,cmap="viridis")
plt.title("kmeans")
plt.xlabel("f1")
plt.ylabel("f2")
plt.show()