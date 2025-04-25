import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df=pd.read_csv("C:\\Users\\RAKESH\\Downloads\\IBM-HR-Analytics-Updated-With-DateTime.csv")

df.fillna(df.mean(numeric_only=True),inplace=True)

for col in df.select_dtypes(include=['object']).columns:
    df[col]=LabelEncoder().fit_transform(df[col])


scale=StandardScaler()
scaled_data=scale.fit_transform(df)

sse=[]
K=range(1,11)
for k in K:
    kmeans=KMeans(n_clusters=k,random_state=42)
    kmeans.fit(scaled_data)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(8,4))
plt.plot(K,sse,'bo-')
plt.xlabel('no of cluster')
plt.ylabel('sse(inertia)')
plt.title('elbow method for cluster calculation')
plt.grid(True)
plt.show()

k_opti=3

kmeans=KMeans(n_clusters=k_opti,random_state=42)
kmeans.fit(scaled_data)
labels=kmeans.labels_

score=silhouette_score(scaled_data,labels)
print('score',score)

df['cluster']=labels

plt.figure(figsize=(6,5))
sns.scatterplot(x=scaled_data[:,0],y=scaled_data[:,1],hue=labels,palette="Set2")
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.show()