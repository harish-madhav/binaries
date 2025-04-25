import nltk
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('punkt_tab')
doc1=r"C:\Users\nagam\OneDrive\Desktop\Dei Parama Padii Da\Data Science\lb\doc1.txt"
doc2=r"C:\Users\nagam\OneDrive\Desktop\Dei Parama Padii Da\Data Science\lb\doc2.txt"
with open(doc1,'r',encoding='utf-8') as file:
    text1=file.read()
with open(doc2,'r',encoding='utf-8') as file:
    text2=file.read()
token1=word_tokenize(text1)
token2=word_tokenize(text2)
print("token1",token1)
print("token2",token2)

wc1=' '.join(token1)
wc2=' '.join(token2)
ans1=WordCloud(height=600,width=300,background_color='white').generate(wc1)
ans2=WordCloud(height=600,width=300,background_color='white').generate(wc2)

plt.subplot(1,2,1)
plt.imshow(ans1,interpolation='bilinear')
plt.axis('off')
plt.title("doc1")

plt.subplot(1,2,2)
plt.imshow(ans2,interpolation='bilinear')
plt.axis('off')
plt.title("doc2")
plt.tight_layout()
plt.show()
sia=SentimentIntensityAnalyzer()
s1=sia.polarity_scores(text1)
s2=sia.polarity_scores(text2)
print("sentiment1",s1)
print("sentiment2",s2)