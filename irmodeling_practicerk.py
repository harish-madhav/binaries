#pip install textblob
#pip install scikit-learn
#pip install python-docx
# download python -m textblob.download_corpora


from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from docx import Document
import os

def extract_text(filepath):
    if not os.path.exists(filepath):
        print("not found")
        return ""
    doc=Document(filepath)
    return '\n'.join([para.text for para in doc.paragraphs])

def find_sentiment(text1):
    blob=TextBlob(text1)
    return blob.sentiment.polarity,blob.sentiment.subjectivity


def compute_sim(text1,text2):
    vector=TfidfVectorizer()
    vectors=vector.fit_transform([text1,text2])
    return cosine_similarity(vectors[0:1],vectors[1:2])[0][0]

file1='test1.docx'
file2='test2.docx'

text1=extract_text(file1)
text2=extract_text(file2)

polar,subj=find_sentiment(text1)
print("polarity",polar)
print("subjectivity",subj)

sim=compute_sim(text1,text2)
print("similarity",sim)

