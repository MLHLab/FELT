import numpy as np
import pandas as pd
import io
import os
import collections

!pip install sentence_transformers
!apt install libomp-dev
!pip install faiss-cpu

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sentence_transformers import SentenceTransformer
import faiss

from google.colab import files
uploaded = files.upload()

train_data = pd.read_csv(io.BytesIO(uploaded['Chk_TrainB (2).csv']), encoding='unicode_escape')
#train_data = train_data[train_data['Company'] == 'CITIBANK, N.A.']
print(train_data.shape)
print(train_data)

from google.colab import files
uploaded2 = files.upload()

test_data = pd.read_csv(io.BytesIO(uploaded2['Chk_Test.csv']), encoding='unicode_escape')
print(test_data.shape)
print(test_data)

print(train_data['Issue'].value_counts())
print(test_data['Issue'].value_counts())

model = SentenceTransformer("stsb-bert-base")

def text_to_vector(text):
    vector = model.encode(text)
    return vector

test_vectors = np.array([text_to_vector(text) for text in test_data['Consumer complaint narrative']])
faiss.normalize_L2(test_vectors)

train_vectors = np.array([text_to_vector(text) for text in train_data['Consumer complaint narrative']])
faiss.normalize_L2(train_vectors)

print(train_vectors.shape)
print(test_vectors.shape)

index = faiss.IndexFlatL2(train_vectors.shape[1])
index.add(train_vectors)

k = 1
distances, indices = index.search(test_vectors, k)

result_dft = pd.DataFrame(columns=['Test Index', 'actual_Issue', 'predicted_Issue', 'Distance'])

for i in range(len(test_data)):
    min_distance_index = indices[i][0]  # Index of nearest neighbor with smallest distance
    predicted_Issue = train_data.iloc[min_distance_index]['Issue']
    actual_Issue = test_data.iloc[i]['Issue']
    test_index = i
    distance = distances[i]
    result_dft = result_dft._append({'Test Index': test_index,
                                      'actual_Issue': actual_Issue,
                                      'predicted_Issue': predicted_Issue,
                                      'Distance': distance
                                    }, ignore_index=True)

display(result_dft)

print(result_dft.shape)
result_dft_cleaned = result_dft.dropna(subset=['actual_Issue'])
result_dft_cleaned = result_dft_cleaned.dropna(subset=['predicted_Issue'])
print(result_dft_cleaned.shape)

def all_metrics(dft,actual, predicted):
    y_true = dft[actual]
    y_pred = dft[predicted]

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred,average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred,average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred,average='weighted', zero_division=0)

    return {
        "Accuracy": round(accuracy,4),
        "Precision": round(precision,4),
        "Recall/ Sensetivity": round(recall,4),
        "f1" : round(f1,4)
    }

metrics1 = all_metrics(result_dft_cleaned,'actual_Issue', 'predicted_Issue')
display(metrics1)


