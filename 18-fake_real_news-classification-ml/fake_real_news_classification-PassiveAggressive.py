from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import itertools

"""### **Import Dataset**"""

df = pd.read_csv('fake_or_real_news.csv')

"""### **Data Summarization**"""

df.shape

df.head(5)

df = df.set_index('Unnamed: 0')

df.head(5)

"""### **Segregation of Input and Output**"""

#output
y = df.label
print(y)

#input
x = df.drop('label',axis=1)
print(x)

"""### **Splitting Dataset into Train Data and Test Data**"""

x_train, x_test, y_train, y_test = train_test_split(x['text'], y, test_size=0.33, random_state=53)

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)

tfidf_vectorizer.get_feature_names_out()[:10]

tfidf_df = pd.DataFrame(tfidf_train.A, columns = tfidf_vectorizer.get_feature_names_out())
print(tfidf_df)

tfidf_df.head()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          suptitle='Confusion Matrix',
                          cmap=plt.cm.Blues):
  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(suptitle)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes)
  plt.yticks(tick_marks, classes)

  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized Confusion Matrix")
  else:
    print('Confusion matrix, without normalization')

  thresh = cm.max()/2
  for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j,i,cm[i,j],
             horizontalalignment="center",
             color="white" if cm[i,j] > thresh else "black")

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.show()

model = PassiveAggressiveClassifier(max_iter=50)

model.fit(tfidf_train, y_train)

pred = model.predict(tfidf_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy : %0.3f"% score)

cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', "REAL"])
print(cm)
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])

myText = input("enter the input text: ")

tfidfvec_test = tfidf_vectorizer.transform([myText])

pred = model.predict(tfidfvec_test)
print(pred)
