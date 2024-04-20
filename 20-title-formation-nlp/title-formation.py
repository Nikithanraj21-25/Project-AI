import urllib.request #handling URL
from bs4 import BeautifulSoup #handling or parsing html files

import nltk #toolkit
nltk.download('stopwords')

from nltk.corpus import stopwords

#get the information from wesites
response = urllib.request.urlopen('https://en.wikipedia.org/wiki/steve_jobs')
html = response.read()
#print(html)

soup = BeautifulSoup(html,'html5lib')
text = soup.get_text(strip = True)
#print(text)

tokens = [t for t in text.split()]
#print(tokens)

#Removing stopwords
sr = stopwords.words('english')
clean_tokens = tokens[:]
for token in tokens:
       if token in sr:
              clean_tokens.remove(token)


freq = nltk.FreqDist(clean_tokens)
print(freq)
for key,val in freq.items():
       print(str(key) + ':' + str(val))
freq.plot(20, cumulative=False)


