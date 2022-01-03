import requests
import spacy
from bs4 import BeautifulSoup
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('max_rows', 400)

NOUN1 = []
ADJECTIVES1 = []
VERBS1 = []

print("******************PART-1******************")
print("WEBSITE-1")

print("WEBPAGE-1 DATA")
URL1 = "http://nu.edu.pk/"
TEXT_WEB_SITE1_1 = requests.get(URL1)
TEXT1 = TEXT_WEB_SITE1_1.text

Soup = BeautifulSoup(TEXT1, 'html.parser')
W1_WEBPAGE1 = Soup.get_text()
print(W1_WEBPAGE1)

nlp = spacy.load('en_core_web_sm')

W1_1 = nlp(W1_WEBPAGE1)
for token in W1_1:
    if token.pos_ == 'NOUN':
        NOUN1.append(token.text)
for token in W1_1:
    if token.pos_ == 'ADJ':
        ADJECTIVES1.append(token.text)
for token in W1_1:
    if token.pos_ == 'VERB':
        VERBS1.append(token.text)

print("WEBPAGE-2 DATA")
URL2 = "http://nu.edu.pk/Degree-Programs"
TEXT_WEB_SITE1_2 = requests.get(URL2)
TEXT2 = TEXT_WEB_SITE1_2.text

Soup = BeautifulSoup(TEXT2, 'html.parser')
W1_WEBPAGE2 = Soup.get_text()
print(W1_WEBPAGE2)

nlp = spacy.load('en_core_web_sm')

W1_2 = nlp(W1_WEBPAGE2)
for token in W1_2:
    if token.pos_ == 'NOUN':
        NOUN1.append(token.text)
for token in W1_2:
    if token.pos_ == 'ADJ':
        ADJECTIVES1.append(token.text)
for token in W1_2:
    if token.pos_ == 'VERB':
        VERBS1.append(token.text)

print("WEBPAGE-3 DATA")
URL3 = "http://nu.edu.pk/University/History"
TEXT_WEB_SITE1_3 = requests.get(URL3)
TEXT3 = TEXT_WEB_SITE1_3.text

Soup = BeautifulSoup(TEXT3, 'html.parser')
W1_WEBPAGE3 = Soup.get_text()
print(W1_WEBPAGE3)

nlp = spacy.load('en_core_web_sm')

W1_3 = nlp(W1_WEBPAGE3)
for token in W1_3:
    if token.pos_ == 'NOUN':
        NOUN1.append(token.text)
for token in W1_3:
    if token.pos_ == 'ADJ':
        ADJECTIVES1.append(token.text)
for token in W1_3:
    if token.pos_ == 'VERB':
        VERBS1.append(token.text)

print("WEBPAGE-4 DATA")
URL4 = "http://nu.edu.pk/Career"
TEXT_WEB_SITE1_4 = requests.get(URL4)
TEXT4 = TEXT_WEB_SITE1_4.text

Soup = BeautifulSoup(TEXT4, 'html.parser')
W1_WEBPAGE4 = Soup.get_text()
print(W1_WEBPAGE4)

nlp = spacy.load('en_core_web_sm')

W1_4 = nlp(W1_WEBPAGE4)
for token in W1_4:
    if token.pos_ == 'NOUN':
        NOUN1.append(token.text)
for token in W1_4:
    if token.pos_ == 'ADJ':
        ADJECTIVES1.append(token.text)
for token in W1_4:
    if token.pos_ == 'VERB':
        VERBS1.append(token.text)

print("WEBPAGE-5 DATA")
URL5 = "http://nu.edu.pk/home/ContactUs"
TEXT_WEB_SITE1_5 = requests.get(URL5)
TEXT5 = TEXT_WEB_SITE1_5.text

Soup = BeautifulSoup(TEXT5, 'html.parser')
W1_WEBPAGE5 = Soup.get_text()
print(W1_WEBPAGE5)

nlp = spacy.load('en_core_web_sm')

W1_5 = nlp(W1_WEBPAGE5)
for token in W1_5:
    if token.pos_ == 'NOUN':
        NOUN1.append(token.text)
for token in W1_5:
    if token.pos_ == 'ADJ':
        ADJECTIVES1.append(token.text)
for token in W1_5:
    if token.pos_ == 'VERB':
        VERBS1.append(token.text)


NOUN2 = []
ADJECTIVES2 = []
VERBS2 = []

print("**************** WEBSITE-2 ****************")
print("WEBPAGE-1 DATA")
URL6 = "https://nust.edu.pk/"
TEXT_WEB_SITE2_1 = requests.get(URL6)
TEXT6 = TEXT_WEB_SITE2_1.text

Soup = BeautifulSoup(TEXT6, 'html.parser')
W2_WEBPAGE1 = Soup.get_text()
print(W2_WEBPAGE1)

nlp = spacy.load('en_core_web_sm')

W2_1 = nlp(W2_WEBPAGE1)
for token in W2_1:
    if token.pos_ == 'NOUN':
        NOUN2.append(token.text)
for token in W2_1:
    if token.pos_ == 'ADJ':
        ADJECTIVES2.append(token.text)
for token in W2_1:
    if token.pos_ == 'VERB':
        VERBS2.append(token.text)

print("WEBPAGE-2 DATA")
URL7 = "https://nust.edu.pk/admissions/"
TEXT_WEB_SITE2_2 = requests.get(URL7)
TEXT7 = TEXT_WEB_SITE2_2.text

Soup = BeautifulSoup(TEXT7, 'html.parser')
W2_WEBPAGE2 = Soup.get_text()
print(W2_WEBPAGE2)

nlp = spacy.load('en_core_web_sm')

W2_2 = nlp(W2_WEBPAGE2)
for token in W2_2:
    if token.pos_ == 'NOUN':
        NOUN2.append(token.text)
for token in W2_2:
    if token.pos_ == 'ADJ':
        ADJECTIVES2.append(token.text)
for token in W2_2:
    if token.pos_ == 'VERB':
        VERBS2.append(token.text)

print("WEBPAGE-3 DATA")
URL8 = "https://rni.nust.edu.pk/"
TEXT_WEB_SITE2_3 = requests.get(URL8)
TEXT8 = TEXT_WEB_SITE2_3.text

Soup = BeautifulSoup(TEXT8, 'html.parser')
W2_WEBPAGE3 = Soup.get_text()
print(W2_WEBPAGE3)

nlp = spacy.load('en_core_web_sm')

W2_3 = nlp(W2_WEBPAGE3)
for token in W2_3:
    if token.pos_ == 'NOUN':
        NOUN2.append(token.text)
for token in W2_3:
    if token.pos_ == 'ADJ':
        ADJECTIVES2.append(token.text)
for token in W2_3:
    if token.pos_ == 'VERB':
        VERBS2.append(token.text)

print("WEBPAGE-4 DATA")
URL9 = "https://nust.edu.pk/students/"
TEXT_WEB_SITE2_4 = requests.get(URL9)
TEXT9 = TEXT_WEB_SITE2_4.text

Soup = BeautifulSoup(TEXT9, 'html.parser')
W2_WEBPAGE4 = Soup.get_text()
print(W2_WEBPAGE4)

nlp = spacy.load('en_core_web_sm')

W2_4 = nlp(W2_WEBPAGE4)
for token in W2_4:
    if token.pos_ == 'NOUN':
        NOUN2.append(token.text)
for token in W2_4:
    if token.pos_ == 'ADJ':
        ADJECTIVES2.append(token.text)
for token in W2_4:
    if token.pos_ == 'VERB':
        VERBS2.append(token.text)

print("WEBPAGE-5 DATA")
URL10 = "https://nust.edu.pk/about-us"
TEXT_WEB_SITE2_5 = requests.get(URL10)
TEXT10 = TEXT_WEB_SITE2_5.text

Soup = BeautifulSoup(TEXT10, 'html.parser')
W2_WEBPAGE5 = Soup.get_text()
print(W2_WEBPAGE5)

nlp = spacy.load('en_core_web_sm')

W2_5 = nlp(W2_WEBPAGE5)
for token in W2_5:
    if token.pos_ == 'NOUN':
        NOUN2.append(token.text)
for token in W2_5:
    if token.pos_ == 'ADJ':
        ADJECTIVES2.append(token.text)
for token in W2_5:
    if token.pos_ == 'VERB':
        VERBS2.append(token.text)

NOUN3 = []
ADJECTIVES3 = []
VERBS3 = []

print("************ WEBSITE-3 ************")
print("WEBPAGE-1 DATA")
URL11 = "https://giki.edu.pk/"
TEXT_WEB_SITE3_1 = requests.get(URL11)
TEXT11 = TEXT_WEB_SITE3_1.text

Soup = BeautifulSoup(TEXT11, 'html.parser')
W3_WEBPAGE1 = Soup.get_text()
print(W3_WEBPAGE1)

nlp = spacy.load('en_core_web_sm')

W3_1 = nlp(W3_WEBPAGE1)
for token in W3_1:
    if token.pos_ == 'NOUN':
        NOUN3.append(token.text)
for token in W3_1:
    if token.pos_ == 'ADJ':
        ADJECTIVES3.append(token.text)
for token in W3_1:
    if token.pos_ == 'VERB':
        VERBS3.append(token.text)

print("WEBPAGE-3 DATA")
URL12 = "https://giki.edu.pk/vision-and-mission/"
TEXT_WEB_SITE3_2 = requests.get(URL12)
TEXT12 = TEXT_WEB_SITE3_2.text

Soup = BeautifulSoup(TEXT12, 'html.parser')
W3_WEBPAGE2 = Soup.get_text()
print(W3_WEBPAGE2)

nlp = spacy.load('en_core_web_sm')

W3_2 = nlp(W3_WEBPAGE2)
for token in W3_2:
    if token.pos_ == 'NOUN':
        NOUN3.append(token.text)
for token in W3_2:
    if token.pos_ == 'ADJ':
        ADJECTIVES3.append(token.text)
for token in W3_2:
    if token.pos_ == 'VERB':
        VERBS3.append(token.text)

print("WEBPAGE-3 DATA")
URL13 = "https://giki.edu.pk/admissions/"
TEXT_WEB_SITE3_3 = requests.get(URL13)
TEXT13 = TEXT_WEB_SITE3_3.text

Soup = BeautifulSoup(TEXT13, 'html.parser')
W3_WEBPAGE3 = Soup.get_text()
print(W3_WEBPAGE3)

nlp = spacy.load('en_core_web_sm')

W3_3 = nlp(W3_WEBPAGE3)
for token in W3_3:
    if token.pos_ == 'NOUN':
        NOUN3.append(token.text)
for token in W3_3:
    if token.pos_ == 'ADJ':
        ADJECTIVES3.append(token.text)
for token in W3_3:
    if token.pos_ == 'VERB':
        VERBS3.append(token.text)

print("WEBPAGE-4 DATA")
URL14 = "https://giki.edu.pk/scholarships/"
TEXT_WEB_SITE3_4 = requests.get(URL14)
TEXT14 = TEXT_WEB_SITE3_4.text

Soup = BeautifulSoup(TEXT14, 'html.parser')
W3_WEBPAGE4 = Soup.get_text()
print(W3_WEBPAGE4)

nlp = spacy.load('en_core_web_sm')

W3_4 = nlp(W3_WEBPAGE4)
for token in W3_4:
    if token.pos_ == 'NOUN':
        NOUN3.append(token.text)
for token in W3_4:
    if token.pos_ == 'ADJ':
        ADJECTIVES3.append(token.text)
for token in W3_4:
    if token.pos_ == 'VERB':
        VERBS3.append(token.text)

print("WEBPAGE-5 DATA")
URL15 = "https://giki.edu.pk/contact-us/"
TEXT_WEB_SITE3_5 = requests.get(URL15)
TEXT15 = TEXT_WEB_SITE3_5.text

Soup = BeautifulSoup(TEXT15, 'html.parser')
W3_WEBPAGE5 = Soup.get_text()
print(W3_WEBPAGE5)

nlp = spacy.load('en_core_web_sm')

W3_5 = nlp(W3_WEBPAGE5)
for token in W3_5:
    if token.pos_ == 'NOUN':
        NOUN3.append(token.text)
for token in W3_5:
    if token.pos_ == 'ADJ':
        ADJECTIVES3.append(token.text)
for token in W3_5:
    if token.pos_ == 'VERB':
        VERBS3.append(token.text)

# Creating three arrays based upon the lengths

print("******************PART-2******************")
print("++++ NOUNS WEBSITE 1")
print(NOUN1)
print("++++ NOUNS WEBSITE 2")
print(NOUN2)
print("++++ NOUNS WEBSITE 3")
print(NOUN3)
print("++++ ADJECTIVES WEBSITE 1")
print(ADJECTIVES1)
print("++++ ADJECTIVES WEBSITE 2")
print(ADJECTIVES2)
print("++++ ADJECTIVES WEBSITE 3")
print(ADJECTIVES3)
print("++++ VERBS WEBSITE 1")
print(VERBS1)
print("++++ VERBS WEBSITE 2")
print(VERBS2)
print("++++ VERBS WEBSITE 3")
print(VERBS3)

adjective = []
noun = []
verb = []

fastNoun = len(NOUN1)
fastAdjective = len(ADJECTIVES1)
fastVerb = len(VERBS1)
adjective.append(fastAdjective)
noun.append(fastNoun)
verb.append(fastVerb)

nustNoun = len(NOUN2)
nustAdjective = len(ADJECTIVES2)
nustVerb = len(VERBS2)
adjective.append(nustAdjective)
noun.append(nustNoun)
verb.append(nustVerb)


gikiNoun = len(NOUN3)
gikiAdjective = len(ADJECTIVES3)
gikiVerb = len(VERBS3)
adjective.append(gikiAdjective)
noun.append(gikiNoun)
verb.append(gikiVerb)


npFastNoun = np.array(fastNoun)
npFastAdjective = np.array(fastAdjective)
npFastVerb = np.array(fastVerb)

print(adjective)
print(noun)
print(verb)

xaxis = [1, 2, 3]
npNoun = np.array(noun)
npAdjective = np.array(adjective)
npVerb = np.array(verb)

# Visualisation for the Nouns
# Scatter Plot

plt.style.use("seaborn")
plt.plot(xaxis, noun)
plt.plot(xaxis, verb)
plt.plot(xaxis, adjective)
# plt.subplots(2,2)
plt.title("PLOTTING WRT NOUNS, VERBS AND ADJECTIVES")
plt.scatter(xaxis, noun, s=npNoun, c=["Red"])
# plt.scatter(xaxis, verb, s = npVerb , c = ["Blue"] )
# plt.scatter(xaxis, adjective, s = npAdjective , c = ["Yellow"] )
# plt.scatter()
plt.xticks([1, 2, 3], ["WEBSITE1", "WEBSITE2", "WEBSITE3"])
plt.yticks([0, 500, 1000, 1500, 2000, 2500, 3000, 3500])
plt.xlabel("WEBSITES")
plt.ylabel("NUMBER OF NOUNS, VERB AND ADJECTIVES")
plt.grid(True)

# plt.scatter(xaxis, noun, s = npNoun , c = ["Red"] )
plt.scatter(xaxis, verb, s=npVerb, c=["Blue"])
# plt.scatter(xaxis, adjective, s = npAdjective , c = ["Yellow"] )
# plt.scatter()
plt.xticks([1, 2, 3], ["WEBSITE1", "WEBSITE2", "WEBSITE3"])
plt.yticks([0, 500, 1000, 1500, 2000, 2500, 3000, 3500])
plt.xlabel("WEBSITES")
plt.ylabel("NUMBER OF NOUNS, VERB AND ADJECTIVES")
plt.grid(True)


# plt.scatter(xaxis, noun, s = npNoun , c = ["Red"] )
plt.scatter(xaxis, adjective, s=npAdjective/2, c=["Yellow"])
# plt.scatter(xaxis, adjective, s = npAdjective , c = ["Yellow"] )
# plt.scatter()
plt.xticks([1, 2, 3], ["WEBSITE1", "WEBSITE2", "WEBSITE3"])
plt.yticks([0, 500, 1000, 1500, 2000, 2500, 3000, 3500])
plt.xlabel("WEBSITES")
plt.ylabel("NUMBER OF NOUNS, VERB AND ADJECTIVES")
plt.grid(True)
