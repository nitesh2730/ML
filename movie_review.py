import nltk.classify.util
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier

def create_features_list(words):
    useful_words=[word for word in words if word not in stopwords.words("english")]
    dict_word=dict([(word,True) for word in useful_words])
    return dict_word

neg_review=[]
for fileid in movie_reviews.fileids('neg'):
    words=movie_reviews.words(fileid)
    neg_review.append((create_features_list(words),"negative"))
    #print(neg_review[0])
    #print(len(neg_review))

pos_review=[]
for fileid in movie_reviews.fileids('pos'):
    words=movie_reviews.words(fileid)
    pos_review.append((create_features_list(words),"positive"))
    
train_set = neg_review[:750] + pos_review[:750]
test_set =  neg_review[750:] + pos_review[750:]

classifier = NaiveBayesClassifier.train(train_set)

accuracy = nltk.classify.util.accuracy(classifier, test_set)
print(accuracy * 100)

review_santa = '''
 
It would be impossible to sum up all the stuff that sucks about this film, so I'll break it down into what I remember most strongly: a man in an ingeniously fake-looking polar bear costume (funnier than the "bear" from Hercules in New York); an extra with the most unnatural laugh you're ever likely to hear; an ex-dope addict martian with tics; kid actors who make sure every syllable of their lines are slowly and caaarreee-fulll-yyy prrooo-noun-ceeed; a newspaper headline stating that Santa's been "kidnaped", and a giant robot. Yes, you read that right. A giant robot.
 
The worst acting job in here must be when Mother Claus and her elves have been "frozen" by the "Martians'" weapons. Could they be *more* trembling? I know this was the sixties and everyone was doped up, but still.
'''

words = word_tokenize(review_santa)
words = create_features_list(words)
classifier.classify(words)
