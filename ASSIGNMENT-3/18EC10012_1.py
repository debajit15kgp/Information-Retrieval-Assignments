import os
import sys
import re
import nltk
import math
import numpy
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
#nltk.download('wordnet')
#nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score

# preprocess class
class preprocess:
	def __init__(self,text):
		self.token_words = []
		self.text = text
		self.inv_pos_index = {}  
		self.fin_text = ""  

	def lower_case(self):
		self.text = self.text.lower() 
		self.Word_Tokenize()

	def remove_punctuations(self):
		self.text = re.sub(r'[^\w\s]', ' ', self.text)

	def Word_Tokenize(self):
		self.token_words = word_tokenize(self.text)

	def remove_nonalpha(self):
		clean = []
		for word in self.token_words:
			new_word = ''.join([i for i in word if i.isalpha()])
			clean.append(new_word)
		self.token_words = clean

	def extra_process(self):
		self.text = self.text.replace('\\n',' ')

	def remove_stopwords(self):
		stop_words = stopwords.words('english')
		clean=[]
		for word in self.token_words:
			if not word in stop_words:
				clean.append(word)
		self.token_words = clean

	def lemma_sentence(self):
		wordnet_lemmatizer = WordNetLemmatizer()
		lemm_sentence=[]
		for word in self.token_words:
			lemm_sentence.append(wordnet_lemmatizer.lemmatize(word))
		self.token_words = lemm_sentence

	def token_to_text(self):
		for token in self.token_words:
			self.fin_text += token+' '
		self.fin_text.rstrip(' ')

	def auto_process(self):
		self.extra_process()
		self.remove_punctuations()
		self.lower_case()
		self.remove_stopwords()
		self.remove_nonalpha()
		self.lemma_sentence()
		self.token_to_text()

class DATA_UTIL:
	def __init__(self,directory):
		self.text = ""
		self.directory = directory
		self.sentence_lists_train = []
		self.sentence_label_train = []
		self.sentence_lists_test = []
		self.sentence_label_test = []
		self.get_text = ""
		self.topKlist = [1,10,100,1000,10000]
		self.X_new = []
		self.X_train = []
		self.y_train = []
		self.X_test = []
		self.y_test = []
		self.multi_score = []
		self.bernouli_score = []
		self.knn_score = []
		self.rocchio_score = []
		self.neighbors = [1,10,50]
		self.margins = [0]

	def preprocess_util(self,text):
		PRE_UTIL = preprocess(text)
		PRE_UTIL.auto_process()
		self.get_text = PRE_UTIL.fin_text

	def load_data_train(self):
		dir_1 = self.directory+"/class1/train"
		dir_2 = self.directory+"/class2/train"

		now_text = ""
		for file_name in sorted(os.listdir(dir_1)):
			with open(os.path.join(dir_1,file_name),"rb") as fp:
				self.preprocess_util(str(fp.read()))
				now_text = self.get_text
			self.sentence_lists_train.append(now_text)
			self.sentence_label_train.append(0)

		for file_name in sorted(os.listdir(dir_2)):
			with open(os.path.join(dir_2,file_name),"rb") as fp:
				self.preprocess_util(str(fp.read()))
				now_text = self.get_text
			self.sentence_lists_train.append(now_text)
			self.sentence_label_train.append(1)

		vectorizer = CountVectorizer()
		self.vec = vectorizer.fit(self.sentence_lists_train)
		self.X_train = self.vec.transform(self.sentence_lists_train)
		self.y_train = numpy.array(self.sentence_label_train)

	def load_data_test(self):
		dir_1 = self.directory+"/class1/test"
		dir_2 = self.directory+"/class2/test"

		now_text = ""
		for file_name in sorted(os.listdir(dir_1)):
			with open(os.path.join(dir_1,file_name),"rb") as fp:
				self.preprocess_util(str(fp.read()))
				now_text = self.get_text
			self.sentence_lists_test.append(now_text)
			self.sentence_label_test.append(0)

		for file_name in sorted(os.listdir(dir_2)):
			with open(os.path.join(dir_2,file_name),"rb") as fp:
				self.preprocess_util(str(fp.read()))
				now_text = self.get_text
			self.sentence_lists_test.append(now_text)
			self.sentence_label_test.append(1)

		self.X_test = self.vec.transform(self.sentence_lists_test)
		self.y_test = numpy.array(self.sentence_label_test)

	def find_naivebayes_score(self):
		multi_preds,bernouli_preds = [],[]
		for topK in self.topKlist:
			mnb = MultinomialNB(alpha=1.0)
			bnb = BernoulliNB()
			X = SelectKBest(mutual_info_classif,k=topK).fit(self.X_train,self.y_train)
			X_train_features = X.transform(self.X_train)
			X_test_features = X.transform(self.X_test)

			mnX = mnb.fit(X_train_features,self.y_train)
			multi_y = mnX.predict(X_test_features)
			self.multi_score.append(f1_score(self.y_test, multi_y, average='macro'))

			bnX = bnb.fit(X_train_features,self.y_train)
			bernouli_y = bnX.predict(X_test_features)
			self.bernouli_score.append(f1_score(self.y_test, bernouli_y, average='macro'))

	def save_outputs(self,file_name):

		 # for naive bayes score
		name = ["NumFeature","MultinomialNB","BernoulliNB"]
		s1 = ["1",str(self.multi_score[0]),str(self.bernouli_score[0])]
		s10 = ["10",str(self.multi_score[1]),str(self.bernouli_score[1])]
		s100 = ["100",str(self.multi_score[2]),str(self.bernouli_score[2])]
		s1000 = ["1000",str(self.multi_score[3]),str(self.bernouli_score[3])]
		s10000 = ["10000",str(self.multi_score[4]),str(self.bernouli_score[4])]
		mname = len(max(name,key=len))
		ms1 = len(max(s1,key=len))
		ms10 = len(max(s10,key=len))
		ms100 = len(max(s100,key=len))
		ms1000 = len(max(s1000,key=len))
		ms10000 = len(max(s10000,key=len))
		fp = open(file_name,"w")
		for i in range(3):
			print("{0} {1} {2} {3} {4} {5}".format(name[i].ljust(mname+1),s1[i].ljust(ms1+1),s10[i].ljust(ms10+1),s100[i].ljust(ms100+1),s1000[i].ljust(ms1000+1),s10000[i].ljust(ms10000+1)))
		for i in range(3):
			fp.write(
	    		"{0} {1} {2} {3} {4} {5}".format(
	       			name[i].ljust(mname+1),
	        		s1[i].ljust(ms1+1),
	        		s10[i].ljust(ms10+1),
	        		s100[i].ljust(ms100+1),
	        		s1000[i].ljust(ms1000+1),
	        		s10000[i].ljust(ms10000+1)
	        	)
	    	)
			fp.write('\n')
		fp.close()



if __name__=="__main__":
	sys_args =(sys.argv)
	path = sys_args[1]
	directory = sys_args[1]
	data = DATA_UTIL(directory)
	data.load_data_train()
	data.load_data_test()
	data.find_naivebayes_score()
	file_name = sys_args[2]
	data.save_outputs(file_name)
	