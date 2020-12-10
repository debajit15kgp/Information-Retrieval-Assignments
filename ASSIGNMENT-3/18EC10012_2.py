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

	def find_tfidf(self):
		vectorizer = TfidfVectorizer()
		tfidf_vec = vectorizer.fit(self.sentence_lists_train)
		self.tf_idf_train = tfidf_vec.transform(self.sentence_lists_train)
		self.tf_idf_test = tfidf_vec.transform(self.sentence_lists_test)

	def mod_mag(self,l1,l2):
		l3 = []
		for i in range(len(l1)):
			l3.append(l1[i]-l2[i])
		self.ans = 0
		for i in l3:
			self.ans += i*i
		self.ans = math.sqrt(self.ans)

	def find_rocchio_score(self):
		tfidf = self.tf_idf_train.toarray()
		self.centroids = []
		for i in range(2):
			centroid = [0 for x in range(len(tfidf[0]))]
			self.centroids.append(centroid)
		for i in range(len(self.sentence_label_train)):	
			if(self.sentence_label_train[i] == 0):
				for j in range(len(tfidf[i])):
					self.centroids[0][j] += tfidf[i][j]
			else:
				for j in range(len(tfidf[i])):
					self.centroids[1][j] += tfidf[i][j]
		for i in range(2):
			self.centroids[i] = [x/len(self.sentence_label_train) for x in self.centroids[i]]
		test_arr = self.tf_idf_test.toarray()
		for margin in self.margins:
			rocchio_y = []
			for i in range(len(test_arr)):
				
				self.mod_mag(test_arr[i],self.centroids[0])
				score_class1 = self.ans
				self.mod_mag(test_arr[i],self.centroids[1])
				score_class2 = self.ans
				if(score_class1<score_class2-margin):
					rocchio_y.append(0)
				else:
					rocchio_y.append(1)
			self.rocchio_score.append(f1_score(self.y_test,numpy.array(rocchio_y),average='macro'))


	def save_outputs(self,file_name):

		# for rocchio score
		fp = open(file_name,"w")
		name = ["NumFeature","Rocchio"]
		s1 = ["0",str(self.rocchio_score[0])]
		mname = len(max(name,key=len))
		ms1 = len(max(s1,key=len))
		for i in range(2):
			print("{0} {1}".format(name[i].ljust(mname+1),s1[i].ljust(ms1+1)))
		for i in range(2):
			fp.write(
	    		"{0} {1}".format(
	       			name[i].ljust(mname+1),
	        		s1[i].ljust(ms1+1)
	        	)
	    	)
			fp.write('\n')
		fp.close()



if __name__=="__main__":
	sys_args =(sys.argv)
	directory = sys_args[1]
	data = DATA_UTIL(directory)
	data.load_data_train()
	data.load_data_test()
	data.find_tfidf()
	data.find_rocchio_score()
	file_name = sys_args[2]
	data.save_outputs(file_name)