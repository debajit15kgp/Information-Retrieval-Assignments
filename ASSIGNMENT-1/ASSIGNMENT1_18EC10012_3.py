# importing libraries required
import nltk
# nltk.download('wordnet')
# nltk.download('stopwords')
import os
import json
import re
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords


# path of the documents
pwd = os.getcwd()

# making a class for preprocessing
class preprocess:
	def __init__(self,text,article_no):
		self.token_words = []
		self.text = text
		self.inv_pos_index = {}
		self.article_no = article_no

	def lower_case(self):
		self.text = self.text.lower() 

	def remove_punctuations(self):
		self.text = re.sub(r'[^\w\s]', '', self.text)

	def Word_Tokenize(self):
		self.token_words = word_tokenize(self.text)

	def remove_nonalpha(self):
		clean = []
		for word in self.token_words:
				if(word.isalpha()):
					clean.append(word)
		self.token_words = clean

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

	def stem_sentence(self):
		porter=PorterStemmer()
		stem_sentence=[]
		for word in self.token_words:
				stem_sentence.append(porter.stem(word))
		self.token_words = stem_sentence

	

	

# create dictionary
inv_pos_index = {}
permu_term = {}

def create_permuterm(val):
	permu_term[val] = [] 
	for length in range(0,len(val)+1):
		each = val[length:] + "$" + val[:length]
		permu_term[val].append(each)



if __name__=="__main__":
	
	inv_pos_index = {}

	for filename in sorted(os.listdir(pwd+"/ECTText/")): 

		article_no = filename.partition("-")[0].strip(".txt")
		text = open(pwd+"/ECTText/"+filename,'r').readlines()[0];

		# word preprocessing by creating a class 
		token = preprocess(text,article_no)

		token.lower_case()
		token.remove_punctuations()
		token.Word_Tokenize()
		token.remove_nonalpha()
		token.remove_stopwords()
		token.lemma_sentence()


		# build final index 
		for index,val in enumerate(token.token_words):
			if(val not in inv_pos_index.keys()):
				inv_pos_index[val] = {}
			if(article_no not in inv_pos_index[val].keys()):
				inv_pos_index[val][article_no] = []
			inv_pos_index[val][article_no].append(index)
		print(article_no)
		
		
	# creating the inverted_index dictionary
	with open(pwd+"/inverted_posting_index.json","w") as fp:
		json.dump(inv_pos_index,fp)
		


	
	
