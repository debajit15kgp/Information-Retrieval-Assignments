import os
import os.path
from os import path
import sys
import re
import nltk
import pickle5 as pickle
import math
from io import StringIO
from html.parser import HTMLParser
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
#nltk.download('wordnet')
#nltk.download('stopwords')
from nltk.corpus import stopwords
import json

# to parse HTML files
class MLStripper(HTMLParser):
	def __init__(self):
		super().__init__()
		self.reset()
		self.strict = False
		self.convert_charrefs= True
		self.text = StringIO()
	def handle_data(self, d):
		self.text.write(d)
	def get_data(self):
		return self.text.getvalue()

# preprocess class
class preprocess:
	def __init__(self,text):
		self.token_words = []
		self.text = text
		self.inv_pos_index = {}    

	def lower_case(self):
		self.text = self.text.lower() 
		self.Word_Tokenize()

	def remove_punctuations(self):
		self.text = re.sub(r'[^\w\s]', '', self.text)

	def Word_Tokenize(self):
		self.token_words = word_tokenize(self.text)

	def remove_nonalpha(self):
		clean = []
		for word in self.token_words:
			new_word = ''.join([i for i in word if  i.isalpha()])
			clean.append(new_word)
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

	def auto_process(self):
		self.remove_punctuations()
		self.lower_case()
		self.remove_stopwords()
		self.remove_nonalpha()
		self.lemma_sentence()
		return self.token_words


class RANKED_RETRIEVAL_UTIL:
	def __init__(self,directory,num):
		self.directory = directory
		self.static_quality_scores = []
		self.leaders = []
		self.tokens = []
		self.inv_posindex_tf = {}
		self.inv_posindex_idf = {}
		self.text = ""
		self.champion_local = {}
		self.champion_global = {}
		self.total_files = N
		self.tokens_set = []
	
	# preproces using the class
	def preprocess_here(self):
		preprocess_util = preprocess(self.text)
		preprocess_util.auto_process()
		return preprocess_util.token_words

	# build tf files
	def build_posting_tf(self,article_no,tokens):
		for index,val in enumerate(tokens):
			tup = (article_no,math.log10(1+tokens.count(val)))
			if(val not in self.inv_posindex_tf.keys()):
				self.inv_posindex_tf[val] = []
			if(tup not in self.inv_posindex_tf[val]):
				self.inv_posindex_tf[val].append(tup)

	# build idf files
	def build_posting_idf(self,tokens,N):
		for index,val in enumerate(tokens):
			df = len(self.inv_posindex_tf[val])
			self.inv_posindex_idf[val] = math.log10(N/df)

	# load the data
	def load_data(self):
		if path.exists('tf.json') and path.exists('idf.json'):
			with open("tf.json", "r") as fp:
				self.inv_posindex_tf = json.load(fp)
			with open("idf.json", "r") as fp:
				self.inv_posindex_idf = json.load(fp)
		else:
			for filename in sorted(os.listdir(self.directory)):
				if (filename.endswith(".html")):
					doc = open(os.path.join(directory, filename)).read()
					html_util = MLStripper()
					html_util.feed(doc)
					self.text = html_util.get_data()
					tokens = self.preprocess_here()
					article_no = filename.strip(".html")
					self.build_posting_tf(article_no,tokens)
					self.build_posting_idf(tokens,len(os.listdir(self.directory)))
					self.tokens.extend(tokens)
					print(article_no)
				else:
					continue
			with open('tf.json', 'w') as handle:
				json.dump(self.inv_posindex_tf, handle)
			with open('idf.json', 'w') as handle:
				json.dump(self.inv_posindex_idf, handle)
			self.tokens_set = set(self.tokens)
			print(len(self.tokens_set))

	# read static scores
	def load_static_score(self):
		with open("../Dataset/StaticQualityScore.pkl", "rb") as fp:
			self.static_quality_scores = pickle.load(fp)

	# building local champion list
	def build_champion_local(self, N=50):
		if path.exists('champion_local.json'):
			with open("champion_local.json", "r") as fp:
				self.champion_local = json.load(fp)
		else:
			for index,val in enumerate(self.inv_posindex_tf.keys()):
				list_doc = []
				self.champion_local[val] = sorted(self.inv_posindex_tf[val], key = lambda x: x[1], reverse = True)[:N]
				for item in self.champion_local[val]:
					list_doc.append(item[0])
				self.champion_local[val] = list_doc
				print(index,len(self.champion_local[val]))
	
	# building the global champion list
	def build_champion_global(self, N=50):
		if path.exists('champion_global.json'):
			with open("champion_global.json", "r") as fp:
				self.champion_global = json.load(fp)
		else:
			for index,val in enumerate(self.inv_posindex_tf.keys()):
				list_doc = []
				self.champion_global[val] = sorted(self.inv_posindex_tf[val], key = lambda x: (x[1]*self.inv_posindex_idf[val]+self.static_quality_scores[int(x[0])]), reverse = True)[:N]
				for item in self.champion_global[val]:
					list_doc.append(item[0])
				self.champion_global[val] = list_doc
				print(index,len(self.champion_global[val]))

	# save the intermediate files as json
	def save_lists(self):
		if(path.exists('tf.json') and path.exists('idf.json') and path.exists('champion_global.json') and path.exists('champion_local.json')):
			pass
		else:
			with open('tf.json', 'w') as handle:
				json.dump(self.inv_posindex_tf, handle)
			with open('idf.json', 'w') as handle:
				json.dump(self.inv_posindex_idf, handle)
			with open('champion_local.json', 'w') as handle:
				json.dump(self.champion_local, handle)
			with open('champion_global.json', 'w') as handle:
				json.dump(self.champion_global, handle)


class SCORE_UTIL:

	# initialised the SCORE_UTIL class
	def __init__(self,tokens,N):
		self.inv_posindex_tf = {}
		self.inv_posindex_idf = {}
		self.champion_local = {}
		self.champion_global = {}
		self.tokens = tokens
		self.total_files = N
		self.tf_idf_scores = {}
		self.vq_idf = []
		self.leaders = []
		self.text = ""
		self.denom_score = {}
		for i in range(1000):
			self.denom_score[i] = 0

	# open the cached documents
	def load_cache(self):
		with open('tf.json', 'r') as fp:
			self.inv_posindex_tf = json.load(fp)
		with open('idf.json', 'r') as fp:
			self.inv_posindex_idf = json.load(fp)
		with open('champion_local.json', 'r') as fp:
			self.champion_local = json.load(fp)
		with open('champion_global.json', 'r') as fp:
			self.champion_global = json.load(fp)

	# load the leader.pkl file
	def load_leaders(self):
		with open("../Dataset/Leaders.pkl", "rb") as fp:
			self.leaders = pickle.load(fp)

	# calculate cosines similarity score with using norm of all terms
	def cosine_calculate(self,q,d,denom):
		eq,num = 0,0
		for x in range(len(self.tokens)):
			eq += q[x]*q[x]
			num += q[x]*d[x]
		print(eq,num,denom)
		try:
			cos = num/(math.sqrt(eq)*math.sqrt(denom))
			return cos
		except:
			return 0

	# calculate cosines similarity score for nearest leader
	def cosine_calculate_near(self,q,d):
		eq,ed,num = 0,0,0
		for x in range(len(self.tokens)):
			eq += q[x]*q[x]
			ed += d[x]*d[x]
			num += q[x]*d[x]
		try:
			cos = num/(math.sqrt(eq)*math.sqrt(ed))
			return cos
		except:
			return 0
		
	def tf_idf_score_util(self):
		try:

			# calculating the idf of query words
			self.vq_idf = []
			for x in self.tokens:
				if x in self.inv_posindex_idf.keys():
					self.vq_idf.append(self.inv_posindex_idf[x])
				else:
					self.vq_idf.append(0)

			vq_idf = self.vq_idf
			vd_tf_idf = []

			# calculating tf_idf_scores
			for val in self.inv_posindex_tf.keys():
				for item in self.inv_posindex_tf[val]:
					if(int(item[0]) not in self.tf_idf_scores.keys()):
						self.tf_idf_scores[int(item[0])] = {}
					self.tf_idf_scores[int(item[0])][val] = self.inv_posindex_idf[val]*item[1]
					self.denom_score[int(item[0])] += (self.tf_idf_scores[int(item[0])][val])**2
			
			# using tf_idf_scores to find the most appropiate docs
			now_list = []
			for doc_no in range(self.total_files):
				list_doc = []
				for x in self.tokens:
					try:
						list_doc.append(self.tf_idf_scores[doc_no][x])
					except:
						list_doc.append(0)
				now_list.append((self.cosine_calculate(vq_idf,list_doc,self.denom_score[doc_no]),doc_no))
			now_list.sort(reverse=True)
			text = ""
			for i in range(min(10,len(now_list))):
				text += "<"+str(now_list[i][1])+","+str(now_list[i][0])+">,"
			text = text.rstrip(",")
			self.text += text+'\n'
			print(text)
		except:
			self.text += '\n'

	def local_champion_score(self):
		try:

			# using local champion lists to calculate the list of documents
			docs = []
			for token in self.tokens:
				docs.extend(self.champion_local[token])
			docs = set(docs)
			docs = list(docs)
			now_list = []
			for doc_no in docs:
				list_doc = []
				for x in self.tokens:
					try:
						list_doc.append(self.tf_idf_scores[int(doc_no)][x])
					except:
						list_doc.append(0)
				now_list.append((self.cosine_calculate(self.vq_idf,list_doc,self.denom_score[int(doc_no)]),int(doc_no)))
			now_list.sort(reverse=True)
			text = ""
			for i in range(min(10,len(now_list))):
				text += "<"+str(now_list[i][1])+","+str(now_list[i][0])+">,"
			text = text.rstrip(",")
			self.text += text + '\n'
			print(text)
		except:
			self.text += '\n'

	def global_champion_score(self):
		try:

			# using global champion lists to calculate the list of documents
			docs = []
			for token in self.tokens:
				docs.extend(self.champion_global[token])
			docs = set(docs)
			docs = list(docs)
			now_list = []
			for doc_no in docs:
				list_doc = []
				for x in self.tokens:
					try:
						list_doc.append(self.tf_idf_scores[int(doc_no)][x])
					except:
						list_doc.append(0)
				now_list.append((self.cosine_calculate(self.vq_idf,list_doc,self.denom_score[int(doc_no)]),int(doc_no)))
			now_list.sort(reverse=True)
			text = ""
			for i in range(min(10,len(now_list))):
				text += "<"+str(now_list[i][1])+","+str(now_list[i][0])+">,"
			text = text.rstrip(",")
			self.text += text + '\n'
			print(text)
		except:
			self.text += '\n'

	def cluster_pruning_score(self):
		try:
			leaders_score = {}
			followers = {}
			for doc_no in self.leaders:
				followers[doc_no] = []
				list_doc = []
				for x in self.tokens:
					try:
						list_doc.append(self.tf_idf_scores[int(doc_no)][x])
					except:
						list_doc.append(0)
				leaders_score[doc_no] = list_doc

			# building list of followers
			for doc_no in range(self.total_files):
				if(doc_no in self.leaders):
					pass
				else:
					list_doc = []
					for x in self.tokens:
						try:
							list_doc.append(self.tf_idf_scores[int(doc_no)][x])
						except:
							list_doc.append(0)
					now_list = []
					for list_leader in leaders_score.keys():
						now_list.append((self.cosine_calculate_near(leaders_score[list_leader],list_doc),list_leader))
					now_list.sort(reverse=True)
					followers[now_list[0][1]].append(doc_no)
			
			# finding the closest leader
			now_list = []
			for list_leader in leaders_score.keys():
				now_list.append((self.cosine_calculate_near(leaders_score[list_leader],self.vq_idf),list_leader))
			now_list.sort(reverse=True)
			closest_leader = now_list[0][1]
			# creating union of closest leader and follower
			docs = [closest_leader]
			docs.extend(followers[closest_leader])
			print(docs)
			# finding the scores for the documents in the union
			now_list = []
			for doc_no in docs:
				list_doc = []
				for x in self.tokens:
					try:
						list_doc.append(self.tf_idf_scores[int(doc_no)][x])
					except:
						list_doc.append(0)
				print(self.vq_idf,list_doc,doc_no,self.denom_score[doc_no])
				now_list.append((self.cosine_calculate(self.vq_idf,list_doc,self.denom_score[doc_no]),doc_no))
			now_list.sort(reverse=True)
			text = ""
			for i in range(min(10,len(now_list))):
				text += "<"+str(now_list[i][1])+","+str(now_list[i][0])+">,"
			text = text.rstrip(",")
			self.text += text + '\n'
			print(text)
		except:
			self.text += '\n'

	def get_text(self):
		return self.text

if __name__=="__main__":
	
	directory = "../Dataset/Dataset"
	N = len(os.listdir(directory))
	denom_score = {}

	if(path.exists('tf.json') and path.exists('idf.json') and path.exists('champion_global.json') and path.exists('champion_local.json')):
		pass
	else:
		token = RANKED_RETRIEVAL_UTIL(directory,N)
		token.load_data()
		token.load_static_score()
		token.build_champion_local(50)
		token.build_champion_global(50)
		token.save_lists()
		

	sys_args =(sys.argv)
	queries = open(sys_args[1],'r').readlines()
	queries = [x.rstrip('\n') for x in queries]
	print(queries)
	text = ""
	for query_sent in queries:
		text += query_sent + '\n'
		query_util = preprocess(query_sent)
		query_words = query_util.auto_process()
		query_words = set(query_words)
		query_words = list(query_words)
		print(query_words)
		if(query_words == []):
			continue
		score_util  = SCORE_UTIL(query_words,N)
		score_util.load_cache()
		score_util.tf_idf_score_util()
		score_util.local_champion_score()
		score_util.global_champion_score()
		score_util.load_leaders()
		score_util.cluster_pruning_score()
		text += score_util.get_text() + '\n'
	
	with open("RESULTS2_18EC10012.txt","w") as fp:
		fp.write(text)