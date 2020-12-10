# importing libraries required
import nltk
import os
import json
import argparse
import sys


# using binary seach of token list and reverse token list for to find in O(log(M))

# path of the documents
pwd = os.getcwd()

inv_pos_index = {}
final_text = ""

# finding the lower_bound in sorted list of tokens
def find_lowerbound(L, target):
	start = 0
	end = len(L) - 1

	while start <= end:
		middle = int((start + end)/ 2)
		midpoint = L[middle]
		if midpoint >= target:
			end = middle - 1
		else:
			start = middle + 1
	return start

# finding the upper_bound in sorted list of tokens
def find_upperbound(L, target):
	start = 0
	end = len(L) - 1

	while start <= end:
		middle = int((start + end)/ 2)
		midpoint = L[middle]
		if midpoint > target:
			end = middle - 1
		else:
			start = middle + 1
	return start

# finding the next word
def nextWord(s): 
    if (s == " "): 
        return "a"
    i = len(s) - 1
    while (s[i] == 'z' and i >= 0): 
        i -= 1
    if (i == -1): 
        s = s + 'a'
    else: 
        l = list(s)
        l[i] = chr(ord(l[i])+1)
        s = "".join(l) 
    return s 

def intersection(lst1, lst2): 
    return list(set(lst1) & set(lst2)) 

def print_tolerant(lst):
	text = ""
	print(len(lst))
	for word in lst:
		print(word,end = " ")
		print()
		text += word+":"
		for article in inv_pos_index[word].keys():
			lst1=[]
			for index in inv_pos_index[word][article]:
				text += "<"+article+","+str(index)+">"+","
				lst1.append(index)
		text = text.rstrip(",")
		text += ";"
	#print(text)
	text = text.rstrip(";")
	text += "\n" 
	global final_text
	final_text += text
	

if __name__=="__main__":

	# retriving the postng list
	sys_args =(sys.argv)
	with open("inverted_posting_index.json", 'r') as fp:
		inv_pos_index = json.load(fp)

	# opeing query.txt
	queries = open(sys_args[1],'r').readlines()

	# making sorted token list as well as reverse token list
	tokens_list,tokens_list_reverse = [],[]
	for key in inv_pos_index.keys():
		tokens_list.append(key)
		tokens_list_reverse.append(key[::-1])
	tokens_list.sort()
	tokens_list_reverse.sort()

	# iterating for each query word
	for query_word in queries:
		if(query_word[-1]=='\n'):
			query_word = query_word.rstrip('\n')

		# if query word ends with *
		if(query_word[-1]=="*"):

			l_ind = find_lowerbound(tokens_list, query_word[:-1])
			u_ind = find_lowerbound(tokens_list, nextWord(query_word[:-1]))
			lst = []
			for i in range(l_ind,u_ind):
				lst.append(tokens_list[i])
			print_tolerant(lst)

		# if query word begin with *
		elif(query_word[0]=="*"):

			l_ind = find_lowerbound(tokens_list_reverse, query_word[1:][::-1])
			u_ind = find_lowerbound(tokens_list_reverse, nextWord(query_word[1:][::-1]))
			lst = []
			for i in range(l_ind,u_ind):
				lst.append(tokens_list_reverse[i][::-1])
			print_tolerant(lst)

		# if * is in the middle of the query word
		else:
			index = query_word.find('*')
			f_word = query_word[:index]
			l_word = query_word[index+1:]
			l_ind1 = find_lowerbound(tokens_list, f_word)
			u_ind1 = find_lowerbound(tokens_list, nextWord(f_word))
			l_ind2 = find_lowerbound(tokens_list_reverse, l_word[::-1])
			u_ind2 = find_lowerbound(tokens_list_reverse, nextWord(l_word[::-1]))
			lst1,lst2 = [],[]

			for i in range(l_ind1,u_ind1):
				lst1.append(tokens_list[i])
			for i in range(l_ind2,u_ind2):
				lst2.append(tokens_list_reverse[i][::-1])
			print_tolerant(intersection(lst1,lst2))		

	# printing results to output file
	with open("RESULTS1_18EC10012.txt",'w') as fp:
		fp.write(final_text)
