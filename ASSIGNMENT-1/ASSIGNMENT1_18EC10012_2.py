# importing libraries required
import os
from bs4 import BeautifulSoup
import datefinder
import re
import json

# path of the documents
pwd = os.getcwd()

def buildECTNestedDict():
	cnt = 0
	for filename in sorted(os.listdir(pwd+"/ECT/")):
		# main dictionary which will later be saved as ECTNestedDict
		main_dict = {}
		try:
			# opening all the files ending with html
			if filename.endswith(".html"):

				# getting the page data as text in a variable
				doc = open(pwd+"/ECT/"+filename).read()

				# getting the article number which is the key in main_dict
				article_no = filename.partition("-")[0]
				
				# opening using bs4
				soup = BeautifulSoup(doc,features="lxml")
				
				# just for better visibility
				# soup.prettify()

				# open all div tags with id a-body
				v1 = soup.find(id="a-body")
				tags = v1.find_all(class_="p p1")
				
				# extracting the date as it is usually in the first line
				first_line = tags[0].get_text()
				date = re.findall('.*((January|February|March|April|June|July|August|September|October|November|December)+.+ET)', first_line)

				# finding the Participants
				speakers,desig = ["Operator"],["Operator"]
				track = 0
				for it in tags[1:]:
					text = it.get_text()
					if (it.find("strong")): 
						track += 1
						continue
					if(track == 3):
						break
					speaker = text.partition(" -")[0]
					speakers.append(speaker)
					desig.append(text)


				# initialising variables
				serial_num = 0
				speaker = ""
				main_dict["Date"] = date[0][0]
				main_dict["Participants"] = desig
				main_dict["Presentation"] = {}
				main_dict["Questionnaire"] = {}

				# making the dictionary of Presentation and questionnaire from a-body
				flag_ques = True
				for item in tags[len(speakers)+track-1:]:
					text = item.get_text()
					text = text.rstrip(" ")

					# if we are still checking the body
					if(flag_ques):
						if(text == "Question-and-Answer Session"):
								flag_ques = False
								continue

						# if the text is a heading it must be a speaker
						if (item.find("strong") or (text in speakers)):	
							speaker = text.rstrip(" ")
							if text not in main_dict["Presentation"]:
								main_dict["Presentation"][text] = []
							continue
						main_dict["Presentation"][speaker].append(text)
					else:

						# Questionnaire is detected
						if (item.find("strong") or (text in speakers) or text.startswith("A -") or text.startswith("Q -")):
							serial_num += 1
							main_dict["Questionnaire"][serial_num] = {}
							main_dict["Questionnaire"][serial_num]["Speaker"] = text
							main_dict["Questionnaire"][serial_num]["Remark"] = []
							continue
						main_dict["Questionnaire"][serial_num]["Remark"].append(text)	



				# making the dictionary of Presentation and Questionnaire after a-body
				v2 = soup.find_all(class_="p_count")

				for num in range(2,len(v2)+2):
					now = v1.find_all(class_="p p%d"%num)

					for html_text in now:

						# if we are still checking the body
						if(flag_ques):
							text = html_text.get_text()
							text = text.rstrip(" ")
							if(text == "Question-and-Answer Session"):
								flag_ques = False
								continue

							# if the text is a headin it must be a speaker
							if(html_text.find("strong") or text in speakers):
								speaker = text
								if text not in main_dict["Presentation"]:
									main_dict["Presentation"][text] = []
								continue

							main_dict["Presentation"][speaker].append(text)
						else:
							
							# Questionnaire is detected
							text = html_text.get_text()
							text = text.rstrip(" ")
							if(text == "Question-and-Answer Session"):
								continue
							if(html_text.find("strong") or (text in speakers) or text.startswith("A -") or text.startswith("Q -")):
								serial_num += 1
								main_dict["Questionnaire"][serial_num] = {}
								main_dict["Questionnaire"][serial_num]["Speaker"] = text
								main_dict["Questionnaire"][serial_num]["Remark"] = []
								continue
							main_dict["Questionnaire"][serial_num]["Remark"].append(text)		

				# creating a directory of ECTNestedDict
				with open(pwd+'/ECTNestedDict/'+article_no+'.json', 'w') as fp:
					json.dump(main_dict, fp)

				print(article_no)
			else:
				continue
		except:
			pass

def buildECTText():
	for filename in sorted(os.listdir(pwd+"/ECTNestedDict/")):

		# extracting the article_no
		article_no = filename.partition("-")[0]

		# opening the ECTNestedDict
		with open(pwd+"/ECTNestedDict/"+article_no, 'r') as fp:
			main_dict = json.load(fp)

		# making the text corpus
		text = "Date "+main_dict["Date"]+" Participants "
		for key in main_dict["Participants"]:
			text += " "+key
		
		text = text + " Presentation "
		for key in main_dict["Presentation"]:
			for item in main_dict["Presentation"][key]:
				text += " "+item

		text = text + " Questionnaire "
		for key in main_dict["Questionnaire"]:
			for item in main_dict["Questionnaire"][key]["Remark"]:
				text += " "+item

		article_no = article_no.strip(".json")
		print(article_no)

		# making ECTText
		with open(pwd+"/ECTText/"+article_no+'.txt', 'w') as fp:
			fp.write(text)

# calling the two modular functions
if __name__=="__main__":
	os.makedirs('ECTNestedDict', exist_ok=True)
	os.makedirs('ECTText', exist_ok=True)
	buildECTNestedDict()
	buildECTText()