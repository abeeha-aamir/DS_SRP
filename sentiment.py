import pandas as pandas
import matplotlib.pylab as plt
from textblob import TextBlob

data = pandas.read_csv('merge.csv') #changed

text = data['text']
date = data['date']
category = data['category']

cats = data.category.unique()


x=0
while x < len(date):
	if date[x] != "None":
		date[x] = str(pandas.to_datetime(date[x]))
		
		
	x=x+1


artperday = {}

for x in date:
	if x not in artperday.keys():
		artperday[x] = 1
	else:
		artperday[x] = artperday[x] + 1

print sorted(artperday.items())
list1=[]
list2=[]

for it in sorted(artperday.keys()):
	list1.append(it[8:10])
	list2.append(artperday[it])

print list1,list2

fig = plt.figure()
ax = fig.add_subplot(111)
plt.bar(list1, list2, color='g')
fig.autofmt_xdate()
plt.show()

#changed from here...
#Sentiment Analysis
polarity = []
subjectivity = []
pols_neg = {}
pols_pos = {}
subs_above_half = {}
subs_below_half = {}

i = 0

for x in cats:
	pols_pos[x] = 0
	pols_neg[x] = 0
	subs_above_half[x] = 0
	subs_below_half[x] = 0

while i < len(text):
	x = text[i]
	if x != []:
		p = TextBlob(x.decode('utf-8')).sentiment.polarity
		s = TextBlob(x.decode('utf-8')).sentiment.subjectivity
		
		if p <= 0:
			pols_neg[category[i]]+=1
			
		else:
			pols_pos[category[i]]+=1

		if s <= 0.5:
			subs_below_half[category[i]]+= 1
			
		else:
			subs_above_half[category[i]]+=1

		polarity.append(p)
		subjectivity.append(s)
	else:
		polarity.append("N/A")
		subjectivity.append("N/A")
	i += 1

data['polarity'] = polarity
data['subjectivity'] = subjectivity

data.to_csv("filename.csv", sep=',', encoding='utf-8')


list1=[]
list2=[]
list3=[]
list4=[]
list5=[]


for c in cats:
	print c + ': '
	print 'Negative polarity: ' + str(pols_neg[c])
	print 'Positive polarity: ' + str(pols_pos[c])
	print 'Subjectivity > 0.5: ' + str(subs_above_half[c])
	print 'Subjectivity <= 0.5: ' + str(subs_below_half[c])
	list1.append(c)
	list2.append(pols_neg[c])
	list3.append(pols_pos[c])
	list4.append(subs_below_half[c])
	list5.append(subs_above_half[c])
	
print list1
fig = plt.figure()
ax = fig.add_subplot(111)
plt.bar(list1, list2, color='g')
plt.xlabel("Categories")
plt.ylabel("Negative Polarity")
fig.autofmt_xdate()
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111)
plt.bar(list1, list3, color='g')
plt.xlabel("Categories")
plt.ylabel("Positive Polarity")
fig.autofmt_xdate()
plt.show()



fig = plt.figure()
ax = fig.add_subplot(111)
plt.bar(list1, list4, color='g')
plt.xlabel("Categories")
plt.ylabel("Subjectivity <= 0.5")
fig.autofmt_xdate()
plt.show()



fig = plt.figure()
ax = fig.add_subplot(111)
plt.bar(list1, list5, color='g')
plt.xlabel("Categories")
plt.ylabel("Subjectivity > 0.5")
fig.autofmt_xdate()
plt.show()





