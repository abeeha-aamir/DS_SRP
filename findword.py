import pandas as pandas
import matplotlib.pylab as plt
import string as string

data = pandas.read_csv('month.csv')

text = data['text']
date = data['date']

x=0
while x < len(date):
	if date[x] != "None":
		date[x] = str(pandas.to_datetime(date[x]))
		#print date[x]
		
	x=x+1


entry = raw_input('What word are you looking for?')
listfreq = []
num = 0
while num < len(text):
	listfreq.append(0)
	text[num] = text[num].translate(None, string.punctuation)
	text[num] = text[num].lower()
	num = num + 1

z=0

for x in text:
	if x!=[]:
		if entry in x:
			listfreq[z] = listfreq[z] + 1

	z = z + 1


print listfreq

y = 0
wordarray = {}

while y < len(listfreq):
	if date[y] not in wordarray.keys():
		wordarray[date[y]] = listfreq[y]
	else:
		wordarray[date[y]] = wordarray[date[y]] + listfreq[y]

	y = y+1

print sorted(wordarray.items())

list1=[]
list2=[]

for it in sorted(wordarray.keys()):
	list1.append(it[8:10])
	list2.append(wordarray[it])

print list1,list2


fig = plt.figure()
ax = fig.add_subplot(111)
plt.bar(list1, list2, color='g')
fig.autofmt_xdate()
plt.show()


