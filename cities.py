import pandas as pd
import matplotlib.pylab as plt

data = pd.read_csv('month.csv')

pile =  data['text']
cities = []
i=0
dictionary = {}

print len(pile)
while i < len(pile):
	dead = pile[i]
	#print dead
	dead2 = dead.split()
	#print dead2
	if dead2 != []:
		text = dead2[0]
		if text[-1] == ":":
			#print text
			if text[0]==",":
				text = text[1:]
				
			cities.append(text[:-1])



	i=i+1

#print cities

for x in cities:
	if x in dictionary.keys():
		dictionary[x] = dictionary[x] + 1
	else:
		dictionary[x] = 0
		dictionary[x] = 1


#print dictionary

newlist = []
newfreq = []
for item in dictionary.keys():
	if dictionary[item] > 5:
		newlist.append(item)
		newfreq.append(dictionary[item])

#print newlist
fig = plt.figure()
ax = fig.add_subplot(111)
plt.bar(newlist, newfreq, color='g')
fig.autofmt_xdate()
plt.show()


