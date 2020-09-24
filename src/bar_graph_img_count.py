import pathlib
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
data_dir = pathlib.Path('imgs/train')

def photo_count(categories):
	'''Takes in iterable of categories of photos aligning with folder names.
	Returns the counts of each in same order as given'''
	counts = []
	for i in categories:
		counts.append(len(list(data_dir.glob(i+'/*.jpg'))))
	print(counts)
	return counts

def plot_bar(x_pos, categories):
	'''Plots counts of each photo category'''
	y = photo_count(categories)
	plt.figure(figsize=(10,10))
	plt.bar((x_pos), (y))
	plt.xticks(ticks=(x_pos), labels=(categories))
	plt.title('Photo Class Distribution')
	plt.savefig('Class_dist.png')

if __name__ == "__main__":
	categories = input("Enter Img Categories with only comma separating: ").split(",")
	print(categories)
	print(type(categories))
	x_pos = input("Enter x pos with only comma separating: ").split(",")
	plot_bar(x_pos, categories)
