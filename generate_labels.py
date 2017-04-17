import os

data_dir = './faces/'

labels = [name for name in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, name))]

with open('labels.txt', 'w') as labels_file:
	labels_file.write('\n'.join(labels))


