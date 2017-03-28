# Temporary file to make directory structure
# format: [['casia_mtcnn_cropped/CASIA_XXXXXXX/001.jpg', 'casia_mtcnn_cropped/CASIA_XXXXXX/002.jpg', ...], ...]

import tensorflow as tf
#tempFile = tf.gfile.GFile('~')

image_files = []

baseFileName = './casia_mtcnn_cropped/'
#files = tf.gfile.Walk('./casia_mtcnn_cropped/CASIA_2426012')
files = tf.gfile.Walk('./casia_mtcnn_cropped/')
for file in files:
#	print(file)
	tempFile = []
	for jpeg in file[2]:
		tempFileName = file[0] + '/' + jpeg
		#tempFileName = baseFileName + 'CASIA_2426012/' + jpeg 
		tempFile.append(tempFileName)
	image_files.append(tempFile)	
print(image_files)


