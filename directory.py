# Temporary file to make directory structure
# format: [['casia_mtcnn_cropped/CASIA_XXXXXXX/001.jpg', 'casia_mtcnn_cropped/CASIA_XXXXXX/002.jpg', ...], ...]

import tensorflow as tf
#tempFile = tf.gfile.GFile('~')


def get_cropped_CASIA_files():	
	image_files = []

	baseFileName = './faces/' #'./casia_mtcnn_cropped/'
	#files = tf.gfile.Walk('./casia_mtcnn_cropped/CASIA_2426012')
	files = tf.gfile.Walk(baseFileName)
	for file in files:
	#	print(file)
		tempFile = []
		for jpeg in file[2]:
			tempFileName = file[0] + '/' + jpeg
			#tempFileName = baseFileName + 'CASIA_2426012/' + jpeg 
			tempFile.append(tempFileName)
		
		if len(tempFile) > 0:
			image_files.append(tempFile)	
	#print(image_files)
	return image_files


files = get_cropped_CASIA_files()
with open('cropped_file_names.txt', 'w') as image_file:
	image_file.write('\n'.join([(filedata[0]+' '+filedata[0].split('/')[-2]) for filedata in files]))
