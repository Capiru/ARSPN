import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import cv2
import os


#GLOBAL VARIABLES
NUMBER_BITS = 8
NOISE_AMOUNT = 0.05
MAX_PIXEL_VALUE = 2**NUMBER_BITS-1

#returns the weight to be used in ARSPN by calculating the difference between the pixels
def get_weight(original_pixel_coord:tuple,desired_pixel_coord:tuple):
	dist = max(abs(original_pixel_coord[0]-desired_pixel_coord[0]),abs(original_pixel_coord[1]-desired_pixel_coord[1]))
	return 10**((dist-1)*-1)

#adds salt and pepper noise to the selected image with selected noise amount, and maximum pixel value
def add_salt_and_pepper_noise(img,amount = 0.5,MAX_PIXEL_VALUE = 255):
	rows,columns = img.shape
	out_img = np.copy(img)
	num_altered_pixels = np.ceil(amount * img.size * 0.5)
	coordinates_salts = [np.random.randint(0,i-1,int(num_altered_pixels)) for i in img.shape]
	out_img[coordinates_salts] = MAX_PIXEL_VALUE
	coordinates_peppers = [np.random.randint(0,i-1,int(num_altered_pixels)) for i in img.shape]
	out_img[coordinates_peppers] = 0
	return out_img

#this step is done before adding salt and pepper noise to make sure that we don't run into the problem of our algorithm thinking that part of the image is a salt or pepper, 
#or that the entire background is peppers
def fix_min_and_max(img,MAX_PIXEL_VALUE=MAX_PIXEL_VALUE):
	img[img == 0] = 1
	img[img == MAX_PIXEL_VALUE] = MAX_PIXEL_VALUE-1
	return img

#selecting a file from the image folder
file_list = os.listdir("Brain Tumor")
img_path = os.path.join("Brain Tumor",file_list[1])

#transforming the image into a numpy array (matrix)
input_image = np.array(imread(img_path))

#it tries retrieving the rows columns and channels of our matrix, if it raises a ValueError, 
#it's because we're trying to select the channels of a grayscale image, if so we only select the rows and columns
try:
	rows,columns,channels = input_image.shape
except ValueError:
	rows,columns = input_image.shape

#we select only the first channel of the image
try:
	input_image = input_image[:,:,0]
except IndexError:
	input_image = input_image[:,:]

#we fix the background by changing 0s to 1s before adding noise
input_image = fix_min_and_max(input_image)

#we add noise according to our global variable noise amount and max pixel value
noisy_image = add_salt_and_pepper_noise(input_image,NOISE_AMOUNT,MAX_PIXEL_VALUE)

#we get a median blur image from cv2 with a 3x3 filter just for showing the difference
median_blur = cv2.medianBlur(noisy_image,3)

#we initialize the output_image with the noisy_image and set the array to be writable
output_image = np.array(noisy_image)
output_image.setflags(write = 1)


#we initialize our variables to be used in the algorithm
d = 1
count = 0
sum_pixel_values = 0
weight = 0
#we do a for every row and every column to select all pixels
for row in range(rows):
	for column in range(columns):
		#we check if current pixel is a Salt or Pepper
		if output_image[row,column] == 0 or output_image[row,column] == MAX_PIXEL_VALUE:
			# Start with a window distance 1 from pixel (3x3 and keep increasing until a healthy pixel shows up)
			for i in range(max(row-d,0),min(row+d+1,rows-1)):
				for j in range(max(column-d,0),min(column+d+1,columns-1)):
					#we check if pixels in the window are also noise, or if healthy we add it to the median value of that pixel
					if not (output_image[i,j] == 0 or output_image[i,j] == MAX_PIXEL_VALUE):
						for k in range(1,d+1):
							if k <= d:
								#we calculate the weight by calculating the difference from this pixel (i,j) to the noisy pixel in (row,column)
								weight = get_weight((row,column),(i,j))
								#we sum the weighted contribution of this pixel to the final value
								sum_pixel_values += output_image[i,j]*weight
								count += 1
							else:
								d += 1
								continue
		#if there's no division by zero, the pixel value will be equal to the sum of healthy pixel values divided by their count and weights
		if not weight*count == 0:
			output_image[row,column] = sum_pixel_values/(weight*count)
			count = 0
			sum_pixel_values = 0
			d = 1

#we do a plot to check the difference between input, noisy, median blur and ARSPN images
fig = plt.figure(figsize=(16, 4))
ax = []

ax.append(fig.add_subplot(1, 4, 1))
ax[-1].set_title("Original")
plt.imshow(input_image,cmap='gray')
ax.append(fig.add_subplot(1, 4, 2))
ax[-1].set_title(str(NOISE_AMOUNT*100)+"% S&P")
plt.imshow(noisy_image,cmap='gray')
ax.append(fig.add_subplot(1, 4, 3))
ax[-1].set_title("Median Blur")
plt.imshow(median_blur,cmap='gray')
ax.append(fig.add_subplot(1, 4, 4))
ax[-1].set_title("ARSPN")
plt.imshow(output_image,cmap='gray')
plt.savefig('output.png')
plt.show()
