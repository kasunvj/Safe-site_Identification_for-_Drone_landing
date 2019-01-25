import numpy as np
import matplotlib.pyplot as plt 
import matplotlib  
import math

nu_images = 10 

img_size = 8
min_object_size = 1
max_object_size = 4

num_of_objects = 1

images = np.zeros((nu_images, img_size,img_size))
bboxes = np.zeros((nu_images, num_of_objects,4))

for i_img in range (nu_images):
	for i_obj in range(num_of_objects):
		w, h = np.random.randint (min_object_size,max_object_size, size= 2) #random size
		x = np.random.randint( 0 , img_size - w)
		y = np.random.randint( 0 , img_size - h)
		images[i_img , x:x+w , y:y+h] = 1
		bboxes[i_img, i_obj] = [x,y,w,h]

print (images.shape)
print(bboxes.shape)

i = 0
plt.imshow(images[i].T, cmap='Greys', interpolation='none', origin='lower', extent=[0, img_size, 0, img_size])
for bbox in bboxes[i]:
    plt.gca().add_patch(matplotlib.patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], ec='r', fc='none'))