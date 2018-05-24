import math
import time


def gen_weights(image, new_h, new_w):
    (batch, h, w , channel) = image.shape

    x_ratio = (w-1)/new_w
    y_ratio = (h-1)/new_h
   
    def _bilinear_interpolation(y, x , c):
        x1 = math.floor(x_ratio * x)
        y1 = math.floor(y_ratio * y)
        x_diff = (x_ratio * x) - x1
        y_diff = (y_ratio * y) - y1

        return [x1, y1, x_diff, y_diff]

    weights = np.ones((new_h, new_w, channel, 4)).reshape(new_h, new_w, channel, 4)

    for i in range(channel):
        for j in range(new_h):
            for k in range(new_w):
                weights[j][k][i] = _bilinear_interpolation(j, k , i)
    return weights


def bilinear(image, new_h, new_w):
    (batch, h, w , channel) = image.shape

    weights = gen_weights(image, new_h, new_w)

    print("W Shape:", weights.shape);

    scaled_image = np.ones((batch, new_h, new_w, channel)).reshape(batch, new_h, new_w, channel)

    for i in range(channel):
        for j in range(new_h):
            for k in range(new_w):
                x1 = int(weights[j][k][i][0])
                y1 = int(weights[j][k][i][1])
                x_diff = weights[j][k][i][2]
                y_diff = weights[j][k][i][3]

                A = image[0][y1][x1][i];
                B = image[0][y1][x1+1][i];
                C = image[0][y1+1][x1][i];
                D = image[0][y1+1][x1+1][i];

                #Y = A(1-w)(1-h) + B(w)(1-h) + C(h)(1-w) + Dwh
                pixel = (
                        A*(1-x_diff)*(1-y_diff) +  B*(x_diff)*(1-y_diff) +
                        C*(y_diff)*(1-x_diff)   +  D*(x_diff*y_diff)
                        );

                scaled_image[0][j][k][i] = pixel

    print ("Out SHape:", scaled_image.shape)
    return scaled_image


from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

image = Image.open("elephant-299.jpg")

image = np.array(image)
img_gray = np.dot(image[..., :3], [0.30, 0.59, 0.11])
plt.imshow(img_gray, cmap="gray")
plt.show()

scaled = bilinear(np.expand_dims(image, axis=0), 150, 150)
image = scaled[0];

img_gray = np.dot(image[..., :3], [0.30, 0.59, 0.11])
plt.imshow(img_gray, cmap="gray")
plt.show()

scaled = bilinear(np.expand_dims(image, axis=0), 500, 500)
image = scaled[0];

img_gray = np.dot(image[..., :3], [0.30, 0.59, 0.11])
plt.imshow(img_gray, cmap="gray")
plt.show()
