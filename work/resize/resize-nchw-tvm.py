from __future__ import absolute_import, print_function

import tvm
import topi
from topi.image.resize import resize
import numpy as np
from tvm.contrib.image import bilinear_weights

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

image = Image.open("/media/sf_VMShare/elephant-299.jpg")
#image = Image.open("/media/sf_VMShare/flower.jpg")

image = np.array(image)
img_gray = np.dot(image[..., :3], [0.30, 0.59, 0.11])
#plt.imshow(img_gray, cmap="gray")
#plt.show()

to_width = 500
to_height = 500

image = np.expand_dims(image, axis=0)
image = np.transpose(image, (0,3,1,2))

weights = bilinear_weights(image, to_height, to_width, "NCHW")

print ("W Shape:", weights.shape)

A = tvm.placeholder(image.shape, name='A', dtype='uint8')
B = tvm.placeholder(weights.shape, name='B', dtype='float32')
C = resize(data=A, weights=B, out_size=(to_height, to_width), layout="NCHW")

out_shape = (image.shape[0], image.shape[1], to_height, to_width)
dtype = A.dtype

ctx = tvm.context('llvm', 0)
with tvm.target.create('llvm'):
    s = topi.generic.schedule_injective(C)
    a = tvm.nd.array(image, ctx)
    b = tvm.nd.array(weights, ctx)
    c = tvm.nd.array(np.zeros(out_shape, dtype=dtype), ctx)

    f = tvm.build(s, [A, B, C], 'llvm')

    #print (f.get_source())

    f(a, b, c)


    scaled = c.asnumpy()

    print ("SCALE:", scaled.shape)

    scaled = np.transpose(scaled, (0,2,3,1))

    image = scaled[0];


    print ("Out Shape:", image.shape)

    img_gray = np.dot(image[..., :3], [0.30, 0.59, 0.11])
    plt.imshow(img_gray, cmap="gray")
    plt.show()
