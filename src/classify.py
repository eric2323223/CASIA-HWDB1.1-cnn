from keras.models import model_from_json
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# first layer, index=0
def load_local_image(file):
    img = image.load_img(file, target_size=(64, 64))
    x = image.img_to_array(img)
    plt.imshow(x[:,:,0])
    plt.show()
    x = np.expand_dims(x[:,:,:1], axis=0)
    print(x.shape)
    return x

# 4th layer, index=3
def load_received_image(file):
    im = Image.open(file)
    im = np.array(im.resize((64, 64), Image.ANTIALIAS))

    # invert color
    im = 255 - im

    # plt.imshow(im[:,:,3])
    # plt.show()
    im = np.expand_dims(im[:,:,3:], axis=0)

    return im

def topN(result, dict, n):
    # indexes = result.argsort()[-(n):][::-1]
    indexes = np.argsort(result)
    indexes = indexes[0][::-1][:n]
    output = []
    for idx in indexes:
        # print(idx, result[0][idx])
        output.append((dict.get(idx), result[0][idx]))
        # output.append(bytes(dict.get(idx)).decode('gb2312'), result[0][idx])
    return output


def loadMapping(file):
    f = open(file, 'r')
    entries = (f.read().split('\n'))
    dict = {}
    for entry in entries:
        k = int(entry.split(',')[0])
        v = entry.split(',')[1]
        dict[k] = v
    # print(dict)
    return dict

mapping = loadMapping('subset_GBcode')
# load json and create model
json_file = open('../model/1520265573-model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("../model/1520263770-weights-0.953545.hdf5")

# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
# score = loaded_model.evaluate(X, Y, verbose=0)
# print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1 ] *100))

# x = load_local_image('../data/predict/he5.png')
x = load_received_image('../misc/testImage.png')
# print(x.shape)
result = loaded_model.predict(x)
# print(result.shape)
print(topN(result, mapping, 5))
