from keras.models import model_from_json
from keras.preprocessing import image
import numpy as np

def load_image(file):
    img = image.load_img(file, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x[:,:,0:1], axis=0)
    return x

def topN(result, n):
    # indexes = result.argsort()[-(n):][::-1]
    indexes = np.argsort(result)
    indexes = indexes[0][::-1][:n]
    output = []
    for idx in indexes:
        # print(idx, result[0][idx])
        output.append((idx, result[0][idx]))
    return output

# load json and create model
json_file = open('../model/1520265573-model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("../model/1520263770-weights-0.953545.hdf5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
# score = loaded_model.evaluate(X, Y, verbose=0)
# print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1 ] *100))

x = load_image('../data/predict/lin4.png')
# print(x.shape)
result = loaded_model.predict(x)
# print(result.shape)
print(topN(result, 5))