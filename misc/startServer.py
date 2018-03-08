import tornado.ioloop
import tornado.web
from keras.models import model_from_json
from keras.preprocessing import image
import numpy as np
import base64
import os

class MainHandler(tornado.web.RequestHandler):
    def loadMapping(self, file):
        f = open(file, 'r')
        entries = (f.read().split('\n'))
        dict = {}
        for entry in entries:
            k = int(entry.split(',')[0])
            v = entry.split(',')[1]
            dict[k] = v
        # print(dict)
        return dict

    def topN(self, result, dict, n):
        # indexes = result.argsort()[-(n):][::-1]
        indexes = np.argsort(result)
        indexes = indexes[0][::-1][:n]
        output = {}
        for idx in indexes:
            # print(idx, result[0][idx])
            gbcode = dict.get(idx)
            chinese_char = self.gb2312_to_character(gbcode)
            output[chinese_char] = str(result[0][idx])
            # output.append(bytes(dict.get(idx)).decode('gb2312'), result[0][idx])
        return output

    def load_image(self, file):
        print("Image file size: {}".format(os.path.getsize(file)))
        img = image.load_img(file, target_size=(64, 64))
        x = image.img_to_array(img)
        x = np.expand_dims(x[:, :, 0:1], axis=0)
        return x

    def classify(self, file):
        mapping = self.loadMapping('../src/subset_GBcode')
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

        x = self.load_image(file)
        # print(x.shape)
        result = loaded_model.predict(x)
        # print(result.shape)
        return (self.topN(result, mapping, 5))

    def get(self):
        # self.write("Hello, world")
        self.write(self.classify())

    def post(self):
        print(self.request.body)
        imgStr = self.request.body.decode('utf-8')
        imgStr = imgStr.split(",")[1]
        print(imgStr)
        with open('testImage.png', 'wb') as f:
            f.write(base64.b64decode(imgStr))
        output = self.classify("testImage.png")
        print(output)
        self.write(output)

    def gb2312_to_character(self, gbcode):
        a = int(gbcode[:2], 16)
        b = int(gbcode[2:], 16)
        return bytes([a, b]).decode('gb2312')


def make_app():
    return tornado.web.Application([
        (r"/hccr", MainHandler),
        (r"/(.*)", tornado.web.StaticFileHandler, {'path':"C:/Users/43971153/Downloads/workspace/DL/CASIA-HWDB1.1-cnn-master/misc"})
    ])

if __name__ == "__main__":
    app = make_app()
    app.listen(8888)
    tornado.ioloop.IOLoop.current().start()
