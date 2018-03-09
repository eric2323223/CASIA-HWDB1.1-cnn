import tornado.ioloop
import tornado.web
from keras.models import model_from_json
import numpy as np
import base64
from PIL import Image

instance = None

class Loader():
    def __new__(cls):
        def buildModel():
            json_file = open('../model/1520265573-model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            # load weights into new model
            model.load_weights("../model/1520263770-weights-0.953545.hdf5")
            print("Loaded model from disk")

            # evaluate loaded model on test data
            model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
            return model

        def buildMapping(file):
            print("building mapping...")
            with open(file, 'r') as f:
                entries = (f.read().split('\n'))
                dict = {}
                for entry in entries:
                    k = int(entry.split(',')[0])
                    v = entry.split(',')[1]
                    dict[k] = v
                return dict

        global instance
        if instance is not None:
            return instance
        instance = object.__new__(cls)
        # instance.foo = arg
        instance.model = buildModel()
        instance.mapping = buildMapping('../src/subset_GBcode')
        return instance

    def getModel(self):
        return self.model

    def getMapping(self):
        return self.mapping

class MainHandler(tornado.web.RequestHandler):

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

    def load_received_image(self, file):
        im = Image.open(file)
        im = np.array(im.resize((64, 64), Image.ANTIALIAS))

        im = 255 - im       # invert color!!!

        # plt.imshow(im[:,:,3])
        # plt.show()
        im = np.expand_dims(im[:, :, 3:], axis=0)
        return im

    def classify(self, file):
        x = self.load_received_image(file)
        # print(x.shape)
        result = Loader().getModel().predict(x)
        # print(result.shape)
        return (self.topN(result, Loader().getMapping(), 5))

    def get(self):
        self.write("Hello, world")
        # self.write(self.classify())

    def post(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")
        print(self.request.body)
        imgStr = self.request.body.decode('utf-8')
        imgStr = imgStr.split(",")[1]
        print(imgStr)
        with open('testImage.png', 'wb') as f:
            f.write(base64.urlsafe_b64decode(imgStr))
            # f.write(base64.b64decode(imgStr))
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
