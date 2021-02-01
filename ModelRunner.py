import tensorflow as tf
import numpy as np
import cv2

class ModelRunner:
    def __init__(self, model_path):
        self.session = tf.Session()
        tf.saved_model.loader.load(self.session, [tf.saved_model.tag_constants.SERVING], model_path)
        self.input = tf.get_default_graph().get_tensor_by_name('input:0')
        self.output = tf.get_default_graph().get_tensor_by_name('output:0')

    def __del__(self):
        self.session.close()

    def run_on_image(self, input_image):
        # input_image is expected to be BGR uint8 of shape [h, w, 3]
        input_image = cv2.resize(input_image, (68, 68), interpolation=cv2.INTER_CUBIC)
        input_image = input_image[:,:,::-1].astype(np.float32)  # to RGB and float 32
        input_image = input_image / 255  # from [0,255] to [0, 1]
        input_image = np.expand_dims(input_image, axis=0)  # from [h, w, 3] to [1, h, w, 3]
        output_class_scores = self.session.run(self.output, feed_dict={self.input: input_image})
        output_class_scores = np.maximum(output_class_scores, 0)  # remove negative values
        output_class_scores = output_class_scores / np.sum(output_class_scores)  # normalize
        return output_class_scores[0]  # remove batch dimension

if __name__ == "__main__":
    test_image_path = './test/lasagna/139.png'
    t = cv2.imread(test_image_path)
    model_runner = ModelRunner('./export')
    print(model_runner.run_on_image(t))