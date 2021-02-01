import tensorflow as tf
import model

if __name__ == '__main__':
    input_placeholder = tf.placeholder(shape=[1, 68, 68, 3], dtype=tf.float32, name='input')  # exporting model with a fixed batch size of 1
    output_tensor = model.simple_dl_classifier(input_placeholder)
    model_output = tf.squeeze(output_tensor, axis=[1, 2])  # [1, 1, 1, 2] -> [1, 2]
    output_placeholder = tf.identity(model_output, name='output')
    saver = tf.train.Saver()
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        saver.restore(session, tf.train.latest_checkpoint('./checkpoints'))
        tf.saved_model.simple_save(session, './export', inputs={'input': input_placeholder}, outputs={'output': output_placeholder})
