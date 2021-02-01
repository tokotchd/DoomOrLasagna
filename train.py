import numpy as np
import tensorflow as tf
import model
import dl_loader as dl

if __name__ == '__main__':
    BATCH_SIZE = 15  # chosen because no GPU :(
    MAX_EPOCH = 25
    STAGE = 0  # 0 for training from scratch, change to 1 for resuming training

    input_placeholder = tf.placeholder(shape=[BATCH_SIZE, 68, 68, 3], dtype=tf.float32)
    correct_class_placeholder = tf.placeholder(shape=[BATCH_SIZE], dtype=tf.int32)

    model_guesses = model.simple_dl_classifier(input_placeholder)  # of shape [BATCH_SIZE, 1, 1, 2]
    model_guesses = tf.squeeze(model_guesses, axis=[1,2])  # of shape [BATCH_SIZE, 2]

    loss = tf.losses.sparse_softmax_cross_entropy(logits=model_guesses, labels=correct_class_placeholder)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
    train_op = optimizer.minimize(loss)

    model_checkpoint_path = './checkpoints/checkpoint.ckpt'

    epoch_counter = tf.Variable(0, name='epoch_counter', trainable=False)
    increment_epoch_op = tf.assign_add(epoch_counter, 1, name='increment_epoch')

    model_saver = tf.train.Saver()

    # ha, we don't need these, we can't afford those fancy things
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True

    # to make the training faster and since there aren't that many images, we can preload them all into memory here
    # rather than repeatedly load from disk while training
    train_lasagna_images = dl.load_and_preprocess_image_list(dl.get_image_paths_from_dir('./train/lasagna'), (68,68))
    train_doom_images = dl.load_and_preprocess_image_list(dl.get_image_paths_from_dir('./train/doom'), (68,68))
    test_lasagna_images = dl.load_and_preprocess_image_list(dl.get_image_paths_from_dir('./test/lasagna'), (68,68))
    test_doom_images = dl.load_and_preprocess_image_list(dl.get_image_paths_from_dir('./test/doom'), (68,68))

    # create class values for all of the image data
    train_lasagna_classes = np.full(shape=[len(train_lasagna_images)], fill_value=0)
    train_doom_classes = np.full(shape=[len(train_doom_images)], fill_value=1)
    test_lasagna_classes = np.full(shape=[len(test_lasagna_images)], fill_value=0)
    test_doom_classes = np.full(shape=[len(test_doom_images)], fill_value=1)

     # concatenate the training images together, the testing images together, the training classes together, and the testing classes together
    train_images = np.concatenate([train_lasagna_images, train_doom_images], axis=0)
    test_images = np.concatenate([test_lasagna_images, test_doom_images], axis=0)
    train_classes = np.concatenate([train_lasagna_classes, train_doom_classes], axis=0)
    test_classes = np.concatenate([test_lasagna_classes, test_doom_classes], axis=0)

    # some calculations for epoch/batch stuff
    assert(len(train_images) == len(train_classes))
    assert(len(test_images) == len(test_classes))
    train_batches_per_epoch = len(train_images) // BATCH_SIZE
    test_batches_per_epoch = len(test_images) // BATCH_SIZE

    # tensorboard summary shenanigans
    epoch_loss_placeholder = tf.placeholder(dtype=tf.float32, shape=[])
    epoch_loss_summary_op = tf.summary.scalar('loss', epoch_loss_placeholder)
    image_summary_op = tf.summary.image('images', input_placeholder, max_outputs=6)

    with tf.Session() as session:

        # tensorboard logging, including the graph setup
        train_tensorboard_writer = tf.summary.FileWriter('./logs/train', session.graph)
        test_tensorboard_writer = tf.summary.FileWriter('./logs/test')

        # variables initialization
        session.run(tf.global_variables_initializer())
        if STAGE != 0:
            model_saver.restore(session, tf.train.latest_checkpoint('./checkpoints'))

        # epoch loop
        for epoch in range(epoch_counter.eval(), MAX_EPOCH):

            # epoch training loop
            dl.shuffle_pairwise(train_images, train_classes)
            this_epoch_loss = 0
            for batch in range(train_batches_per_epoch):
                # batch training loop
                print('Training epoch:', epoch, 'batch', batch, end='\r', flush=True)
                current_train_images = train_images[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE]
                current_train_classes = train_classes[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE]
                if batch == 0:  # on the first batch of every epoch, we display example images for each class
                    _, batch_loss, image_summary = session.run([train_op, loss, image_summary_op], feed_dict=
                        {input_placeholder: current_train_images, correct_class_placeholder:current_train_classes})
                    this_epoch_loss += batch_loss
                    train_tensorboard_writer.add_summary(image_summary, epoch)
                else:
                    _, batch_loss = session.run([train_op, loss], feed_dict=
                        {input_placeholder: current_train_images, correct_class_placeholder:current_train_classes})
                    this_epoch_loss += batch_loss

            # post training per-epoch data
            this_epoch_loss /= train_batches_per_epoch
            loss_summary = session.run(epoch_loss_summary_op, feed_dict={epoch_loss_placeholder: this_epoch_loss})
            train_tensorboard_writer.add_summary(loss_summary, epoch)
            print('Epoch', epoch, 'average training loss:', this_epoch_loss)

            # epoch testing loop
            dl.shuffle_pairwise(test_images, test_classes)
            this_epoch_loss = 0
            for batch in range(test_batches_per_epoch):
                # batch testing loop
                print('Testing epoch:', epoch, 'batch', batch, end='\r', flush=True)
                current_test_images = test_images[batch * BATCH_SIZE:(batch + 1) * BATCH_SIZE]
                current_test_classes = test_classes[batch * BATCH_SIZE:(batch + 1) * BATCH_SIZE]
                if batch == 0:  # on the first batch of every epoch, we display example images for each class
                    batch_loss, image_summary = session.run([loss, image_summary_op], feed_dict=
                    {input_placeholder: current_test_images, correct_class_placeholder: current_test_classes})
                    this_epoch_loss += batch_loss
                    test_tensorboard_writer.add_summary(image_summary, epoch)
                else:
                    batch_loss = session.run(loss, feed_dict=
                    {input_placeholder: current_test_images, correct_class_placeholder: current_test_classes})
                    this_epoch_loss += batch_loss

            # post testing per-epoch data
            this_epoch_loss /= test_batches_per_epoch
            loss_summary = session.run(epoch_loss_summary_op, feed_dict={epoch_loss_placeholder: this_epoch_loss})
            test_tensorboard_writer.add_summary(loss_summary, epoch)
            print('Epoch', epoch, 'average testing loss:', this_epoch_loss)

            # per-epoch model checkpoint saving
            session.run(increment_epoch_op)
            model_saver.save(session, save_path=model_checkpoint_path)




