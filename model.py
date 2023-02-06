import os
import logging
import numpy as np
from tqdm import trange
from sklearn.metrics import roc_auc_score

import efficientnet.keras as efn
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint

from data import CXR, CXRPair


class DetNochange(CXR):

    def __init__(self, args):
        # model configuration
        self.image_size = args.image_size
        self.num_channel = args.num_channel
        self.batch_size = args.batch_size
        self.initial_epoch = args.initial_epoch
        self.epochs = args.epochs
        self.learning_rate = args.learning_rate
        self.decay = args.decay
        self.data = args.data
        self.logdir = args.logdir
        self.weight = args.weight

        self.model = efn.EfficientNetB4(include_top=True, weights=None, input_shape=(self.image_size, self.image_size, self.num_channel), classes=2)
        print ("\n>>> classifier: EfficientNetB4")
        print (f"    - input_shape: {self.model.input.shape[1:]}")


    def train(self):
        data_train = CXRPair(os.path.join(self.data, 'train'))
        train_generator = self.data_generator(data_train, batch_size=self.batch_size)

        data_valid = CXRPair(os.path.join(self.data, 'valid'))
        valid_generator = self.data_generator(data_valid, batch_size=self.batch_size)

        #################################################################################################

        if not os.path.exists(self.logdir): os.mkdir(self.logdir)
        checkpoint_path = os.path.join(self.logdir, "FU_3ch_e*epoch*.h5")
        checkpoint_path = checkpoint_path.replace("*epoch*", "{epoch:04d}")
        print (f"\n>>> checkpoint & log: {self.logdir}\n")

        callbacks = [ModelCheckpoint(checkpoint_path)]

        #################################################################################################

        self.model.compile(optimizer=SGD(lr=self.learning_rate, decay=self.decay), loss='binary_crossentropy', metrics=['accuracy'])

        self.model.fit_generator(train_generator, 
                                 initial_epoch=self.initial_epoch, 
                                 epochs=self.epochs,
                                 steps_per_epoch=len(data_train.data)/self.batch_size,
                                 validation_data=valid_generator,
                                 validation_steps=len(data_valid.data)/self.batch_size,
                                 callbacks=callbacks)


    def test(self):
        data_test = CXRPair(self.data)

        print (f"\n>>> weight: {self.weight}")
        self.model.load_weights(self.weight)

        y_true, y_prob = [], []
        for _idx in trange(len(data_test)):
            img_input, gt_class = data_test.load_data(_idx)
            score = np.squeeze(self.model.predict(np.expand_dims(img_input, axis=0)))
            
            y_true.append(np.argmax(gt_class))
            y_prob.append(score[1])

        print (f"\n>>> AUROC: {roc_auc_score(y_true, y_prob):.03f}")
    

    def data_generator(self, data, shuffle=True, batch_size=1, augmentation=None):
        num_data = len(data)
        indices = list(range(num_data))

        idx_data = -1
        idx_batch = 0
        error_count = 0

        while True:
            try:
                idx_data = (idx_data+1) % num_data
                if shuffle and idx_data == 0:
                    np.random.shuffle(indices)

                input_img, output_class = data.load_data(indices[idx_data])

                # init batch arrays
                if idx_batch == 0:
                    batch_inputs = np.zeros((batch_size,) + input_img.shape, dtype=input_img.dtype)
                    batch_outputs = np.zeros((batch_size,) + output_class.shape, dtype=output_class.dtype)

                # add to batch
                batch_inputs[idx_batch] = input_img
                batch_outputs[idx_batch] = output_class
                idx_batch += 1

                # batch full?
                if idx_batch >= batch_size:
                    inputs = [batch_inputs]
                    outputs = [batch_outputs]

                    yield inputs, outputs

                    # start a new batch
                    idx_batch = 0

            except(GeneratorExit, KeyboardInterrupt):
                raise

            except:
                # log it and skip the image
                logging.exception(f"Error while processing the image \"{data.data[indices[idx_data]]['registered_fu']}\"")
                error_count += 1
                if error_count > 5: raise
