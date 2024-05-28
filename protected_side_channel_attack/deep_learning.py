import os
import sys

import numpy as np

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Flatten, Dense, Input, Conv1D, AveragePooling1D, BatchNormalization, Activation, add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical

from utils import *

RWS_SUBTRACE, RWS_SUBKEYWIDTH = np.arange(500, 5500), KEYROUND_WIDTH_B4 // 4

def check_file_exists(file_path):
	file_path = os.path.normpath(file_path)
	if os.path.exists(file_path) == False:
		raise ValueError("Error: provided file path '%s' does not exist!" % file_path)
	return

class ResNetSCA:
    # https://github.com/ANSSI-FR/ASCAD/blob/master/ASCAD_train_models.py
    @staticmethod
    def resnet_layer(inputs, num_filters=16, kernel_size=11, strides=1, activation='relu', batch_normalization=True, conv_first=True):
        conv = Conv1D(num_filters, kernel_size=kernel_size, strides=strides, padding='same', kernel_initializer='he_normal')
        x = inputs
        if conv_first:
            x = conv(x)
            if batch_normalization:
                x = BatchNormalization()(x)
            if activation is not None:
                x = Activation(activation)(x)
        else:
            if batch_normalization:
                x = BatchNormalization()(x)
            if activation is not None:
                x = Activation(activation)(x)
            x = conv(x)
        return x

    @staticmethod
    def rws_branch(x):
        x = Dense(1024, activation='relu', name='rws_perm')(x)
        x = BatchNormalization()(x)
        x = Dense(KEYROUND_WIDTH_B4, activation="softmax", name='rws_perm_output')(x)
        return x

    @staticmethod
    def rws_mask_branch(x, keyround_idx, share_idx):
        x = Dense(512, activation='relu', name=f'rws_mask_{keyround_idx}_{share_idx}')(x)
        x = BatchNormalization()(x)
        x = Dense(len(KEY_ALPHABET), activation="softmax", name=f'rws_mask_{keyround_idx}_{share_idx}_output')(x)
        return x

    @staticmethod
    def round_branch(x):
        x = Dense(256, activation='relu', name='round_perm')(x)
        x = BatchNormalization()(x)
        x = Dense(KEYROUND_WIDTH_B4 // BLOCK_WIDTH_B4, activation="softmax", name='round_perm_output')(x)
        return x

    @staticmethod
    def block_branch(x, round_idx):
        x = Dense(256, activation='relu', name=f'block_perm_{round_idx}')(x)
        x = BatchNormalization()(x)
        x = Dense(BLOCK_WIDTH_B4, activation="softmax", name=f'block_perm_{round_idx}_output')(x)
        return x
    
    @staticmethod
    def mask_branch(x, round_idx, block_idx):
        x = Dense(512, activation='relu', name=f'mask_{round_idx}_{block_idx}')(x)
        x = BatchNormalization()(x)
        x = Dense(len(KEY_ALPHABET) ** NR_SHARES, activation="softmax", name=f'mask_{round_idx}_{block_idx}_output')(x)
        return x

    @staticmethod
    def rws_multilabel_to_categorical(Y):
        y = {}        
        y['rws_perm_output'] = to_categorical(Y['rws_perm'], num_classes=KEYROUND_WIDTH_B4).astype(np.int8)

        for keyround_idx in range(RWS_SUBKEYWIDTH):
            for share_idx in range(NR_SHARES):
                y[f'rws_mask_{keyround_idx}_{share_idx}_output'] = to_categorical(Y[f'rws_mask_{keyround_idx}_{share_idx}'], num_classes=len(KEY_ALPHABET)).astype(np.int8)
        return y

    @staticmethod
    def multilabel_to_categorical(Y):
        y = {}
        y['round_perm_output'] = to_categorical(Y['round_perm'], num_classes=KEYROUND_WIDTH_B4 // BLOCK_WIDTH_B4).astype(np.int8)

        for round_idx in range(EARLIEST_ROUND, LATEST_ROUND):
            y[f'block_perm_{round_idx}_output'] = to_categorical(Y[f'block_perm_{round_idx}'], num_classes=BLOCK_WIDTH_B4).astype(np.int8)
            for block_idx in range(BLOCK_WIDTH_B4):
                y[f'mask_{round_idx}_{block_idx}_output'] = to_categorical(Y[f'mask_{round_idx}_{block_idx}'], num_classes=len(KEY_ALPHABET) ** NR_SHARES).astype(np.int8)
        return y

    def __init__(self, rws, epochs, dataset_size, batch_size, depth=19):
        #depth = 37 if rws else 19
        if (depth - 1) % 18 != 0:
            raise ValueError('depth should be 18n+1 (eg 19, 37, 55 ...)')
        # Start model definition.
        num_filters = 16
        num_res_blocks = int((depth - 1) / 18)
        inputs = Input(shape=(len(RWS_SUBTRACE) if rws else 1950, 1))
        x = ResNetSCA.resnet_layer(inputs=inputs)
        # Instantiate the stack of residual units
        for stack in range(8 if rws else 6):
            for res_block in range(num_res_blocks):
                strides = 1
                if stack > 0 and res_block == 0:
                    strides = 2
                y = ResNetSCA.resnet_layer(inputs=x,
                                num_filters=num_filters,
                                strides=strides)
                y = ResNetSCA.resnet_layer(inputs=y,
                                num_filters=num_filters,
                                activation=None)
                if stack > 0 and res_block == 0:
                    x = ResNetSCA.resnet_layer(inputs=x,
                                    num_filters=num_filters,
                                    kernel_size=1,
                                    strides=strides,
                                    activation=None,
                                    batch_normalization=False)
                x = add([x, y])
                x = Activation('relu')(x)
            if (num_filters<256):
                num_filters *= 2
        x = AveragePooling1D(pool_size=4)(x)
        x = Flatten()(x)
        if rws:
            x_rws = ResNetSCA.rws_branch(x)
            x_rws_masks = []
            for keyround_idx in range(RWS_SUBKEYWIDTH):
                for share_idx in range(NR_SHARES):
                    x_rws_masks.append(ResNetSCA.rws_mask_branch(x, keyround_idx, share_idx))
            total = [x_rws] + x_rws_masks
        else:
            x_round = ResNetSCA.round_branch(x)
            x_block = []
            x_masks = []
            for round_idx in range(EARLIEST_ROUND, LATEST_ROUND):
                x_block.append(ResNetSCA.block_branch(x, round_idx))
                for block_idx in range(BLOCK_WIDTH_B4):
                    x_masks.append(ResNetSCA.mask_branch(x, round_idx, block_idx))
            total = [x_round] + x_block + x_masks

        self.model = Model(inputs, total, name='extract_resnet')
        optimizer = Adam(learning_rate=ExponentialDecay(initial_learning_rate=0.001, decay_steps=epochs*dataset_size/batch_size, decay_rate=0.9))
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy']*len(total))
        self.rws = rws
        self.epochs = epochs
        self.dataset_size = dataset_size
        self.batch_size = batch_size

    def train_model(self, X_profiling, Y_profiling, save_file_name, validation_split=0.2, early_stopping=1, patience=10):
        assert X_profiling.shape[0] == self.dataset_size
        check_file_exists(os.path.dirname(save_file_name))
        # Save model calllback
        save_model = ModelCheckpoint(save_file_name, save_best_only=True)
        callbacks=[save_model]
        # Early stopping callback
        if (early_stopping != 0):
            if validation_split == 0:
                validation_split=0.1
            callbacks.append(EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True))
        # Get the input layer shape
        if isinstance(self.model.get_layer(index=0).output.shape, list):
            input_layer_shape = self.model.get_layer(index=0).output.shape[0]
        else:
            input_layer_shape = self.model.get_layer(index=0).output.shape
        # Sanity check
        if input_layer_shape[1] != len(X_profiling[0]):
            print("Error: model input shape %d instead of %d is not expected ..." % (input_layer_shape[1], len(X_profiling[0])))
            sys.exit(-1)
        # Adapt the data shape according our model input
        if len(input_layer_shape) == 2:
            # This is a MLP
            Reshaped_X_profiling = X_profiling
        elif len(input_layer_shape) == 3:
            # This is a CNN: expand the dimensions
            Reshaped_X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1], 1))
        else:
            print("Error: model input shape length %d is not expected ..." % len(input_layer_shape))
            sys.exit(-1)
        y = ResNetSCA.rws_multilabel_to_categorical(Y_profiling) if self.rws else ResNetSCA.multilabel_to_categorical(Y_profiling)
        history = self.model.fit(x=Reshaped_X_profiling, y=y, batch_size=self.batch_size, verbose = 1, validation_split=validation_split, epochs=self.epochs, callbacks=callbacks)
        return history

def prepare_data_dl(rws, traces_total, round_perms_labels, copy_perms_labels, masks_labels, rws_perms_labels, rws_masks_labels):
    if rws:
        X_total = np.concatenate((
            traces_total[:, RWS_SUBTRACE],
            ), axis=1) # Size: 5000 features
        y_total = {"rws_perm": rws_perms_labels}
        for keyround_idx in range(RWS_SUBKEYWIDTH):
            for share_idx in range(NR_SHARES):
                y_total[f'rws_mask_{keyround_idx}_{share_idx}'] = rws_masks_labels[keyround_idx, share_idx]
    else:
        X_total = np.concatenate((
            traces_total[:, 19350:21250],
    #          traces_total[:, 23950:24400],
    #          traces_total[:, 26700:27150],
    #          traces_total[:, 29450:29900],
    #          traces_total[:, 31450:35500],
    #          traces_total[:, 37650:37750],
    #          traces_total[:, 39750:39850],
    #          traces_total[:, 42100:42200],
    #          traces_total[:, 44950:45050],
    #          traces_total[:, 53350:53450],
    #          traces_total[:, 53750:53850],
    #          traces_total[:, 56150:56250],
            traces_total[:, 61900:61950]), axis=1) # Size: 1950 features
    #          traces_total[:, 75350:75700],
    #          traces_total[:, 77500:77800]), axis=1) # Size: 8700 features
    #        traces_total[:, 75350:77800]), axis=1) # Size: 4400 features

        y_total = {"round_perm": round_perms_labels}
        for round_idx in range(EARLIEST_ROUND, LATEST_ROUND):
            y_total[f'block_perm_{round_idx}'] = copy_perms_labels[round_idx]
            for block_idx in range(BLOCK_WIDTH_B4):
                y_total[f'mask_{round_idx}_{block_idx}'] = 16 * masks_labels[round_idx, block_idx, 0] + masks_labels[round_idx, block_idx, 1]

    return X_total, y_total

def extract_key(X_extraction, save_file_name):
    model = load_model(save_file_name)
    Y_extraction = model.predict(X_extraction)

    return Y_extraction