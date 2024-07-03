import os
import sys

import numpy as np

#path = r"C:\Users\iot-user\miniconda3\envs\tf-gpu\Lib\site-packages\nvidia\cudnn\bin"
#os.environ['PATH'] += ';'+path
#import tensorflow as tf
#from tensorflow.python.client import device_lib
#devices = device_lib.list_local_devices()
#kept_devices = [d.name[-len("GPU:X"):] for d in devices if "NVIDIA RTX 4500 Ada Generation, pci bus id: 0000:db:00.0" in d.physical_device_desc or d.device_type == "CPU"]
#tf.config.set_visible_devices([d for d in tf.config.list_physical_devices() if any(kept in d.name for kept in kept_devices)])

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Flatten, Dense, Input, Conv1D, AveragePooling1D, BatchNormalization, Activation, add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical, Sequence

from utils import *

PARAMS_LIST = {
    "old_orig_rws_first_24": {
        "rws_perm": True,
        "rws": True,
        "round_perm": False,
        "rounds": False,
        "subtrace": np.arange(500, 5500),
        "rws_subkey_width": np.arange(KEYROUND_WIDTH_B4 // 4),
        "perm_nodes": 1024,
        "mask_nodes": 512,
        "resnet_depth": 19,
        "num_res_stacks": 8,
        "batch_size": 32
    },
    "old_orig_1st_round": {
        "rws_perm": False,
        "rws": False,
        "round_perm": True,
        "rounds": True,
        "subtrace": np.concatenate((
            np.arange(19350, 21250),
            np.arange(61900, 61950),
            np.arange(75350, 77800)
        )),
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
    #          traces_total[:, 75350:75700],
    #          traces_total[:, 77500:77800
        "targeted_rounds": [0],
        "targeted_blocks": np.arange(BLOCK_WIDTH_B4),
        "targeted_shares": np.arange(NR_SHARES),
        "perm_nodes": 1024,
        "mask_nodes": 512,
        "resnet_depth": 19,
        "num_res_stacks": 9,
        "batch_size": 32
    },
    "haar_2_rws_and_14_rounds": {
        "rws_perm": True,
        "rws": True,
        "round_perm": True,
        "rounds": True,
        "subtrace": np.concatenate(( # Haar 2
            np.arange(300, 5500),    # RWS perm + RWS masks
            np.arange(5500, 5900),   # Round 0
            np.arange(8450, 8950),   # Round 1
            np.arange(11550, 12050), # Round 2
            np.arange(14600, 15100), # Round 3
            np.arange(17700, 18200), # Round 4
            np.arange(20750, 21250), # Round 5
            np.arange(23850, 24350), # Round 6
            np.arange(26900, 27400), # Round 7
            np.arange(30000, 30500), # Round 8
            np.arange(33050, 33550), # Round 9
            np.arange(36150, 36650), # Round 10
            np.arange(39225, 39725), # Round 11
            np.arange(42300, 42800), # Round 12
            np.arange(45400, 45900), # Round 13
            np.arange(48465, 48515), # Round perm
        )),
        "rws_subkey_width": np.arange(KEYROUND_WIDTH_B4),
        "targeted_rounds": np.arange(KEYROUND_WIDTH_B4 // BLOCK_WIDTH_B4),
        "targeted_blocks": np.arange(BLOCK_WIDTH_B4),
        "targeted_shares": np.arange(NR_SHARES),
        "perm_nodes": 512,
        "mask_nodes": 128,
        "resnet_depth": 19,
        "num_res_stacks": 9,
        "batch_size": 128
    },
    "haar_2_rws_only": {
        "rws_perm": True,
        "rws": True,
        "round_perm": False,
        "rounds": False,
        "subtrace": np.concatenate((
            np.arange(300, 5500),    # RWS perm + RWS masks
        )),
        "rws_subkey_width": np.arange(KEYROUND_WIDTH_B4),
        "perm_nodes": 512,
        "mask_nodes": 256,
        "resnet_depth": 19,
        "num_res_stacks": 8,
        "batch_size": 64
    },
}
for i, pois in enumerate([
    np.arange(5500, 5900),   # Round 0
    np.arange(8450, 8950),   # Round 1
    np.arange(11550, 12050), # Round 2
    np.arange(14600, 15100), # Round 3
    np.arange(17700, 18200), # Round 4
    np.arange(20750, 21250), # Round 5
    np.arange(23850, 24350), # Round 6
    np.arange(26900, 27400), # Round 7
    np.arange(30000, 30500), # Round 8
    np.arange(33050, 33550), # Round 9
    np.arange(36150, 36650), # Round 10
    np.arange(39225, 39725), # Round 11
    np.arange(42300, 42800), # Round 12
    np.arange(45400, 45900), # Round 13
]):
    PARAMS_LIST[f"haar_2_round_{i}"] = {
        "rws_perm": False,
        "rws": False,
        "round_perm": False,
        "rounds": True,
        "subtrace": np.concatenate((
            pois,
        )),
        "targeted_rounds": [i],
        "targeted_blocks": np.arange(BLOCK_WIDTH_B4),
        "targeted_shares": np.arange(NR_SHARES),
        "perm_nodes": 1024,
        "mask_nodes": 1024,
        "resnet_depth": 19,
        "num_res_stacks": 4,
        "batch_size": 64
    }

PARAMS_LIST["orig_rws_perm_only"] = {
    "rws_perm": True,
    "rws": False,
    "round_perm": False,
    "rounds": False,
    "subtrace": np.concatenate((
        np.arange(1250, 21100),    # RWS perm
    )),
    "perm_nodes": 1024,
    "resnet_depth": 19,
    "num_res_stacks": 10,
    "batch_size": 32
}
PARAMS_LIST["orig_rws_0"] = {
    "rws_perm": False,
    "rws": True,
    "round_perm": False,
    "rounds": False,
    "subtrace": np.concatenate((
        np.arange(1300, 6175),    # RWS masks
    )),
    "rws_subkey_width": np.arange(0, KEYROUND_WIDTH_B4 // 4),
    "mask_nodes": 512,
    "resnet_depth": 19,
    "num_res_stacks": 8,
    "batch_size": 64
}
PARAMS_LIST["orig_rws_1"] = {
    "rws_perm": False,
    "rws": True,
    "round_perm": False,
    "rounds": False,
    "subtrace": np.concatenate((
        np.arange(6175, 11030),    # RWS masks
    )),
    "rws_subkey_width": np.arange(KEYROUND_WIDTH_B4 // 4, 2 * (KEYROUND_WIDTH_B4 // 4)),
    "mask_nodes": 512,
    "resnet_depth": 19,
    "num_res_stacks": 8,
    "batch_size": 64
}
PARAMS_LIST["orig_rws_2"] = {
    "rws_perm": False,
    "rws": True,
    "round_perm": False,
    "rounds": False,
    "subtrace": np.concatenate((
        np.arange(11030, 15890),    # RWS masks
    )),
    "rws_subkey_width": np.arange(2 * (KEYROUND_WIDTH_B4 // 4), 3 * (KEYROUND_WIDTH_B4 // 4)),
    "mask_nodes": 512,
    "resnet_depth": 19,
    "num_res_stacks": 8,
    "batch_size": 64
}
PARAMS_LIST["orig_rws_3"] = {
    "rws_perm": False,
    "rws": True,
    "round_perm": False,
    "rounds": False,
    "subtrace": np.concatenate((
        np.arange(15890, 21160),    # RWS masks
    )),
    "rws_subkey_width": np.arange(3 * (KEYROUND_WIDTH_B4 // 4), KEYROUND_WIDTH_B4),
    "mask_nodes": 512,
    "resnet_depth": 19,
    "num_res_stacks": 8,
    "batch_size": 64
}
PARAMS_LIST["orig_rws_only"] = {
    "rws_perm": True,
    "rws": True,
    "round_perm": False,
    "rounds": False,
    "subtrace": np.concatenate((
        np.arange(1100, 21500),    # RWS perm + RWS masks
    )),
    "rws_subkey_width": np.arange(KEYROUND_WIDTH_B4),
    "perm_nodes": 512,
    "mask_nodes": 256,
    "resnet_depth": 19,
    "num_res_stacks": 10,
    "batch_size": 128
}
PARAMS_LIST["orig_round_perm_only"] = {
    "rws_perm": False,
    "rws": False,
    "round_perm": True,
    "rounds": False,
    "subtrace": np.concatenate((
        np.arange( 21750,  21800),
        np.arange( 33925,  33975),
        np.arange( 46250,  46300),
        np.arange( 58550,  25600),
        np.arange( 70860,  70910),
        np.arange( 83180,  83230),
        np.arange( 95480,  95530),
        np.arange(107800, 107850),
        np.arange(120110, 120160),
        np.arange(132415, 132465),
        np.arange(144725, 144775),
        np.arange(157040, 157090),
        np.arange(169350, 169400),
        np.arange(181660, 181710),
        np.arange(193960, 194010),
    )),
    "perm_nodes": 1024,
    "resnet_depth": 19,
    "num_res_stacks": 5,
    "batch_size": 32
}
for i, pois in enumerate([
    np.arange( 21950,  23350), # Round 0
    np.arange( 34250,  35650), # Round 1
    np.arange( 46550,  47950), # Round 2
    np.arange( 58850,  60250), # Round 3
    np.arange( 71150,  72550), # Round 4
    np.arange( 83500,  84900), # Round 5
    np.arange( 95800,  97200), # Round 6
    np.arange(108100, 109500), # Round 7
    np.arange(120400, 121800), # Round 8
    np.arange(132700, 134100), # Round 9
    np.arange(145025, 146425), # Round 10
    np.arange(157350, 158750), # Round 11
    np.arange(169650, 171050), # Round 12
    np.arange(181950, 183350), # Round 13
]):
    PARAMS_LIST[f"orig_round_{i}"] = {
        "rws_perm": False,
        "rws": False,
        "round_perm": False,
        "rounds": True,
        "subtrace": np.concatenate((
            pois,
        )),
        "targeted_rounds": [i],
        "targeted_blocks": np.arange(BLOCK_WIDTH_B4),
        "targeted_shares": np.arange(NR_SHARES),
        "perm_nodes": 1024,
        "mask_nodes": 512,
        "resnet_depth": 19,
        "num_res_stacks": 6,
        "batch_size": 64
    }
PARAMS_LIST["test_mask_0_0_0"] = {
    "rws_perm": False,
    "rws": False,
    "round_perm": False,
    "rounds": True,
    "subtrace": np.concatenate((
        np.arange(22100, 22170),
    )),
    "targeted_rounds": [0],
    "targeted_blocks": [0],
    "targeted_shares": [0],
    "mask_nodes": 512,
    "resnet_depth": 19,
    "num_res_stacks": 0,
    "batch_size": 64
}



PARAMS = {}

def check_file_exists(file_path):
	file_path = os.path.normpath(file_path)
	if os.path.exists(file_path) == False:
		raise ValueError("Error: provided file path '%s' does not exist!" % file_path)
	return

class DataGenerator(Sequence):
    def __init__(self, x_set, y_set):
        self.x = x_set.reshape((x_set.shape[0], x_set.shape[1], 1))

        self.y = {}
        if PARAMS["rws_perm"]:
            self.y.update(ResNetSCA.rws_perm_multilabel_to_categorical(y_set))
        if PARAMS["rws"]:
            self.y.update(ResNetSCA.rws_multilabel_to_categorical(y_set))
        if PARAMS["round_perm"]:
            self.y.update(ResNetSCA.round_perm_multilabel_to_categorical(y_set))
        if PARAMS["rounds"]:
            self.y.update(ResNetSCA.rounds_multilabel_to_categorical(y_set))

        self.batch_size = PARAMS["batch_size"]

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        return self.x[idx * self.batch_size:(idx + 1) * self.batch_size], {k: v[idx * self.batch_size:(idx + 1) * self.batch_size] for k, v in self.y.items()}
    
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
        x = Dense(PARAMS["perm_nodes"], activation='relu', name='rws_perm')(x)
        x = BatchNormalization()(x)
        x = Dense(KEYROUND_WIDTH_B4, activation="softmax", name='rws_perm_output')(x)
        return x

    @staticmethod
    def rws_mask_branch(x, keyround_idx, share_idx):
        x = Dense(PARAMS["mask_nodes"], activation='relu', name=f'rws_mask_{keyround_idx}_{share_idx}')(x)
        x = BatchNormalization()(x)
        x = Dense(len(KEY_ALPHABET), activation="softmax", name=f'rws_mask_{keyround_idx}_{share_idx}_output')(x)
        return x

    @staticmethod
    def round_branch(x):
        x = Dense(PARAMS["perm_nodes"], activation='relu', name='round_perm')(x)
        x = BatchNormalization()(x)
        x = Dense(KEYROUND_WIDTH_B4 // BLOCK_WIDTH_B4, activation="softmax", name='round_perm_output')(x)
        return x

    @staticmethod
    def block_branch(x, round_idx):
        x = Dense(PARAMS["perm_nodes"], activation='relu', name=f'block_perm_{round_idx}')(x)
        x = BatchNormalization()(x)
        x = Dense(BLOCK_WIDTH_B4, activation="softmax", name=f'block_perm_{round_idx}_output')(x)
        return x
    
    @staticmethod
    def mask_branch(x, round_idx, block_idx, share_idx):
        x = Dense(PARAMS["mask_nodes"], activation='relu', name=f'mask_{round_idx}_{block_idx}_{share_idx}')(x)
        x = BatchNormalization()(x)
        x = Dense(len(KEY_ALPHABET), activation="softmax", name=f'mask_{round_idx}_{block_idx}_{share_idx}_output')(x)
        return x

    @staticmethod
    def rws_perm_multilabel_to_categorical(Y):
        y = {}
        y['rws_perm_output'] = to_categorical(Y['rws_perm'], num_classes=KEYROUND_WIDTH_B4).astype(np.int8)

        return y
    
    @staticmethod
    def rws_multilabel_to_categorical(Y):
        y = {}

        for keyround_idx in PARAMS["rws_subkey_width"]:
            for share_idx in range(NR_SHARES):
                y[f'rws_mask_{keyround_idx}_{share_idx}_output'] = to_categorical(Y[f'rws_mask_{keyround_idx}_{share_idx}'], num_classes=len(KEY_ALPHABET)).astype(np.int8)
        return y

    @staticmethod
    def round_perm_multilabel_to_categorical(Y):
        y = {}
        y['round_perm_output'] = to_categorical(Y['round_perm'], num_classes=KEYROUND_WIDTH_B4 // BLOCK_WIDTH_B4).astype(np.int8)

        return y
    
    @staticmethod
    def rounds_multilabel_to_categorical(Y):
        y = {}

        for round_idx in PARAMS["targeted_rounds"]:
            y[f'block_perm_{round_idx}_output'] = to_categorical(Y[f'block_perm_{round_idx}'], num_classes=BLOCK_WIDTH_B4).astype(np.int8)
            for block_idx in PARAMS["targeted_blocks"]:
                for share_idx in PARAMS["targeted_shares"]:
                    y[f'mask_{round_idx}_{block_idx}_{share_idx}_output'] = to_categorical(Y[f'mask_{round_idx}_{block_idx}_{share_idx}'], num_classes=len(KEY_ALPHABET)).astype(np.int8)
        return y

    def __init__(self, network, epochs, dataset_size):
        global PARAMS
        PARAMS = PARAMS_LIST[network]
        if (PARAMS["resnet_depth"] - 1) % 18 != 0:
            raise ValueError('depth should be 18n+1 (eg 19, 37, 55 ...)')
        # Start model definition.
        num_filters = 16
        num_res_blocks = int((PARAMS["resnet_depth"] - 1) / 18)
        inputs = Input(shape=(len(PARAMS["subtrace"]), 1))
        x = ResNetSCA.resnet_layer(inputs=inputs)
        # Instantiate the stack of residual units
        for stack in range(PARAMS["num_res_stacks"]):
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

        total = []
        if PARAMS["rws_perm"]:
            x_rws = ResNetSCA.rws_branch(x)
            total += [x_rws]
        if PARAMS["rws"]:
            x_rws_masks = []
            for keyround_idx in PARAMS["rws_subkey_width"]:
                for share_idx in range(NR_SHARES):
                    x_rws_masks.append(ResNetSCA.rws_mask_branch(x, keyround_idx, share_idx))
            total += x_rws_masks
        if PARAMS["round_perm"]:
            x_round = ResNetSCA.round_branch(x)
            total += [x_round]
        if PARAMS["rounds"]:
            x_block = []
            x_masks = []
            for round_idx in PARAMS["targeted_rounds"]:
                x_block.append(ResNetSCA.block_branch(x, round_idx))
                for block_idx in PARAMS["targeted_blocks"]:
                    for share_idx in PARAMS["targeted_shares"]:
                        x_masks.append(ResNetSCA.mask_branch(x, round_idx, block_idx, share_idx))
            total += x_block + x_masks

        self.model = Model(inputs, total, name='extract_resnet')
        #optimizer = Adam(learning_rate=ExponentialDecay(initial_learning_rate=0.001, decay_steps=epochs*dataset_size/PARAMS["batch_size"], decay_rate=0.9))
        optimizer = Adam() if not PARAMS["rws"] else Adam(beta_1=0.99)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"]*len(total))
        self.epochs = epochs
        self.dataset_size = dataset_size

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

        y = {}
        if PARAMS["rws_perm"]:
            self.y.update(ResNetSCA.rws_perm_multilabel_to_categorical(Y_profiling))
        if PARAMS["rws"]:
            y.update(ResNetSCA.rws_multilabel_to_categorical(Y_profiling))
        if PARAMS["round_perm"]:
            y.update(ResNetSCA.round_perm_multilabel_to_categorical(Y_profiling))
        if PARAMS["rounds"]:
            y.update(ResNetSCA.rounds_multilabel_to_categorical(Y_profiling))

        history = self.model.fit(x=Reshaped_X_profiling, y=y, batch_size=PARAMS["batch_size"], verbose = 1, validation_split=validation_split, epochs=self.epochs, callbacks=callbacks)
        return history

    def train_model_generator(self, X_y_gen_training, X_y_gen_val, save_file_name, patience=10):
        check_file_exists(os.path.dirname(save_file_name))
        # Save model calllback
        save_model = ModelCheckpoint(save_file_name, save_best_only=True)
        callbacks=[save_model]
        # Early stopping callback
        callbacks.append(EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True))
        
        history = self.model.fit(x=X_y_gen_training, verbose = 1, validation_data=X_y_gen_val, epochs=self.epochs, callbacks=callbacks)
        return history


def prepare_data_dl(traces_total, round_perms_labels, copy_perms_labels, masks_labels, rws_perms_labels, rws_masks_labels):
    X_total = traces_total

    y_total = {}
    y_total["rws_perm"] = rws_perms_labels
    for keyround_idx in range(KEYROUND_WIDTH_B4):
        for share_idx in range(NR_SHARES):
            y_total[f'rws_mask_{keyround_idx}_{share_idx}'] = rws_masks_labels[keyround_idx, share_idx]
    y_total["round_perm"] = round_perms_labels
    for round_idx in range(EARLIEST_ROUND, LATEST_ROUND):
        y_total[f'block_perm_{round_idx}'] = copy_perms_labels[round_idx]
        for block_idx in range(BLOCK_WIDTH_B4):
            for share_idx in range(NR_SHARES):
                y_total[f'mask_{round_idx}_{block_idx}_{share_idx}'] = masks_labels[round_idx, block_idx, share_idx]

    return X_total, y_total
