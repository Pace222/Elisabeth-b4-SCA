import os

from typing import Tuple, Dict

import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Input, Conv1D, AveragePooling1D, BatchNormalization, Activation, add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, History
from tensorflow.keras.utils import to_categorical, Sequence

from utils import *

"""
Different model configurations
"""
PARAMS_LIST = {
    "rws_first_quarter": {
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
    "round_0": {
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
        "subtrace": np.concatenate((
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

params = {}

def check_file_exists(file_path: str):
    """Check that the given file path exists.

    Args:
        file_path (str): File path

    Raises:
        ValueError: If the given file does not exist.
    """
    file_path = os.path.normpath(file_path)
    if os.path.exists(file_path) == False:
        raise ValueError("Error: provided file path '%s' does not exist!" % file_path)

class DataGenerator(Sequence):
    def __init__(self, x_set: np.ndarray, y_set: np.ndarray):
        """Generator of data that returns data by batches through __getitem__.
        This function adds a channel dimension to the input and transforms the output variables to categorical ones.

        Args:
            x_set (np.ndarray): The input features
            y_set (np.ndarray): The output features
        """
        self.x = x_set.reshape((x_set.shape[0], x_set.shape[1], 1))

        self.y = {}
        if params["rws_perm"]:
            self.y.update(ResNetSCA.rws_perm_multilabel_to_categorical(y_set))
        if params["rws"]:
            self.y.update(ResNetSCA.rws_multilabel_to_categorical(y_set))
        if params["round_perm"]:
            self.y.update(ResNetSCA.round_perm_multilabel_to_categorical(y_set))
        if params["rounds"]:
            self.y.update(ResNetSCA.rounds_multilabel_to_categorical(y_set))

        self.batch_size = params["batch_size"]

    def __len__(self) -> int:
        """Number of batches in the dataset

        Returns:
            int: The size of the dataset
        """
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Generate batches of input/output pairs

        Args:
            idx (int): The index of the iteration

        Returns:
            Tuple[np.ndarray, Dict[str, np.ndarray]]: Input/output pair
        """
        return self.x[idx * self.batch_size:(idx + 1) * self.batch_size], {k: v[idx * self.batch_size:(idx + 1) * self.batch_size] for k, v in self.y.items()}
    
class ResNetSCA:
    # SOURCE: https://github.com/ANSSI-FR/ASCAD/blob/master/ASCAD_train_models.py
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
        x = Dense(params["perm_nodes"], activation='relu', name='rws_perm')(x)
        x = BatchNormalization()(x)
        x = Dense(KEYROUND_WIDTH_B4, activation="softmax", name='rws_perm_output')(x)
        return x

    @staticmethod
    def rws_mask_branch(x, keyround_idx, share_idx):
        x = Dense(params["mask_nodes"], activation='relu', name=f'rws_mask_{keyround_idx}_{share_idx}')(x)
        x = BatchNormalization()(x)
        x = Dense(len(KEY_ALPHABET), activation="softmax", name=f'rws_mask_{keyround_idx}_{share_idx}_output')(x)
        return x

    @staticmethod
    def round_branch(x):
        x = Dense(params["perm_nodes"], activation='relu', name='round_perm')(x)
        x = BatchNormalization()(x)
        x = Dense(KEYROUND_WIDTH_B4 // BLOCK_WIDTH_B4, activation="softmax", name='round_perm_output')(x)
        return x

    @staticmethod
    def block_branch(x, round_idx):
        x = Dense(params["perm_nodes"], activation='relu', name=f'block_perm_{round_idx}')(x)
        x = BatchNormalization()(x)
        x = Dense(BLOCK_WIDTH_B4, activation="softmax", name=f'block_perm_{round_idx}_output')(x)
        return x
    
    @staticmethod
    def mask_branch(x, round_idx, block_idx, share_idx):
        x = Dense(params["mask_nodes"], activation='relu', name=f'mask_{round_idx}_{block_idx}_{share_idx}')(x)
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

        for keyround_idx in params["rws_subkey_width"]:
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

        for round_idx in params["targeted_rounds"]:
            y[f'block_perm_{round_idx}_output'] = to_categorical(Y[f'block_perm_{round_idx}'], num_classes=BLOCK_WIDTH_B4).astype(np.int8)
            for block_idx in params["targeted_blocks"]:
                for share_idx in params["targeted_shares"]:
                    y[f'mask_{round_idx}_{block_idx}_{share_idx}_output'] = to_categorical(Y[f'mask_{round_idx}_{block_idx}_{share_idx}'], num_classes=len(KEY_ALPHABET)).astype(np.int8)
        return y

    def __init__(self, network: str, epochs: int):
        global params
        params = PARAMS_LIST[network]
        if (params["resnet_depth"] - 1) % 18 != 0:
            raise ValueError('depth should be 18n+1 (eg 19, 37, 55 ...)')
        # Start model definition.
        num_filters = 16
        num_res_blocks = int((params["resnet_depth"] - 1) / 18)
        inputs = Input(shape=(len(params["subtrace"]), 1))
        x = ResNetSCA.resnet_layer(inputs=inputs)
        # Instantiate the stack of residual units as part of the ResNet
        for stack in range(params["num_res_stacks"]):
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

        # Branches of the network
        total_branches = []
        # RWS perm
        if params["rws_perm"]:
            x_rws = ResNetSCA.rws_branch(x)
            total_branches += [x_rws]
        # RWS masks
        if params["rws"]:
            x_rws_masks = []
            for keyround_idx in params["rws_subkey_width"]:
                for share_idx in range(NR_SHARES):
                    x_rws_masks.append(ResNetSCA.rws_mask_branch(x, keyround_idx, share_idx))
            total_branches += x_rws_masks
        # Round perm
        if params["round_perm"]:
            x_round = ResNetSCA.round_branch(x)
            total_branches += [x_round]
        # Block perm and round masks
        if params["rounds"]:
            x_block = []
            x_masks = []
            for round_idx in params["targeted_rounds"]:
                x_block.append(ResNetSCA.block_branch(x, round_idx))
                for block_idx in params["targeted_blocks"]:
                    for share_idx in params["targeted_shares"]:
                        x_masks.append(ResNetSCA.mask_branch(x, round_idx, block_idx, share_idx))
            total_branches += x_block + x_masks

        self.model = Model(inputs, total_branches, name='extract_resnet')
        optimizer = Adam() if not params["rws"] else Adam(beta_1=0.99)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"]*len(total_branches))
        self.epochs = epochs

    def train_model(self, X_y_gen_training: Sequence, X_y_gen_val: Sequence, save_file_name: str, patience: int=10) -> History:
        """Train the model on the given inputs and outputs.

        Args:
            X_y_gen_training (Sequence): Input/output pairs for training
            X_y_gen_val (Sequence): Input/output pairs for validation
            save_file_name (str): Filepath where to save the model
            patience (int, optional): Training patience. Defaults to 10.

        Returns:
            History: History of the model fitting
        """
        check_file_exists(os.path.dirname(save_file_name))
        # Save model calllback
        save_model = ModelCheckpoint(save_file_name, save_best_only=True)
        callbacks=[save_model]
        # Early stopping callback
        callbacks.append(EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True))
        
        history = self.model.fit(x=X_y_gen_training, verbose = 1, validation_data=X_y_gen_val, epochs=self.epochs, callbacks=callbacks)
        return history


def prepare_data_dl(traces: np.ndarray, round_perms_labels: np.ndarray, copy_perms_labels: np.ndarray, round_masks_labels: np.ndarray, rws_perms_labels: np.ndarray, rws_masks_labels: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Fits all labels in a singled dictionary structure that tensorflow can use.

    Args:
        traces (np.ndarray): Traces
        round_perms_labels (np.ndarray): Labels for round perm
        copy_perms_labels (np.ndarray): Labels for block perms
        round_masks_labels (np.ndarray): Labels for round masks
        rws_perms_labels (np.ndarray): Labels for RWS perm
        rws_masks_labels (np.ndarray): Labels for RWS masks

    Returns:
        Tuple[np.ndarray, Dict[str, np.ndarray]]: Input/output pairs for the entire dataset
    """
    X_total = traces

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
                y_total[f'mask_{round_idx}_{block_idx}_{share_idx}'] = round_masks_labels[round_idx, block_idx, share_idx]

    return X_total, y_total
