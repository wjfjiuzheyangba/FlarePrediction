import os
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.utils import get_custom_objects

get_custom_objects()["TF_CPP_MIN_LOG_LEVEL"] = '3'
from util.load_data import load_data_list, load_data_M
from config.Config import File202240Config, TrainConfig
from model.NN_model import get_NN_model
from util.scoreClass import Metric
import gc
import keras.backend as K
from config.Enum import FeatureEnum

from pathlib import Path

p = Path(__file__).resolve().parent.parent.parent.__str__()

file_config = File202240Config(p)
train_data_path_list = load_data_list(file_config.train_file)
valid_data_path_list = load_data_list(file_config.valid_file)
train_config = TrainConfig()

feature_enum = FeatureEnum()

for feature_name in feature_enum.all_feature_list:
    for cv_index in range(10):
        # load data
        train_data_path = train_data_path_list[cv_index]
        valid_data_path = valid_data_path_list[cv_index]
        print(train_data_path)
        (x_train, y_train, train_weight_dir), (x_valid, y_valid, valid_weight_dir) = load_data_M(train_data_path,
                                                                                                 train_data_path,
                                                                                                 feature_name)
        # load model
        model = get_NN_model(
            learning_rate=train_config.learning_rate,
            dropout_rate=train_config.dropout_rate,
            seed=train_config.glorot_normal_seed,
            score_metrics=train_config.score_metrics,
            feature_size=1
        )
        # start train

        model_save_path = f'{p}/weights/202240_feature_importance/NN_best≥M_time_steps=1'
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        model_save_path = f'{model_save_path}/NN_M_best_{cv_index}.h5'
        best_valid_TSS = float('-inf')
        for epoch in range(train_config.epoch):
            print(f'Training NN model, Epoch {epoch + 1}/train_config.epoch, CV={cv_index + 1}')
            history = model.fit(
                x_train, y_train,
                batch_size=train_config.batch_size,
                epochs=1,
                verbose=train_config.verbose,
                class_weight=train_weight_dir,
                validation_data=(x_valid, y_valid),
            )
            # start prediction
            y_train_true = y_train.argmax(axis=1)
            y_train = model.predict(x_train)
            y_train_pred = y_train.argmax(axis=1)
            # print metrix
            train_metric = Metric(y_train_true, y_train_pred)
            y_true = y_valid.argmax(axis=1)
            y_pred = model.predict(x_valid, batch_size=train_config.batch_size).argmax(axis=1)
            # start evaluate
            y_pred_train = model.predict(x_train)
            y_true_train = y_train.argmax(axis=-1)
            y_pred_train = y_pred_train.argmax(axis=-1)
            m_train = Metric(y_true_train, y_pred_train)
            y_pred_valid = model.predict(x_valid)
            y_true_valid = y_valid.argmax(axis=-1)
            y_pred_valid = y_pred_valid.argmax(axis=-1)
            m_valid = Metric(y_true_valid, y_pred_valid)

            if best_valid_TSS <= m_valid.TSS()[1]:
                model.save(model_save_path)
                print(f'best valid TSS from {best_valid_TSS} to {m_valid.TSS()[1]}')
                best_valid_tss = m_valid.TSS()[1]
            else:
                print(f'best valid TSS is: {best_valid_TSS}, now valid TSS is: {m_valid.TSS()[1]}')
            gc.collect()
        K.clear_session()
        gc.collect()
