import os

import keras.backend as K

from config.Config import TrainConfig
from util.get_model_path import get_model_path
from util.load_data import Rectify
from util.load_data import load_data_C, load_data_M
from util.load_data import load_data_list
from util.score import show_score_and_save_weights
from util.set_seed import set_seed
from util.show_pic_util import save_loss


def train(p: str, file_config, data_type: str, model_name, class_type, get_model) -> None:
    """
    :param p: 根目录地址
    :param file_config: train_config.Config
    :param data_type: TT, TVT, 2018, 2022, 202240
    :param model_name: 模型名称
    :param class_type: 分类类型  C M
    :param get_model: 获取模型函数
    """
    train_config = TrainConfig()
    train_list = load_data_list(file_config.train_file)
    valid_list = load_data_list(file_config.valid_file)
    best_TSS_dir = {}
    if model_name == 'NN':
        time_steps_list = [1]
    else:
        time_steps_list = train_config.time_steps_list
    for time_steps in time_steps_list:
        is_new = True
        best_TSS_list = []  # 保存每个训练集的最好的TSS
        if model_name == 'NN':
            model_save_path = p + '/weights/' + data_type + '/NN_best≥' + class_type + '_time_steps=' + str(time_steps)
        else:
            model_save_path = p + '/weights/' + data_type + '/' + model_name + '_best≥' + class_type + '/time_steps=' + str(
                time_steps)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        for i in range(len(train_list)):
            # 设置随机数种子，方便复现结果
            set_seed()
            # 导入训练数据
            if class_type == 'C':
                (x_train, y_train, train_weight_dir), \
                (x_valid, y_valid, valid_weight_dir) = load_data_C(train_list[i], valid_list[i])
            else:
                (x_train, y_train, train_weight_dir), \
                (x_valid, y_valid, valid_weight_dir) = load_data_M(train_list[i], valid_list[i])
            # 根据时间步修改训练和测试数据形状
            if model_name != 'NN':
                x_train = x_train.reshape(-1, time_steps, 10)
                x_valid = x_valid.reshape(-1, time_steps, 10)
                y_train = Rectify(y_train, time_steps)
                y_valid = Rectify(y_valid, time_steps)
            # 获取模型
            model = get_model(
                time_steps=time_steps,
                learning_rate=train_config.learning_rate,
                dropout_rate=train_config.dropout_rate,
                seed=train_config.glorot_normal_seed,
                score_metrics=train_config.score_metrics
            )
            # 评价指标初始化
            best_TSS = float('-inf')
            loss_list, val_loss_list = [], []
            # 设置训练代数
            for j in range(train_config.epoch):
                # 打印当前训练的时间步、训练集、验证集和代数
                print("time_steps =", time_steps)
                print(train_list[i])
                print(valid_list[i])
                print('Epoch ' + str(j + 1) + '/' + str(train_config.epoch))
                # 开始训练并获取损失函数值
                history = model.fit(
                    x_train, y_train,
                    batch_size=train_config.batch_size,
                    epochs=1,
                    verbose=train_config.verbose,
                    class_weight=train_weight_dir,  # {dict, 'balanced'},
                    validation_data=(x_valid, y_valid),
                )
                # 开始评价
                y_true = y_valid.argmax(axis=1)  # 真实的标签
                y_pred = model.predict(x_valid, batch_size=train_config.batch_size).argmax(axis=1)  # 将数据传入，得到预测的标签
                # 获取模型保存的路径
                if model_name == 'NN':
                    filename = get_model_path(p, data_type, class_type, model_name, 1, i)
                else:
                    filename = get_model_path(p, data_type, class_type, model_name, time_steps, i)
                best_TSS = show_score_and_save_weights(  # 计算最好的TSS，并保存取得最好的TSS的权重
                    model=model,
                    best_TSS=best_TSS,
                    y_true=y_true, y_pred=y_pred,
                    filename=filename
                )
                loss_list.append(history.history['loss'])
                val_loss_list.append(history.history['val_loss'])
                print('==================' + model_name + '====================')
            if class_type == 'NN':
                img_path = model_save_path + '/' + model_name + '_' + class_type + '_best_' + str(i) + '.jpg'
            else:
                img_path = model_save_path + '/' + model_name + '_' + class_type + '_' + str(
                    time_steps) + '_best_' + str(i) + '.jpg'
            save_loss(loss_list, val_loss_list, train_config.epoch, img_path, is_new)
            is_new = False
            best_TSS_list.append(best_TSS)
            K.clear_session()
        # 全部训练完成后，打印所有权重的指标
        print('time_steps =', time_steps)
        best_TSS_dir[time_steps] = best_TSS_list
        for best_TSS in best_TSS_list:
            print(best_TSS)
    print(best_TSS_dir)
