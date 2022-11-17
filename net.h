//
// Created by xuwei on 2022/11/17.
//

#ifndef MYCAFFE_NET_H
#define MYCAFFE_NET_H

#include <iostream>
#include <vector>
#include <unordered_map>

#include "layer.h"

using std::unordered_map;
using std::vector;
using std::string;

struct NetParameter {
    /*学习率*/
    double lr;
    /*学习率衰减系数*/
    double lr_decay;
    /*优化算法,:sgd/momentum/rmsprop*/
    string update;
    /*momentum系数 */
    double momentum;
    /*epoch次数 */
    int num_epochs;
    /*是否使用mini-batch梯度下降*/
    bool use_batch;
    /*每批次样本个数*/
    int batch_size;
    /*每隔几个迭代周期评估一次准确率？ */
    int eval_interval;
    /*是否更新学习率？  true/false*/
    bool lr_update;
    /* 是否保存模型快照；快照保存间隔*/
    bool snap_shot;
    /*每隔几个迭代周期保存一次快照？*/
    int snapshot_interval;
    /* 是否采用fine-tune方式训练*/
    bool fine_tune;
    /*预训练模型文件.gordonmodel所在路径*/
    string preTrainModel;

    /*层名*/
    vector <string> layers;
    /*层类型*/
    vector <string> ltypes;

    /*通过层名访问层参数*/
    unordered_map<string, LayerParameter> lparams;

    void readNetParam(string file);
};

#endif //MYCAFFE_NET_H
