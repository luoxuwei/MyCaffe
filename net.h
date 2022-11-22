//
// Created by xuwei on 2022/11/17.
//

#ifndef MYCAFFE_NET_H
#define MYCAFFE_NET_H

#include <iostream>
#include <vector>
#include <unordered_map>
#include <memory>

#include "layer.h"
#include "blob.h"
#include "snapshot.pb.h"

using std::unordered_map;
using std::vector;
using std::string;
using std::shared_ptr;

struct NetParameter {
    /*学习率*/
    double lr;
    /*学习率衰减系数*/
    double lr_decay;
    /*优化算法,:sgd/momentum/rmsprop*/
    std::string optimizer;
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

class Net
{
public:
    void initNet(NetParameter& param, vector<shared_ptr<Blob>>& X, vector<shared_ptr<Blob>>& Y);
    void trainNet(NetParameter& param);
    void train_with_batch(shared_ptr<Blob>&  X, shared_ptr<Blob>&  Y, NetParameter& param, string mode="TRAIN");
    void optimizer_with_batch(NetParameter& param);
    void evaluate_with_batch(NetParameter& param);
    double calc_accuracy(Blob& Y, Blob& Predict);
    void saveModelParam(shared_ptr<MyCaffe::Snapshot>& snapshot_model);
    void loadModelParam(const shared_ptr<MyCaffe::Snapshot>& snapshot_model);
private:
    // 训练集
    shared_ptr<Blob> X_train_;
    shared_ptr<Blob> Y_train_;
    // 验证集
    shared_ptr<Blob> X_val_;
    shared_ptr<Blob> Y_val_;


    vector<string> layers_;  //层名
    vector<string> ltypes_; //层类型
    double loss_;
    double train_accu_;
    double val_accu_;
    //
    unordered_map<string, vector<shared_ptr<Blob>>> data_;    //前向计算需要用到的Blob data_[0]=X,  data_[1]=W,data_[2] = b;
    unordered_map<string, vector<shared_ptr<Blob>>> diff_;    //反向计算需要用到的Blob diff_[0]=dX,  diff_[1]=dW,diff_[2] = db;
    unordered_map<string, shared_ptr<Layer>> myLayers_;
    unordered_map<string,vector<int>> outShapes_;    //存储每一层的输出尺寸
};

#endif //MYCAFFE_NET_H
