//
// Created by xuwei on 2022/11/17.
//

#include "net.h"
#include <jsoncpp/json/json.h>
#include <fstream>
#include <cassert>

using namespace std;
void NetParameter::readNetParam(std::string file) {
    ifstream ifs;
    ifs.open(file);
    assert(ifs.is_open());
    Json::Reader reader;  /*解析器*/
    Json::Value value;      /*存储器,解析器解析出来的值就存在value对象里*/
    if (reader.parse(ifs, value))
    {
        if (!value["train"].isNull())
        {
            /*拿到train对象里面的所有元素*/
            auto &tparam = value["train"];
            this->lr = tparam["learning rate"].asDouble();
            this->lr_decay = tparam["lr decay"].asDouble();
            this->update = tparam["update method"].asString();
            this->momentum = tparam["momentum parameter"].asDouble();
            this->num_epochs = tparam["num epochs"].asInt();
            this->use_batch = tparam["use batch"].asBool();
            this->batch_size = tparam["batch size"].asInt();
            this->eval_interval = tparam["evaluate interval"].asInt();
            this->lr_update = tparam["lr update"].asBool();
            this->snap_shot = tparam["snapshot"].asBool();
            this->snapshot_interval = tparam["snapshot interval"].asInt();
            this->fine_tune = tparam["fine tune"].asBool();
            this->preTrainModel = tparam["pre train model"].asString();
        }
        if (!value["net"].isNull())
        {
            /*net数组里面的对象是所有的网络层*/
            auto &nparam = value["net"];
            for (int i = 0; i < (int)nparam.size(); ++i)
            {
                auto &ii = nparam[i];
                this->layers.push_back(ii["name"].asString());  //层名称
                this->ltypes.push_back(ii["type"].asString());   //层类型

                if (ii["type"].asString() == "Conv")
                {
                    int num = ii["kernel num"].asInt();
                    int width = ii["kernel width"].asInt();
                    int height = ii["kernel height"].asInt();
                    int pad = ii["pad"].asInt();
                    int stride = ii["stride"].asInt();

                    this->lparams[ii["name"].asString()].conv_stride = stride;
                    this->lparams[ii["name"].asString()].conv_kernels = num;
                    this->lparams[ii["name"].asString()].conv_pad = pad;
                    this->lparams[ii["name"].asString()].conv_width = width;
                    this->lparams[ii["name"].asString()].conv_height = height;
                }
                if (ii["type"].asString() == "Pool")
                {
                    int width = ii["kernel width"].asInt();
                    int height = ii["kernel height"].asInt();
                    int stride = ii["stride"].asInt();
                    this->lparams[ii["name"].asString()].pool_stride = stride;
                    this->lparams[ii["name"].asString()].pool_width = width;
                    this->lparams[ii["name"].asString()].pool_height = height;
                }
                if (ii["type"].asString() == "Fc")
                {
                    int num = ii["kernel num"].asInt();
                    this->lparams[ii["name"].asString()].fc_kernels = num;
                }
            }
        }


    }

}

void Net::initNet(NetParameter& param, vector<shared_ptr<Blob>>& X, vector<shared_ptr<Blob>>& Y)
{
    /*层名*/
    layers_ = param.layers;
    /*层类型*/
    ltypes_ = param.ltypes;
    for (int i = 0; i < layers_.size(); ++i)
    {
        cout << "layer = " << layers_[i] << " ; " << "ltype = " << ltypes_[i] << endl;
    }
    /*数据集*/
    X_train_ = X[0];
    Y_train_ = Y[0];
    X_val_ = X[1];
    Y_val_ = Y[1];
    /*遍历每一层*/
    for (int i = 0; i < (int)layers_.size(); ++i)
    {
        /*为每一层创建前向计算要用到的3个Blob x w b*/
        data_[layers_[i]] = vector<shared_ptr<Blob>>(3, NULL);
        /*为每一层创建反向计算要用到的3个Blob dw db */
        diff_[layers_[i]] = vector<shared_ptr<Blob>>(3, NULL);
        /*存储每一层的输出尺寸*/
        outShapes_[layers_[i]] = vector<int>(4);
    }

    /*完成每一层的W和b的初始化*/
    shared_ptr<Layer> myLayer(NULL);
    vector<int> inShape = { param.batch_size,
                            X_train_->get_C(),
                            X_train_->get_H(),
                            X_train_->get_W() };
    cout << "input -> (" << inShape[0] << ", " << inShape[1] << ", " << inShape[2] << ", " << inShape[3] << ")" << endl;
    /*遍历每一层，最后一层softmax层，没什么初始化的，所以代码里减去了这一层*/
    for (int i = 0; i < (int)layers_.size()-1; ++i)
    {
        string lname = layers_[i];
        string ltype = ltypes_[i];
        //conv1->relu1->pool1->fc1->softmax
        if (ltype == "Conv")
        {
            myLayer.reset(new ConvLayer);
        }
        if (ltype == "Relu")
        {
            myLayer.reset(new ReluLayer);
        }
        if (ltype == "Pool")
        {
            myLayer.reset(new PoolLayer);
        }
        if (ltype == "Fc")
        {
            myLayer.reset(new FcLayer);
        }
        myLayers_[lname] = myLayer;
        /*初始化参数*/
        myLayer->initLayer(inShape, lname, data_[lname], param.lparams[lname]);
        /*计算维度*/
        myLayer->calcShape(inShape, outShapes_[lname], param.lparams[lname]);
        /*本层的输入是前层的输出*/
        inShape.assign(outShapes_[lname].begin(), outShapes_[lname].end());
        cout << lname << "->(" << outShapes_[lname][0] << "," << outShapes_[lname][1] << "," << outShapes_[lname][2] << "," << outShapes_[lname][3] << ")" << endl;
    }

}