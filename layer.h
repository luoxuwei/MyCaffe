//
// Created by xuwei on 2022/11/17.
//

#ifndef MYCAFFE_LAYER_H
#define MYCAFFE_LAYER_H

#include <vector>
#include <memory>
#include "blob.h"

using std::vector;
using std::shared_ptr;

/*层参数，主要是每一层的细节部分*/
struct LayerParameter {
    /*1.卷积层超参数 */
    int conv_stride;
    int conv_pad;
    int conv_width;
    int conv_height;
    int conv_kernels;

    /*2.池化层超参数*/
    int pool_stride;
    int pool_width;
    int pool_height;

    /*3.全连接层超参数（即该层神经元个数） */
    int fc_kernels;
};

class Layer
{
public:
    Layer(){}
    virtual ~Layer(){}
    virtual void initLayer(const vector<int>& inShape, const string& lname, vector<shared_ptr<Blob>>& in, const LayerParameter& param) = 0;
    /*每一层的Blob尺寸都是由上一层Blob尺寸经过一定计算规则计算得到的，需要为每一层计算输出尺寸的方法。*/
    virtual void calcShape(const vector<int>&inShape, vector<int>&outShape, const LayerParameter& param) = 0;
};

class ConvLayer : public Layer
{
public:
    ConvLayer(){}
    ~ConvLayer(){}
    void initLayer(const vector<int>& inShape, const string& lname, vector<shared_ptr<Blob>>& in, const LayerParameter& param);
    void calcShape(const vector<int>&inShape, vector<int>&outShape, const LayerParameter& param);
};

class ReluLayer : public Layer
{
public:
    ReluLayer(){}
    ~ReluLayer(){}
    void initLayer(const vector<int>& inShape, const string& lname, vector<shared_ptr<Blob>>& in, const LayerParameter& param);
    void calcShape(const vector<int>&inShape, vector<int>&outShape, const LayerParameter& param);
};

class PoolLayer : public Layer
{
public:
    PoolLayer(){}
    ~PoolLayer(){}
    void initLayer(const vector<int>& inShape, const string& lname, vector<shared_ptr<Blob>>& in, const LayerParameter& param);
    void calcShape(const vector<int>&inShape, vector<int>&outShape, const LayerParameter& param);
};

class FcLayer : public Layer
{
public:
    FcLayer(){}
    ~FcLayer(){}
    void initLayer(const vector<int>& inShape, const string& lname, vector<shared_ptr<Blob>>& in, const LayerParameter& param);
    void calcShape(const vector<int>&inShape, vector<int>&outShape, const LayerParameter& param);
};

#endif //MYCAFFE_LAYER_H
