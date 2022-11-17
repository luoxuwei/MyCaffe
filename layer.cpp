//
// Created by xuwei on 2022/11/18.
//

#include "layer.h"

using namespace std;

void ConvLayer::initLayer(const vector<int>& inShape, const string& lname, vector<shared_ptr<Blob>>& in, const LayerParameter& param)
{
    //1.获取卷积核尺寸(F,C,H,W)
    int tF = param.conv_kernels;
    int tC = inShape[1];
    int tH = param.conv_height;
    int tW = param.conv_width;

    //2.初始化存储W和b的Blob  (in[1]->W和in[2]->b)
    if (!in[1])   //存储W的Blob不为空
    {
        in[1].reset(new Blob(tF, tC, tH, tW, TRANDN));  //标准高斯初始化（μ= 0和σ= 1）标准差为1太大，初始化的权值要尽量小    //np.randn()*0.01
        (*in[1]) *= 1e-2;
        cout << "initLayer: " << lname << "  Init weights  with standard Gaussian ;" << endl;
    }
    if (!in[2])   //存储b的Blob不为空
    {
        /*不论卷积核的通道是多少都只有一个b*/
        in[2].reset(new Blob(tF, 1, 1, 1, TRANDN));  //标准高斯初始化（μ= 0和σ= 1）    //np.randn()*0.01
        (*in[2]) *= 1e-2;
        cout << "initLayer: " << lname << "  Init bias  with standard Gaussian ;" << endl;
    }
    return;
}

void ConvLayer::calcShape(const vector<int>&inShape, vector<int>&outShape, const LayerParameter& param)
{
    //1.获取输入Blob尺寸
    int Ni = inShape[0]; //batch size 直接传递不会变
    int Ci = inShape[1];
    int Hi = inShape[2];
    int Wi = inShape[3];
    //2.获取卷积核尺寸
    int tF = param.conv_kernels;   //卷积核个数（由层名称索引得到），就是输出特征图的通道数
    int tH = param.conv_height;   //卷积核高
    int tW = param.conv_width;    //卷积核宽
    int tP = param.conv_pad;        //padding数
    int tS = param.conv_stride;    //滑动步长
    //3.计算卷积后的尺寸
    int No = Ni;
    int Co = tF;
    int Ho = (Hi + 2 * tP - tH) / tS + 1;    //卷积后图片高度
    int Wo = (Wi + 2 * tP - tW) / tS + 1;  //卷积后图片宽度
    //4.赋值输出Blob尺寸
    outShape[0] = No;
    outShape[1] = Co;
    outShape[2] = Ho;
    outShape[3] = Wo;
    return;
}

/*relu和池化层没有参数*/
void ReluLayer::initLayer(const vector<int>& inShape, const string& lname, vector<shared_ptr<Blob>>& in, const LayerParameter& param)
{
    cout << "ReluLayer::initLayer()  ok!!!" << endl;
    return;
}

void ReluLayer::calcShape(const vector<int>&inShape, vector<int>&outShape, const LayerParameter& param)
{
    outShape.assign(inShape.begin(), inShape.end());//将inShape复制一份给outShape（深拷贝）
    return;
}

void PoolLayer::initLayer(const vector<int>& inShape, const string& lname, vector<shared_ptr<Blob>>& in, const LayerParameter& param)
{
    cout << "PoolLayer::initLayer()  ok!!!" << endl;
    return;
}

void PoolLayer::calcShape(const vector<int>&inShape, vector<int>&outShape, const LayerParameter& param)
{
    //1.获取输入Blob尺寸
    int Ni = inShape[0];
    int Ci = inShape[1];
    int Hi = inShape[2];
    int Wi = inShape[3];
    //2.获取池化核尺寸
    int tH = param.pool_height;    //池化核高
    int tW = param.pool_width;    //池化核宽
    int tS = param.pool_stride;      //池化核滑动步长
    //3.计算池化后的尺寸
    int No = Ni;
    int Co = Ci;
    int Ho = (Hi - tH) / tS + 1;    //卷积后图片高度
    int Wo = (Wi - tW) / tS + 1;  //卷积后图片宽度
    //4.赋值输出Blob尺寸
    outShape[0] = No;
    outShape[1] = Co;
    outShape[2] = Ho;
    outShape[3] = Wo;
    return;
}

void FcLayer::initLayer(const vector<int>& inShape, const string& lname, vector<shared_ptr<Blob>>& in, const LayerParameter& param)
{
    //1.获取全连接核尺寸(F,C,H,W)
    int tF = param.fc_kernels;
    int tC = inShape[1];
    int tH = inShape[2];
    int tW = inShape[3];

    //2.初始化存储W和b的Blob  (in[1]->W和in[2]->b)
    if (!in[1])   //存储W的Blob不为空
    {
        in[1].reset(new Blob(tF, tC, tH, tW, TRANDN));  //标准高斯初始化（μ= 0和σ= 1）    //np.randn()*0.01
        (*in[1]) *= 1e-2;
        cout << "initLayer: " << lname << "  Init weights  with standard Gaussian ;" << endl;
    }
    if (!in[2])   //存储b的Blob不为空
    {
        in[2].reset(new Blob(tF, 1, 1, 1, TZEROS));  //标准高斯初始化（μ= 0和σ= 1）    //np.randn()*0.01
        cout << "initLayer: " << lname << "  Init bias  with Zeros ;" << endl;
    }
    return;
}

void FcLayer::calcShape(const vector<int>&inShape, vector<int>&outShape, const LayerParameter& param)
{
    //1.计算输出Blob尺寸
    int No = inShape[0];                 //该batch的样本数
    int Co = param.fc_kernels;		   //该层神经元个数
    int Ho = 1;
    int Wo = 1;
    /*神经元个数就是分类数就是输出维度,当然没有宽高了，只有每个分类的得分*/
    //（200,10,1,1）
    //2.赋值
    outShape[0] = No;
    outShape[1] = Co;
    outShape[2] = Ho;
    outShape[3] = Wo;
    return;
}
