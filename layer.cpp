//
// Created by xuwei on 2022/11/18.
//
#include <cassert>
#include "layer.h"

using namespace std;
using namespace arma;

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

void ConvLayer::forward(const vector<shared_ptr<Blob>>& in, shared_ptr<Blob>& out, const LayerParameter& param)
{
    cout << "ConvLayer::forward()..." << endl;
    if (out)
        out.reset();
    //-------step1.获取相关尺寸（输入，卷积核，输出）
    assert(in[0]->get_C() == in[1]->get_C());  //断言：输入Blob通道数和卷积核Blob通道数一样（务必保证这一点）

    int N = in[0]->get_N();        //输入Blob中cube个数（该batch样本个数）
    int C = in[0]->get_C();         //输入Blob通道数
    int Hx = in[0]->get_H();      //输入Blob高
    int Wx = in[0]->get_W();    //输入Blob宽

    int F = in[1]->get_N();		  //卷积核个数
    int Hw = in[1]->get_H();     //卷积核高
    int Ww = in[1]->get_W();   //卷积核宽

    int Ho = (Hx + param.conv_pad * 2 - Hw) / param.conv_stride + 1;    //输出Blob高（卷积后）
    int Wo = (Wx + param.conv_pad * 2 - Ww) / param.conv_stride + 1;  //输出Blob宽（卷积后）
    //-------step2.根据要求做padding操作
    Blob padX = in[0]->pad(param.conv_pad);
    out.reset(new Blob(N, F, Ho, Wo));
    for (int n = 0; n < N; ++n)   //输出cube数
    {
        for (int f = 0; f < F; ++f)  //输出通道数
        {
            for (int hh = 0; hh < Ho; ++hh)   //输出Blob的高
            {
                for (int ww = 0; ww < Wo; ++ww)   //输出Blob的宽
                {
                    cube window = padX[n](	span(hh*param.conv_stride, hh*param.conv_stride + Hw - 1),
                                              span(ww*param.conv_stride, ww*param.conv_stride + Ww - 1),
                                              span::all);//span::all表示所有通道。这个位置的参数是传入通道的起始位置，我们是要全部通道。
                    //out = Wx+b，%是cube重载的运算符，就是对应位置元素相乘，accu是求一个cube里所有元素的和。
                    //b是一个cube不是数值，要把里面的数值取出来，通常可以传入坐标(*in[2])[f](0,0,0)，as_scalar是arma提供的做这样操作的方法
                    (*out)[n](hh, ww, f) = accu(window % (*in[1])[f]) + as_scalar((*in[2])[f]);    //b = (F,1,1,1)
                }
            }
        }
    }
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

void ReluLayer::forward(const vector<shared_ptr<Blob>>& in, shared_ptr<Blob>& out, const LayerParameter& param)
{
    cout << "ReluLayer::forward()..." << endl;
    if (out)
        out.reset();

    out.reset(new Blob(*in[0]));//会调用Blob的拷贝构造函数，这样out里的内容就和in一样。
    out->maxIn(0);
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

void PoolLayer::forward(const vector<shared_ptr<Blob>>& in, shared_ptr<Blob>& out, const LayerParameter& param)
{
    cout << "PoolLayer::forward()..." << endl;
    if (out)
        out.reset();
    //-------step1.获取相关尺寸（输入，池化核，输出）
    int N = in[0]->get_N();        //输入Blob中cube个数（该batch样本个数）
    int C = in[0]->get_C();         //输入Blob通道数
    int Hx = in[0]->get_H();      //输入Blob高
    int Wx = in[0]->get_W();    //输入Blob宽

    int Hw = param.pool_height;     //池化核高
    int Ww = param.pool_width;   //池化核宽

    //池化层没有padding
    int Ho = (Hx  - Hw) / param.pool_stride + 1;    //输出Blob高（池化后）
    int Wo = (Wx - Ww) / param.pool_stride + 1;  //输出Blob宽（池化后）

    //-------step2.开始池化
    out.reset(new Blob(N, C, Ho, Wo));//输出通道数等于输入通道数，这点与卷积核不同

    for (int n = 0; n < N; ++n)   //输出cube数
    {
        for (int c = 0; c < C; ++c)  //输出通道数
        {
            for (int hh = 0; hh < Ho; ++hh)   //输出Blob的高
            {
                for (int ww = 0; ww < Wo; ++ww)   //输出Blob的宽
                {
                    (*out)[n](hh, ww, c) = (*in[0])[n](span(hh*param.pool_stride, hh*param.pool_stride + Hw - 1),
                                                       span(ww*param.pool_stride, ww*param.pool_stride + Ww - 1),
                                                       span(c, c)).max();
                }
            }
        }
    }

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

void FcLayer::forward(const vector<shared_ptr<Blob>>& in, shared_ptr<Blob>& out, const LayerParameter& param)
{
    cout << "FcLayer::forward()..." << endl;
    return;
}
