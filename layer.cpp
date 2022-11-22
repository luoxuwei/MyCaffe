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

void ConvLayer::calcShape(const vector<int>& inShape, vector<int>& outShape, const LayerParameter& param)
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

void ConvLayer::backward(	const shared_ptr<Blob>& din,   //输入梯度
                             const vector<shared_ptr<Blob>>& cache,
                             vector<shared_ptr<Blob>>& grads,
                             const LayerParameter& param		)
{
    cout << "ConvLayer::backward()..." << endl;
    //step1. 设置输出梯度Blob的尺寸（dX---grads[0]）
    grads[0].reset(new Blob(cache[0]->size(), TZEROS));
    grads[1].reset(new Blob(cache[1]->size(), TZEROS));
    grads[2].reset(new Blob(cache[2]->size(), TZEROS));
    //step2. 获取输入梯度Blob的尺寸（din）
    int Nd = din->get_N();        //输入梯度Blob中cube个数（该batch样本个数）
    int Cd = din->get_C();         //输入梯度Blob通道数
    int Hd = din->get_H();      //输入梯度Blob高
    int Wd = din->get_W();    //输入梯度Blob宽
    //step3. 获取卷积核相关参数
    int Hw = param.conv_height;
    int Ww = param.conv_width;
    int stride = param.conv_stride;

    //step4. 填充操作
    Blob pad_X = cache[0]->pad(param.conv_pad);  //参与实际反向传播计算的应该是填充过的特征Blob
    Blob pad_dX(pad_X.size(),TZEROS);                      //梯度Blob应该与该层的特征Blob尺寸保持一致

    //step5. 开始反向传播
    for (int n = 0; n < Nd; ++n)   //遍历输入梯度din的样本数
    {
        for (int c = 0; c < Cd; ++c)  //遍历输入梯度din的通道数
        {
            for (int hh = 0; hh < Hd; ++hh)   //遍历输入梯度din的高
            {
                for (int ww = 0; ww < Wd; ++ww)   //遍历输入梯度din的宽
                {
                    //(1). 通过滑动窗口，截取不同输入特征片段
                    cube window = pad_X[n](span(hh*stride, hh*stride + Hw - 1),span(ww*stride, ww*stride + Ww - 1),span::all);
                    //(2). 计算梯度
                    //dX
                    pad_dX[n](span(hh*stride, hh*stride + Hw - 1), span(ww*stride, ww*stride + Ww - 1), span::all)   +=   (*din)[n](hh, ww, c) * (*cache[1])[c];
                    //dW  --->grads[1]
                    (*grads[1])[c] += (*din)[n](hh, ww, c) * window  / Nd;
                    //db   --->grads[2]
                    (*grads[2])[c](0,0,0) += (*din)[n](hh, ww, c) / Nd;
                }
            }
        }
    }

    //step6. 去掉输出梯度中的padding部分
    (*grads[0]) = pad_dX.deletePad(param.conv_pad);

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

void ReluLayer::backward(const shared_ptr<Blob>& din,
                         const vector<shared_ptr<Blob>>& cache,
                         vector<shared_ptr<Blob>>& grads,
                         const LayerParameter& param)
{
    cout << "ReluLayer::backward()..." << endl;
    //step1. 设置输出梯度Blob的尺寸（dX---grads[0]）
    grads[0].reset(new Blob(*cache[0]));

    //step2. 获取掩码mask
    int N = grads[0]->get_N();
    for (int n = 0; n < N; ++n)
    {
        (*grads[0])[n].transform([](double e) {return e > 0 ? 1 : 0; });
    }
    (*grads[0]) = (*grads[0]) * (*din);

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

void PoolLayer::backward(const shared_ptr<Blob>& din,
                         const vector<shared_ptr<Blob>>& cache,
                         vector<shared_ptr<Blob>>& grads,
                         const LayerParameter& param)
{
    cout << "PoolLayer::backward()..." << endl;
    //step1. 设置输出梯度Blob的尺寸（dX---grads[0]）池化层没有w和b
    grads[0].reset(new Blob(cache[0]->size(), TZEROS));
    //step2. 获取输入梯度Blob的尺寸（din）
    int Nd = din->get_N();        //输入梯度Blob中cube个数（该batch样本个数）
    int Cd = din->get_C();         //输入梯度Blob通道数
    int Hd = din->get_H();      //输入梯度Blob高
    int Wd = din->get_W();    //输入梯度Blob宽

    //step3. 获取池化核相关参数
    int Hp = param.pool_height;
    int Wp = param.pool_width;
    int stride = param.pool_stride;

    //step4. 开始反向传播
    for (int n = 0; n < Nd; ++n)   //输出cube数
    {
        for (int c = 0; c < Cd; ++c)  //输出通道数
        {
            for (int hh = 0; hh < Hd; ++hh)   //输出Blob的高
            {
                for (int ww = 0; ww < Wd; ++ww)   //输出Blob的宽
                {
                    //(1). 获取掩码mask
                    mat window = (*cache[0])[n](span(hh*param.pool_stride, hh*param.pool_stride + Hp - 1),
                                                span(ww*param.pool_stride, ww*param.pool_stride + Wp - 1),
                                                span(c, c));
                    double maxv = window.max();
                    //maxv是一个数值，==号的作用是让window里的每一个值如果与maxv相等就置为true(1)，不相等为false(0)
                    mat mask = conv_to<mat>::from(maxv == window);  //"=="返回的是一个umat类型的矩阵！就是里面的元素都是unsigned 无符号的，umat转换为mat
                    //(2). 计算梯度
                    (*grads[0])[n](span(hh*param.pool_stride, hh*param.pool_stride + Hp - 1),
                                   span(ww*param.pool_stride, ww*param.pool_stride + Wp - 1),
                                   span(c, c))       +=       mask*(*din)[n](hh, ww, c);  //umat  -/-> mat
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
    if (out)
        out.reset();
    //-------step1.获取相关尺寸（输入，全连接核，输出）
    int N = in[0]->get_N();        //输入Blob中cube个数（该batch样本个数）
    int C = in[0]->get_C();         //输入Blob通道数
    int Hx = in[0]->get_H();      //输入Blob高
    int Wx = in[0]->get_W();    //输入Blob宽

    int F = in[1]->get_N();		  //全连接核个数
    int Hw = in[1]->get_H();     //全连接核高
    int Ww = in[1]->get_W();   //全连接核宽
    assert(in[0]->get_C() == in[1]->get_C());  //输入Blob通道数和全连接核Blob通道数一样（务必保证这一点）
    assert(Hx == Hw  && Wx == Ww);  //输入Blob高和宽和全连接核Blob高和宽一样（务必保证这一点）

    int Ho =  1;    //输出Blob高（全连接操作后）
    int Wo =  1;  //输出Blob宽（全连接操作后）

    //-------step2.开始全连接运算
    out.reset(new Blob(N, F, Ho, Wo));

    for (int n = 0; n < N; ++n)   //输出cube数
    {
        for (int f = 0; f < F; ++f)  //输出通道数
        {
            (*out)[n](0, 0, f) = accu((*in[0])[n] % (*in[1])[f]) + as_scalar((*in[2])[f]);    //b = (F,1,1,1)
        }
    }

}

void FcLayer::backward(const shared_ptr<Blob>& din,
                       const vector<shared_ptr<Blob>>& cache,
                       vector<shared_ptr<Blob>>& grads,
                       const LayerParameter& param)
{

    cout << "FcLayer::backward()..." << endl;
    //dX,dW,db  -> X,W,b 他们的尺寸是一一对应的
    grads[0].reset(new Blob(cache[0]->size(),TZEROS));
    grads[1].reset(new Blob(cache[1]->size(), TZEROS));
    grads[2].reset(new Blob(cache[2]->size(), TZEROS));
    int N = grads[0]->get_N();
    int F = grads[1]->get_N();
    assert(F == cache[1]->get_N());

    for (int n = 0; n < N; ++n)
    {
        for (int f = 0; f < F; ++f)
        {
            //dX
            (*grads[0])[n] += (*din)[n](0, 0, f) * (*cache[1])[f];
            //dW 由于dw来源于不同样本产生的梯度所以最好除以N得到平均梯度
            (*grads[1])[f] += (*din)[n](0, 0, f) * (*cache[0])[n] / N;
            //db
            (*grads[2])[f] += (*din)[n](0, 0, f) / N;
        }
    }
    return;
}

void SoftmaxLossLayer::softmax_cross_entropy_with_logits(const vector<shared_ptr<Blob>>& in, double& loss, shared_ptr<Blob>& dout)
{
    cout << "SoftmaxLossLayer::softmax_cross_entropy_with_logits()..." << endl;
    if (dout)
        dout.reset();
    //-------step1.获取相关尺寸
    int N = in[0]->get_N();        //输入Blob中cube个数（该batch样本个数）
    int C = in[0]->get_C();         //输入Blob通道数
    int Hx = in[0]->get_H();      //输入Blob高
    int Wx = in[0]->get_W();    //输入Blob宽
    assert(Hx == 1 && Wx==1);

    dout.reset(new Blob(N, C, Hx, Wx));   //（N,C,1,1）
    double loss_ = 0;
    //先做softmax归一化，再计算交叉熵损失
    for (int i = 0; i < N; ++i)
    {
        //softmax归一化
        cube prob = arma::exp((*in[0])[i]) / arma::accu(arma::exp((*in[0])[i]));
        //在通道维度上把各个元素加起来
        loss_ += (-arma::accu((*in[1])[i] % arma::log(prob)) );  //累加各个样本的交叉熵损失值
        //梯度表达式推导：https://blog.csdn.net/qian99/article/details/78046329
        (*dout)[i]=prob - (*in[1])[i];  //计算各个样本产生的误差信号（反向梯度）
    }
    loss = loss_ / N;   //求平均损失


    return;
}

void SVMLossLayer::hinge_with_logits(const vector<shared_ptr<Blob>>& in, double& loss, shared_ptr<Blob>& dout)
{
    if (dout)
        dout.reset();
    //-------step1.获取相关尺寸
    int N = in[0]->get_N();        //输入Blob中cube个数（该batch样本个数）
    int C = in[0]->get_C();         //输入Blob通道数
    int Hx = in[0]->get_H();      //输入Blob高
    int Wx = in[0]->get_W();    //输入Blob宽
    assert(Hx == 1 && Wx == 1);
    dout.reset(new Blob(N, C, Hx, Wx));   //（N,C,1,1）
    double loss_ = 0;
    double delta = 0.2;
    for (int i = 0; i < N; ++i)
    {
        //(1).计算损失
        int idx_max = (*in[1])[i].index_max();//找出正确类别
        double positive_x = (*in[0])[i](0, 0, idx_max);//输出类别中的正确类别得分
        cube tmp = ((*in[0])[i] - positive_x + delta); //代入hinge loss公式 delta应该是超参，先硬编码
        tmp(0, 0, idx_max) = 0;  //剔除正确类里面的值
        tmp.transform([](double e) {return e > 0 ? e : 0; });  //做max()操作，得到各个分类的损失
        loss_ +=arma::accu(tmp);  //得到所有类别的损失和

        //(2).计算梯度
        tmp.transform([](double e) {return e ? 1 : 0; });//计算掩码，大于0为1，小于0为0
        tmp(0,0,idx_max)= -arma::accu(tmp);//求正确类别梯度，等于所有错误类别之和加负号
        (*dout)[i]=tmp;

    }
    loss = loss_ / N;   //求平均损失
    return;
}
