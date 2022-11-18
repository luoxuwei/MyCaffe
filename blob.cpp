//
// Created by xuwei on 2022/11/17.
//

#include "blob.h"
#include "cassert"
using namespace std;
using namespace arma;


Blob::Blob(const int n, const int c, const int h, const int w, int type) : N_(n), C_(c), H_(h), W_(w)
{
    //当前系统时间作为伪随机数的生成种子
    arma_rng::set_seed_random();  //系统随机生成种子(如果没有这一句，就会每次启动程序(进程)时都默认从种子1开始来生成随机数！就是每次都是一样的值。
    _init(N_, C_, H_, W_, type);

}

void Blob::_init(const int n, const int c, const int h, const int w, int type)
{

    if (type == TONES)
    {
        blob_data = vector<cube>(n, cube(h, w, c, fill::ones));
        return;
    }
    if (type == TZEROS)
    {
        blob_data = vector<cube>(n, cube(h, w, c, fill::zeros));
        return;
    }
    if (type == TDEFAULT)
    {
        blob_data = vector<cube>(n, cube(h, w, c));
        return;
    }
    if (type == TRANDU)
    {
        for (int i = 0; i < n; ++i)   //生成n个填充了随机值（均匀分布）的cube
            blob_data.push_back(arma::randu<cube>(h, w, c));
        return;
    }
    if (type == TRANDN)
    {
        for (int i = 0; i < n; ++i)   //生成n个填充了随机值(标准高斯分布）的cube
            blob_data.push_back(arma::randn<cube>(h, w, c));
        return;
    }

}

void Blob::print(string str)
{
    assert(!blob_data.empty());  //断言：blob_data不为空！否则中止程序，因为下面用到了，最好验证一下。
    cout << str << endl;
    for (int i = 0; i < N_; ++i)  //N_为blob_data中cube个数
    {
        printf("N = %d\n", i);
        this->blob_data[i].print();//逐一打印cube，调用cube中重载好的print()
    }
}

cube& Blob::operator[] (int i)
{
    return blob_data[i];
}

Blob& Blob::operator*= (const double k)
{
    for (int i = 0; i < N_; ++i)
    {
        blob_data[i] = blob_data[i] * k;   //调用cube中实现的*操作符
    }
    return *this;
}

Blob& Blob::operator= (double val)
{
    for (int i = 0; i < N_; ++i)
    {
        blob_data[i].fill(val);   //调用cube中实现的*操作符
    }
    return *this;
}

vector<cube>& Blob::get_data()
{
    return blob_data;
}

Blob Blob::subBlob(int low_idx, int high_idx)
{
    //举例： [0,1,2,3,4,5]  -> [1,3)  -> [1,2]
    if (high_idx > low_idx)
    {
        Blob tmp(high_idx - low_idx, C_, H_, W_);  // high_idx > low_idx
        for (int i = low_idx; i < high_idx; ++i)
        {
            tmp[i - low_idx] = (*this)[i];
        }
        return tmp;
    }
    else
    {
        // low_idx >high_idx
        //举例： [0,1,2,3,4,5]  -> [3,2)-> (6 - 3) + (2 -0) -> [3,4,5,0]
        Blob tmp(N_ - low_idx + high_idx, C_, H_, W_);
        for (int i = low_idx; i < N_; ++i)   //分开两段截取：先截取第一段
        {
            tmp[i - low_idx] = (*this)[i];
        }
        for (int i = 0; i < high_idx; ++i)   //分开两段截取：再截取循环到从0开始的这段
        {
            tmp[i + N_ - low_idx] = (*this)[i];
        }
        return tmp;
    }
}

Blob Blob::pad(int pad, double val)
{
    assert(!blob_data.empty());   //Blob自身不为空
    Blob padX(N_, C_, H_ + 2 * pad, W_ + 2 * pad);
    padX = val;
    for (int n = 0; n < N_; ++n)
    {
        for (int c = 0; c < C_; ++c)
        {
            for (int h = 0; h < H_; ++h)
            {
                for (int w = 0; w < W_; ++w)
                {
                    padX[n](h + pad, w + pad, c) = blob_data[n](h, w, c);
                }
            }
        }
    }
    return padX;

}
