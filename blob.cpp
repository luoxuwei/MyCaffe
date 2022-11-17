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

vector<cube>& Blob::get_data()
{
    return blob_data;
}