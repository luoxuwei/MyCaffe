//
// Created by xuwei on 2022/11/17.
//

#ifndef MYCAFFE_BLOB_H
#define MYCAFFE_BLOB_H

#include <vector>
#include <armadillo>
using std::vector;
using arma::cube;
using std::string;

enum FillType
{

    TONES = 1,  //cube所有元素都填充为1
    TZEROS = 2, //cube所有元素都填充为0
    TRANDU = 3,  //将元素设置为[0,1]区间内均匀分布的随机值
    TRANDN = 4,  //使用μ= 0和σ= 1的标准高斯分布设置元素
    TDEFAULT = 5

};

class Blob
{
public:
    Blob() :
        N_(0),/*cube的个数，也就是特征图的个数*/
        C_(0),/*cube或图片的通道数*/
        H_(0),/*图片的高*/
        W_(0)/*图片的宽*/
    {}
    Blob(const int n, const int c, const int h, const int w, int type = TDEFAULT);  /*type:填充立方体的方式*/
    Blob(const vector<int> shape_, int type = TDEFAULT);
    vector<cube>& get_data();
    cube& operator[] (int i);
    Blob& operator*= (const double i);
    Blob& operator= (double val);
    friend Blob operator*(Blob& A, Blob& B);  //声明为友元函数
    Blob subBlob(int low_idx, int high_idx);
    Blob pad(int pad,double val=0);
    void maxIn(double val=0.0);
    vector<int> size() const;
    inline int get_N() const
    {
        return N_;
    }
    inline int get_C() const
    {
        return C_;
    }
    inline int get_H() const
    {
        return H_;
    }
    inline int get_W() const
    {
        return W_;
    }
    void print(string str = "");
private:
    void _init(const int n, const int c, const int h, const int w, int type);
private:
    int N_;
    int C_;
    int H_;
    int W_;
    vector<cube> blob_data;
};

#endif //MYCAFFE_BLOB_H
