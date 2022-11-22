//
// Created by xuwei on 2022/11/17.
//
#include <iostream>
#include <string>
#include <memory>
#include "net.h"
#include "blob.h"

using namespace std;

/*转换字节序，加载mnist数据集时，需要把大端数据转换为我们常用的小端数据*/
int ReverseInt(int i)
{
    unsigned char ch1, ch2, ch3, ch4;//int的4个字节
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

//http://yann.lecun.com/exdb/mnist/
void ReadMnistData(string path, shared_ptr<Blob> &images)
{
    ifstream file(path, ios::binary);
    if (file.is_open())
    {
        //mnist原始数据文件中32位的整型值是大端存储，C/C++变量是小端存储，所以读取数据的时候，需要对其进行大小端转换!!!!
        //1.从文件中获知魔术数字图片数量和图片宽高信息
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);//高低字节调换
        cout << "magic_number=" << magic_number << endl;
        file.read((char*)&number_of_images, sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);
        cout << "number_of_images=" << number_of_images << endl;
        file.read((char*)&n_rows, sizeof(n_rows));
        n_rows = ReverseInt(n_rows);
        cout << "n_rows=" << n_rows << endl;
        file.read((char*)&n_cols, sizeof(n_cols));
        n_cols = ReverseInt(n_cols);
        cout << "n_cols=" << n_cols << endl;

        //2.将图片转为Blob存储
        for (int i = 0; i<number_of_images; ++i)//遍历所有图片
        {
            for (int h = 0; h<n_rows; ++h)//遍历高
            {
                for (int w = 0; w<n_cols; ++w)//遍历宽
                {
                    unsigned char temp = 0;
                    file.read((char*)&temp, sizeof(temp));//读入一个像素值！
                    (*images)[i](h, w, 0) = (double)temp / 255;//除以255，做归一化
                }
            }
        }
    }
    else
    {
        cout << "no data file found :-(" << endl;
    }

}
void ReadMnistLabel(string path, shared_ptr<Blob> &labels)
{

    ifstream file(path, ios::binary);
    if (file.is_open())
    {
        //1.从文件中获知魔术数字，图片数量
        int magic_number = 0;
        int number_of_images = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        cout << "magic_number=" << magic_number << endl;
        file.read((char*)&number_of_images, sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);
        cout << "number_of_Labels=" << number_of_images << endl;
        //2.将所有标签转为Blob存储（手写数字识别：0~9）
        for (int i = 0; i<number_of_images; ++i)
        {
            unsigned char temp = 0;
            file.read((char*)&temp, sizeof(temp));
            //one-hot
            (*labels)[i](0, 0, (int)temp) = 1;
        }
    }
    else
    {
        cout << "no label file found :-(" << endl;
    }
}

void trainModel(string configFile, shared_ptr<Blob> X, shared_ptr<Blob> Y)
{
    //0. 读入并解析网络结构定义
    NetParameter net_param;
    net_param.readNetParam(configFile);

    //1. 将60000张图片以59:1的比例划分为训练集（59000张）和验证集（1000张）
    shared_ptr<Blob> X_train(new Blob(X->subBlob(0,59000)));//左闭右开区间，即[ 0, 59000 )
    shared_ptr<Blob> Y_train(new Blob(Y->subBlob(0, 59000)));
    shared_ptr<Blob> X_val(new Blob(X->subBlob(59000, 60000)));
    shared_ptr<Blob> Y_val(new Blob(Y->subBlob(59000, 60000)));

    /*放在一个vector里只是为了传参方便*/
    vector<shared_ptr<Blob>> XX{ X_train, X_val };
    vector<shared_ptr<Blob>> YY{ Y_train, Y_val };

    //2. 初始化网络结构
    Net model;
    model.initNet(net_param, XX, YY);

    //3. 开始训练
    cout << "------------ step3. Train start... ---------------" << endl;
    model.trainNet(net_param);
    cout << "------------ Train end... ---------------" << endl;

}

int main(int argc, char** argv) {
    //创建两个Blob对象，一个用来存储图片特征值（数据），另一个用来存储标签值
    shared_ptr<Blob> images(new Blob(60000, 1, 28, 28, TZEROS));
    shared_ptr<Blob> labels(new Blob(60000, 10, 1, 1, TZEROS));//保存one-hot编码的标签值
    ReadMnistData("mnist_data/train/train-images.idx3-ubyte", images);//读取data
    ReadMnistLabel("mnist_data/train/train-labels.idx1-ubyte", labels);//读取label
    trainModel("./my_model.json", images, labels);
}