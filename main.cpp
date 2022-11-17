//
// Created by xuwei on 2022/11/17.
//
#include <iostream>
#include <string>
#include "net.h"
#include "blob.h"

using namespace std;

int main(int argc, char** argv) {
    string configFile = "./my_model.json";
    NetParameter net_param;
    net_param.readNetParam(configFile);

    cout << "learning rate =  " << net_param.lr << endl;
    cout << "batch size =  " << net_param.batch_size << endl;

    vector<string> layers_ = net_param.layers;
    vector<string> ltypes_ = net_param.ltypes;
    for (int i = 0; i < layers_.size(); ++i)
    {
        cout << "layer = " << layers_[i] << " ; " << "ltype = " << ltypes_[i] << endl;
    }

    //测试Blob类。
    Blob test_blob(1, 1, 5, 5, TRANDN);
    test_blob.print("Blob 里面的数据是：\n");
}