//
// Created by xuwei on 2022/11/17.
//

#include "net.h"
#include "snapshot.pb.h"
#include <jsoncpp/json/json.h>
#include <fstream>
#include <cassert>

using namespace std;
void NetParameter::readNetParam(std::string file) {
    ifstream ifs;
    ifs.open(file);
    assert(ifs.is_open());
    Json::Reader reader;  /*解析器*/
    Json::Value value;    /*存储器,解析器解析出来的值就存在value对象里*/
    if (reader.parse(ifs, value))
    {
        if (!value["train"].isNull())
        {
            /*拿到train对象里面的所有元素*/
            auto &tparam = value["train"];
            this->lr = tparam["learning rate"].asDouble();
            this->lr_decay = tparam["lr decay"].asDouble();
            this->optimizer = tparam["optimizer"].asString();
            this->momentum = tparam["momentum parameter"].asDouble();
            this->rms_decay = tparam["rmsprop decay"].asDouble();
            this->reg = tparam["reg coefficient"].asDouble();
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
                this->layers.push_back(ii["name"].asString());//层名称
                this->ltypes.push_back(ii["type"].asString());//层类型

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
                    this->lparams[ii["name"].asString()].conv_weight_init = ii["conv weight init"].asString();
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
                    this->lparams[ii["name"].asString()].fc_weight_init = ii["fc weight init"].asString();
                }
                if (ii["type"].asString() == "Dropout")
                {
                    this->lparams[ii["name"].asString()].drop_rate = ii["drop rate"].asDouble();
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
        step_cache_[layers_[i]] = vector<shared_ptr<Blob>>(3, NULL); //为每一层梯度计算创建要用到的3个Blob x,w,b
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
        if (ltype == "Dropout")
        {
            myLayer.reset(new DropoutLayer);
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

    //4. 是否采用fine-tune方式训练模型（也就是是否用预训练模型里面的参数覆盖原来的随机参数值）？
    if (param.fine_tune)
    {
        fstream input(param.preTrainModel, ios::in | ios::binary);
        if (!input)
        {
            cout << param.preTrainModel << " was not found ！！！" << endl;
            return;
        }

        shared_ptr<MyCaffe::Snapshot>  snapshot_model(new MyCaffe::Snapshot);
        if (!snapshot_model->ParseFromIstream(&input))//解析预训练模型（protobuf提供的方法）
        {
            cout<< "Failed to parse the " << param.preTrainModel << " ！！！" << endl;
            return;
        }
        cout << "--- Load the" << param.preTrainModel << " sucessfully ！！！---" << endl;

        loadModelParam(snapshot_model);//加载模型参数
    }

}

void Net::trainNet(NetParameter& param)
{
    int N = X_train_->get_N();
    cout << "N = " << N << endl;
    int iter_per_epoch = N / param.batch_size;//59000/200 = 295
    //总的批次数（迭代次数）= 单个epoch所含批次数 * epoch个数
    int num_batchs = iter_per_epoch * param.num_epochs;// 295 * 2 = 590
    cout << "num_batchs(iterations) = " << num_batchs << endl;

    //for (int iter = 0; iter < num_batchs; ++iter)
    for (int iter = 0; iter < 50; ++iter)
    {
        //----------step1. 从整个训练集中获取一个mini-batch
        shared_ptr<Blob> X_batch;
        shared_ptr<Blob> Y_batch;
        /*可能最后剩下的部分不足一个batch size,截取的时候high-indx会超出数据集大小,%N后能解决这个问题，超出的部分会转到从0开始算，subBlob能处理这种情况*/
        X_batch.reset(new Blob(X_train_->subBlob((iter* param.batch_size) % N,
                                                 ((iter + 1)* param.batch_size) % N)));
        Y_batch.reset(new Blob(Y_train_->subBlob((iter* param.batch_size) % N,
                                                 ((iter + 1)* param.batch_size) % N)));

        //----------step2. 用该mini-batch训练网络模型
        train_with_batch(X_batch, Y_batch, param);

        //----------step3. 评估模型当前准确率（训练集和验证集）
        //----------step3. 评估模型当前准确率（训练集和验证集）
        if (iter%param.eval_interval == 0)
        {
            evaluate_with_batch(param);
            printf("iter_%d    lr: %0.6f    train_loss: %f    val_loss: %f    train_acc: %0.2f%%    val_acc: %0.2f%%\n",
                   iter, param.lr, train_loss_, val_loss_, train_accu_ * 100, val_accu_ * 100);
        }

        //----------step4.保存模型快照 https://blog.csdn.net/u011334621/article/details/51735418
        if (iter > 0 && param.snap_shot && iter % param.snapshot_interval == 0)
        {
            //(1).定义输出文件outputFile
            char outputFile[40];
            sprintf(outputFile, "./iter%d.model", iter);
            fstream output(outputFile, ios::out | ios::trunc | ios::binary);//若此文件存在则先删除再创建，不存在就直接创建！

            //(2).把Blob中的参数保存到（proto定义的）snapshotModel这个数据结构中！
            shared_ptr<MyCaffe::Snapshot> snapshot_model(new MyCaffe::Snapshot);
            saveModelParam(snapshot_model);

            //(3).调用SerializeToOstream()函数将snapshotModel里面的数据写成一个二进制文件outputFile
            if (!snapshot_model->SerializeToOstream(&output))//将数据结构中的w和b中以protobuf协议写入一个文件
            {
                cout << "Failed to Serialize snapshot_model To Ostream." << endl;//模型权重（偏置）参数保存失败
                return;
            }
        }
    }

}

void Net::train_with_batch(shared_ptr<Blob>&  X, shared_ptr<Blob>&  Y, NetParameter& param, string mode)
{
    //------- step1. 将mini-batch填充到初始层的X当中
    data_[layers_[0]][0]=X;
    //把y放到最后一层，反正最后一层是损失层1空着也是空，正好用来传递标签值y
    data_[layers_.back()][1] = Y;

    //------- step2. 逐层前向计算
    int n = layers_.size();//层数
    for (int i = 0; i < n - 1; ++i)//最后一层单独拧出来
    {
        string lname = layers_[i];
        shared_ptr<Blob> out;
        myLayers_[lname]->forward(data_[lname], out, param.lparams[lname], mode);
        data_[layers_[i+1]][0] = out;
    }

    //------- step3. softmax前向计算和计算代价值
    if (mode == "TRAIN")
    {

        if (ltypes_.back() == "Softmax")//layers_.back()最后一个元素, diff_[layers_.back()][0] 是dx
            SoftmaxLossLayer::softmax_cross_entropy_with_logits(data_[layers_.back()], train_loss_, diff_[layers_.back()][0]);
        if (ltypes_.back() == "SVM")
            SVMLossLayer::hinge_with_logits(data_[layers_.back()], train_loss_, diff_[layers_.back()][0]);
    }
    else
    {
        if (ltypes_.back() == "Softmax")
            SoftmaxLossLayer::softmax_cross_entropy_with_logits(data_[layers_.back()], val_loss_, diff_[layers_.back()][0]);
        if (ltypes_.back() == "SVM")
            SVMLossLayer::hinge_with_logits(data_[layers_.back()], val_loss_, diff_[layers_.back()][0]);
    }

    //------- step4. 逐层反向传播
    if (mode == "TRAIN")
    {
        for (int i = n - 2; i >= 0; --i)
        {
            string lname = layers_[i];
            //输入后一层的梯度所以是i+1
            myLayers_[lname]->backward(diff_[layers_[i + 1]][0], data_[lname], diff_[lname], param.lparams[lname]);
        }
    }


    //----------step5.对各层梯度施加L2正则化的影响
    if (param.reg!=0)//正则化系数不为0，意味着需要施加L2正则化！
        regular_with_batch(param,mode);

    //----------step6. 参数更新（利用梯度下降）
    if (mode == "TRAIN")
        optimizer_with_batch(param);

}

void Net::regular_with_batch(NetParameter& param, string mode)
{
    int N = data_[layers_[0]][0]->get_N();//获取该批次样本数
    double reg_loss = 0;
    for (auto lname : layers_)
    {
        if (diff_[lname][1])//只对带权值梯度的层进行处理！
        {
            if (mode == "TRAIN") {
                //梯度加上正则项
                Blob temp = param.reg * (*data_[lname][1]);
                temp = temp / N;
                (*diff_[lname][1])  = (*diff_[lname][1]) + temp;
            }
            //损失加上正则项
            Blob temp = square((*data_[lname][1]));
            reg_loss += accu(temp);
        }
    }
    reg_loss = reg_loss* param.reg / (2 * N);
    if (mode == "TRAIN")
        train_loss_ = train_loss_ + reg_loss;
    else
        val_loss_ = val_loss_ + reg_loss;
}

void Net::optimizer_with_batch(NetParameter& param)
{

    for (auto lname : layers_)//for lname in layers_
    {
        //(1).跳过没有w和b的层
        if (!data_[lname][1] || !data_[lname][2])
        {
            continue;//跳过本轮循环，重新执行循环（注意不是像break那样直接跳出循环）
        }

        //(2).利用梯度下降更新有w和b的层
        for (int i = 1; i <= 2; ++i)
        {
            assert(param.optimizer == "sgd" || param.optimizer == "momentum" || param.optimizer == "rmsprop");//sgd/momentum/rmsprop

            shared_ptr<Blob> dparam(new Blob(data_[lname][i]->size(),TZEROS));
            if (param.optimizer == "rmsprop")
            {
                //V_grad = decay * V_grad + (1-decay)*grad*grad
                //param = param -  learning_rate * grad/sqrt(V_grad + esp)
                double decay_rate = param.rms_decay;
                if (!step_cache_[lname][i])
                    step_cache_[lname][i].reset(new Blob(data_[lname][i]->size(), TZEROS));
                Blob temp1 = decay_rate * (*step_cache_[lname][i]);
                Blob temp2 = (*diff_[lname][i]) * (*diff_[lname][i]);
                temp2 = (1 - decay_rate) * temp2;
                (*step_cache_[lname][i]) = temp1 + temp2;
                Blob temp3 = (*step_cache_[lname][i]) + 1e-8;//加一个极小值
                temp3 = sqrt(temp3);
                Blob temp4 = -param.lr * (*diff_[lname][i]);
                (*dparam) = temp4 / temp3;

            }
            else if (param.optimizer == "momentum")
            {
                //V_grad = momentum * V_grad + grad
                //param = param - V_grad * learning_rate
                if (!step_cache_[lname][i])
                    step_cache_[lname][i].reset(new Blob(data_[lname][i]->size(), TZEROS));
                Blob temp = param.momentum * (*step_cache_[lname][i]);
                (*step_cache_[lname][i]) = temp + (*diff_[lname][i]);
                (*dparam) = -param.lr * (*step_cache_[lname][i]);
            }
            else
            {
                //w:=w-param.lr*dw ;    b:=b-param.lr*db     ---->  "sgd"
                (*dparam) = -param.lr * (*diff_[lname][i]);
            }
            (*data_[lname][i]) = (*data_[lname][i]) + (*dparam);//所有优化算法公用部分（梯度下降）
        }
    }
    //学习率更新
    if (param.lr_update)
        param.lr *= param.lr_decay;

}

void Net::evaluate_with_batch(NetParameter& param)
{
    //(1).评估训练集准确率
    shared_ptr<Blob> X_train_subset;
    shared_ptr<Blob> Y_train_subset;
    int N = X_train_->get_N();
    if (N > 1000)
    {
        X_train_subset.reset(new Blob(X_train_->subBlob(0, 1000)));
        Y_train_subset.reset(new Blob(Y_train_->subBlob(0, 1000)));
    }
    else
    {
        X_train_subset = X_train_;
        Y_train_subset = Y_train_;
    }
    train_with_batch(X_train_subset, Y_train_subset, param,"TEST");//“TEST”，测试模式，只进行前向传播
    train_accu_ =calc_accuracy(*data_[layers_.back()][1], *data_[layers_.back()][0]);

    //(2).评估验证集准确率
    train_with_batch(X_val_, Y_val_, param, "TEST");//“TEST”，测试模式，只进行前向传播
    val_accu_ = calc_accuracy(*data_[layers_.back()][1], *data_[layers_.back()][0]);
}

double Net::calc_accuracy(Blob& Y, Blob& Predict)
{
    //(1). 确保两个输入Blob尺寸一样
    vector<int> size_Y = Y.size();
    vector<int> size_P = Predict.size();
    for (int i = 0; i < 4; ++i)
    {
        assert(size_Y[i] == size_P[i]);//断言：两个输入Blob的尺寸（N,C,H,W）一样！
    }
    //(2). 遍历所有cube（样本），找出标签值Y和预测值Predict最大值所在位置进行比较，若一致，则正确个数+1
    int N = Y.get_N();//总样本数
    int right_cnt = 0;//正确个数
    for (int n = 0; n < N; ++n)
    {
        //参考网址：http://arma.sourceforge.net/docs.html#index_min_and_index_max_member
        //index_max返回的是一个数值，是从左到右，从上到下，从前到后的数
        if (Y[n].index_max() == Predict[n].index_max())
            right_cnt++;
    }
    return (double)right_cnt / (double)N;//计算准确率，返回（准确率=正确个数/总样本数）
}

void Net::saveModelParam(shared_ptr<MyCaffe::Snapshot>& snapshot_model)
{
    cout << endl << "/////////////////////////////// 打印Blob（w和b） /////////////////////////////////" << endl << endl;
    for (auto lname : layers_)//for lname in layers_
    {
        //(1).跳过没有w和b的层
        if (!data_[lname][1] || !data_[lname][2])
        {
            continue;//跳过本轮循环，重新执行循环（注意不是像break那样直接跳出循环）
        }
        cout << "-----" << lname << "-----" << endl;
        for (int i = 1; i <= 2; ++i)
        {
            cout << "-----" << (i==1 ? "WEIGHT" : "BIAS") << "-----" << endl;
            data_[lname][i]->print();
        }
    }



    for (auto lname : layers_)//for lname in layers_
    {
        //(1).跳过没有w和b的层
        if (!data_[lname][1] || !data_[lname][2])
        {
            continue;//跳过本轮循环，重新执行循环（注意不是像break那样直接跳出循环）
        }

        //(2).取出相关Blob中的所有参数，填入snapshotModel中！
        for (int i = 1; i <= 2; ++i)
        {
            MyCaffe::Snapshot::ParamBlok*  param_blok = snapshot_model->add_param_blok();//（动态）添加一个paramBlock
            int N = data_[lname][i]->get_N();//权重（偏置）核个数
            int C = data_[lname][i]->get_C();//权重（偏置）核通道数
            int H = data_[lname][i]->get_H();//权重（偏置）核高
            int W = data_[lname][i]->get_W();//权重（偏置）核宽
            param_blok->set_kernel_n(N);
            param_blok->set_kernel_c(C);
            param_blok->set_kernel_h(H);
            param_blok->set_kernel_w(W);
            param_blok->set_layer_name(lname);
            if (i == 1)
            {
                param_blok->set_param_type("WEIGHT");//写入参数类型
                cout << lname << " : WEIGHT  " << "（" << N << "," << C << "," << H << "," << W << "）" << endl;
            }
            else
            {
                param_blok->set_param_type("BIAS");
                cout << lname << " : BIAS  " << "（" << N << "," << C << "," << H << "," << W << "）" << endl;
            }
            for (int n = 0; n<N; ++n)
            {
                for (int c = 0; c < C; ++c)
                {
                    for (int h = 0; h<H; ++h)
                    {
                        for (int w = 0; w<W; ++w)
                        {
                            MyCaffe::Snapshot::ParamBlok::ParamValue*  param_value = param_blok->add_param_value();//（动态）添加一个paramValue
                            param_value->set_value((*data_[lname][i])[n](h, w, c));
                        }
                    }
                }
            }

        }



    }
}

void Net::loadModelParam(const shared_ptr<MyCaffe::Snapshot>& snapshot_model)
{
    for (int i = 0; i < snapshot_model->param_blok_size(); ++i)//逐个取出模型快照中的的paramBlok，填入我们定义的Blob数据结构中
    {
        //1. 从snapshot_model逐一取出paramBlok
        const MyCaffe::Snapshot::ParamBlok& param_blok = snapshot_model->param_blok(i);//取出对应paramBlok

        //2. 取出paramBlok中的标记型变量
        string lname = param_blok.layer_name();//权重（偏置）核所属层名
        string paramtype = param_blok.param_type();//权重（偏置）核参数类型（WEIGHT或BIAS）
        int N = param_blok.kernel_n();//权重（偏置）核个数
        int C = param_blok.kernel_c();//权重（偏置）核通道数
        int H = param_blok.kernel_h();//权重（偏置）核高
        int W = param_blok.kernel_w();//权重（偏置）核宽
        cout << lname << "：" << paramtype << " ：（" << N << ", " << C << ", " << H << ", " << W << ")" << endl;

        //3.遍历当前paramBlok中的每一个参数，取出来，填入对应的Blob中！
        int val_idx = 0;
        shared_ptr<Blob> simple_blob(new Blob(N, C, H, W));//中间Blob
        for (int n = 0; n<N; ++n)
        {
            for (int c = 0; c < C; ++c)
            {
                for (int h = 0; h<H; ++h)
                {
                    for (int w = 0; w<W; ++w)
                    {
                        const MyCaffe::Snapshot::ParamBlok::ParamValue& param_value = param_blok.param_value(val_idx);
                        (*simple_blob)[n](h,w,c)=param_value.value();//取出某个参数，填入Blob对应位置！
                        val_idx++;//param_blok块索引线性增加！
                    }
                }
            }
        }

        //4. 将simple_blob赋值到data_中
        if (paramtype == "WEIGHT")
            data_[lname][1] = simple_blob;
        else
            data_[lname][2] = simple_blob;

    }


    cout << endl << "/////////////////////////////// 打印Blob（w和b） /////////////////////////////////" << endl << endl;
    for (auto lname : layers_)//for lname in layers_
    {
        //(1).跳过没有w和b的层
        if (!data_[lname][1] || !data_[lname][2])
        {
            continue;//跳过本轮循环，重新执行循环（注意不是像break那样直接跳出循环）
        }
        cout << "-----" << lname << "-----" << endl;
        for (int i = 1; i <= 2; ++i)
        {
            cout << "-----" << (i == 1 ? "WEIGHT" : "BIAS") << "-----" << endl;
            data_[lname][i]->print();
        }
    }

}