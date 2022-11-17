//
// Created by xuwei on 2022/11/17.
//

#ifndef MYCAFFE_LAYER_H
#define MYCAFFE_LAYER_H

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

#endif //MYCAFFE_LAYER_H
