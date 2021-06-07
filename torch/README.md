参考：https://my.oschina.net/u/876354/blog/3070699

（1）什么是LSTM  
为了实现对不定长文字的识别，就需要有一种能力更强的模型，该模型具有一定的记忆能力，能够按时序依次处理任意长度的信息，这种模型就是“循环神经网络”（Recurrent Neural Networks，简称RNN）。
LSTM（Long Short Term Memory，长短期记忆网络）是一种特殊结构的RNN（循环神经网络），用于解决RNN的长期依赖问题，也即随着输入RNN网络的信息的时间间隔不断增大，普通RNN就会出现“梯度消失”或“梯度爆炸”的现象，这就是RNN的长期依赖问题，而引入LSTM即可以解决这个问题。LSTM单元由输入门（Input Gate）、遗忘门（Forget Gate）和输出门（Output Gate）组成，具体的技术原理的工作过程详见之前的文章（文章：白话循环神经网络（RNN）），LSTM的结构如下图所示。  

![](https://github.com/DemonXD/AIOHttp-enhancOCR/blob/master/torch/lstm_ctc/image/lstm.jpg)  

（2）什么是CTC  
CTC（Connectionist Temporal Classifier，联接时间分类器），主要用于解决输入特征与输出标签的对齐问题。例如下图，由于文字的不同间隔或变形等问题，导致同个文字有不同的表现形式，但实际上都是同一个文字。在识别时会将输入图像分块后再去识别，得出每块属于某个字符的概率（无法识别的标记为特殊字符”-”），如下图：  

![](https://github.com/DemonXD/AIOHttp-enhancOCR/blob/master/torch/lstm_ctc/image/ctc1.jpg)  

由于字符变形等原因，导致对输入图像分块识别时，相邻块可能会识别为同个结果，字符重复出现。因此，通过CTC来解决对齐问题，模型训练后，对结果中去掉间隔字符、去掉重复字符（如果同个字符连续出现，则表示只有1个字符，如果中间有间隔字符，则表示该字符出现多次），如下图所示：  
![](https://github.com/DemonXD/AIOHttp-enhancOCR/blob/master/torch/lstm_ctc/image/ctc2.jpg)