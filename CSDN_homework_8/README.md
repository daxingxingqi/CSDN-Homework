# CSDN homework_Week 8_Qi Zichen
在实现Dense net时，我参考了晚上的帖子（copy），请查看一下。

Growth：Growth rate就是每一层的输出通道数，每个dense block有不同的层数，每一个block都把所有层的通道叠加在一起。
稠密链接：每一次迭代，都把之前的输出都与当前的输出叠加，也就是每一次输出都包含之前的feature-maps
![输入图片说明](https://images.gitee.com/uploads/images/2018/0808/215000_3b7b0dd4_1974025.png "Screenshot from 2018-08-08 21-41-02.png")