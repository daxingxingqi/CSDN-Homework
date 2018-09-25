# CSDN homework_Week 10_Qi Zichen

#### 数据集准备完成
- 数据集train和val两个tfrecord文件截图已经展示在下面
![输入图片说明](https://images.gitee.com/uploads/images/2018/0825/121444_cfb0d350_1974025.png "Screenshot from 2018-08-25 12-14-11.png")

#### 模型训练完成
- log file 截图
![输入图片说明](https://images.gitee.com/uploads/images/2018/0825/122518_a85ed78f_1974025.png "1.png")

#### 训练结果完成

1. ![输入图片说明](https://images.gitee.com/uploads/images/2018/0825/122654_52ad5ca4_1974025.jpeg "val_800_img.jpg")
2. ![输入图片说明](https://images.gitee.com/uploads/images/2018/0825/122704_02128e28_1974025.jpeg "val_800_overlay.jpg")
3. ![输入图片说明](https://images.gitee.com/uploads/images/2018/0825/122715_a8d9d95c_1974025.jpeg "val_800_prediction.jpg")
4.![输入图片说明](https://images.gitee.com/uploads/images/2018/0825/122721_fcca9533_1974025.jpeg "val_800_prediction_crfed.jpg")

#### 模型代码补全 

- 代码已经上传在码云中


#### 心得体会
- 具体文字过程*FCN 

第一次卷积conv1、pool1--原图缩小为1/2

第二次卷积conv2、pool2--原图缩小为1/4

第三次卷积conv3、pool3--原图缩小为1/8 --此时保留pool3的featureMap

第四次卷积conv4、pool4--原图缩小为1/16

第五次卷积conv5、pool5--原图缩小为1/32

第六次卷积conv6

第七次卷积conv7

第八步对conv3，conv4，conv5 进行反卷积补充细节然后相加

          


- 代码讲解
![输入图片说明](https://images.gitee.com/uploads/images/2018/0825/125104_cae16838_1974025.png "20161022113219788.png")
1. 首先取出vgg16预训练模型最后的logits（原图1/32）

```
# Define the model that we want to use -- specify to use only two classes at the last layer
with slim.arg_scope(vgg.vgg_arg_scope()):
    logits, end_points = vgg.vgg_16(image_tensor,
                                    num_classes=number_of_classes,
                                    is_training=is_training_placeholder,
                                    spatial_squeeze=False,fc_conv_padding='SAME')

downsampled_logits_shape = tf.shape(logits)

img_shape = tf.shape(image_tensor)

# Calculate the ouput size of the upsampled tensor
# The shape should be batch_size X width X height X num_classes
upsampled_logits_shape = tf.stack([
                                  downsampled_logits_shape[0],
                                  img_shape[1],
                                  img_shape[2],
                                  downsampled_logits_shape[3]
                                  ])
```
2.取出pool4和pool3的logits，然后对其进行全卷积（原图1/16和1/8）
```
#输出pool4和pool3的logits
pool4_feature = end_points['vgg_16/pool4']#get the feature map of pool4
with tf.variable_scope('vgg_16/fc8'):
    aux_logits_16s = slim.conv2d(pool4_feature, number_of_classes, [1, 1],
                                 activation_fn=None,
                                 weights_initializer=tf.zeros_initializer,
                                 scope='conv_pool4')#number_of_classes - num_outputs: Integer, the number of output filters.
    
pool3_feature = end_points['vgg_16/pool3']#get the feature map of pool3
with tf.variable_scope('vgg_16/fc8'):
    aux_logits_8s = slim.conv2d(pool3_feature, number_of_classes, [1, 1],
                                 activation_fn=None,
                                 weights_initializer=tf.zeros_initializer,
                                 scope='conv_pool3')
```

3.对最后的logits和pool5进行上采样（反卷积）（原图1/8）
```
###########################################    
upsample_filter_np_x4 = bilinear_upsample_weights(4, #upsample_factor,
                                                  number_of_classes)

upsample_filter_tensor_x4 = tf.Variable(upsample_filter_np_x4, name='vgg_16/fc8/t_conv_x4')
#对最后的logits和pool4进行转置卷积生成4x4的feature map
upsampled_logits_pool5 = tf.nn.conv2d_transpose(logits, upsample_filter_tensor_x4,
                                          output_shape=tf.shape(aux_logits_8s),
                                          strides=[1, 4, 4, 1],padding='SAME')
upsampled_logits_pool4=tf.nn.conv2d_transpose(aux_logits_16s,
                                              upsample_filter_tensor_x4,
                                              output_shape=tf.shape(aux_logits_8s),
                                              strides=[1, 2, 2, 1],
                                              padding='SAME')
```
4.把pool5和pool4上采样的结果与最后的上采样的logits相加（原图1/8)
```
upsampled_logits = upsampled_logits_pool5 + upsampled_logits_pool4 + aux_logits_8s
```
5.把相加后的logits在进行上采样（恢复到原图）
```
###########################################
#最后输出8*8的feature map
upsample_filter_np_x8 = bilinear_upsample_weights(upsample_factor,
                                                   number_of_classes)

upsample_filter_tensor_x8 = tf.Variable(upsample_filter_np_x8, name='vgg_16/fc8/t_conv_x8')



upsampled_logits = tf.nn.conv2d_transpose(upsampled_logits, upsample_filter_tensor_x8,
                                          output_shape=upsampled_logits_shape,
                                          strides=[1, upsample_factor, upsample_factor, 1],
                                          padding='SAME')



```

- 心得
1.代码实现过程比较简单，但是stride的选择在最后实现出现问题，最后把pool5 反卷积到原图1/8时，步长应该是4而不是2。
2.对于代码，并不是全部理解，希望在直播时多讲一讲细节。把代码实现流程过一遍就最好了。