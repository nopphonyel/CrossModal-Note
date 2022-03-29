# General Details
[==Abstract==](https://arxiv.org/abs/2007.06504v3), [==PDF File==](https://arxiv.org/pdf/2007.06504v3.pdf), [==Code Repository==](https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks)

## งานนี้โดยหลักแล้ว
พยายามที่จะเพิ่ม performance ของโมเดลโดยการลด computational complexity ลง และลดขนาดของ state-of-the-art model  ให้กลายเป็น lightweight model.

## Related work
### Base Architecture
มีการใช้ base architecture จากงาาน ___ มาปรับปรุง ซึ่งงานดังกล่าวได้ state-of-the-art ของ dataset LRW กับ LRW1000

### Efficient Backbone
เพื่อเพิ่มประสิทธิภาพของ model เลยพยายามหา Backbone ตัวใหม่สำหรับแทนที่ ResNet-18 (น่าจะเป็น backbone สำหรับฝั่ง video modal) ซึ่งมีตัวนึงชื่อ ShuffleNet v2 ($\beta \times$ โดยที่ $\beta$ คือ Width Multiplier)
#### เกี่ยวกับ ShuffleNet v2
- ใช้ Separable Convolutions แทน CNN ธรรมดา ซึ่งช่วยลดจำนวน parameters  และ computation complexity ลง 
- มีการทำ Channel shuffling ช่วยให้มีการแลกเปลี่ยนข้อมูลกันระหว่าง Channel
- ShuffleNet v2 (1.0$\times$) มี parameters น้อยกว่า 5 เท่า และ FLOPs น้อยกว่า 12 เท่าเมื่อเทียบกับ ResNet-18

### Depthwise Separable TCN
งานนี้เหมือนจะใช้ Depthwise separable convolution มาแทนที่ Convolution ธรรมดาใน MS-TCN ไอเดียคร่าวๆของ Depthwise Separable TCN คือ จะทำ Convolution แยก channel ก่อน (Depthwise convolution) แล้วค่อยทำ Pointwise (convolution แค่จุดเดียว แต่รวบทุก channel) 
[อ่านต่อ](Separable%20Convolution.md#Separable%20Convolution)

### Knowledge Distillation
เป็นหนึ่งในเทคนิคของการ Transfer learning 

# Framework

# Mathematical Stuff
## Objective function

# Experiment Details
## Datasets
## Evaluation Metrics