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
ไอเดียคร่าวๆของ Depthwise Separable TCN คือ จะทำ Convolution แยก channel ก่อน (Depthwise convolution) แล้วค่อยทำ Pointwise (convolution แค่จุดเดียว แต่รวบทุก channel) 
[อ่านต่อ](Separable%20Convolution.md#Separable%20Convolution)

งานนี้เหมือนจะใช้ Depthwise separable convolution มาแทนที่ Convolution ธรรมดาใน MS-TCN เพื่อลด computation cost และ convolution parameters

### Knowledge Distillation
ขอเรียกสั้นๆว่า **KD** เป็นหนึ่งในเทคนิคของการ Compress model ที่มีขนาดใหญ่ให้เล็กลง เพื่อลดจำนวน parameter และ computation cost ลง โดยการนำบาง layer ออกหรือลดจำนวน neuron ในบาง layer ลง แต่ model ยังคงมี performance ที่ใกล้เคียงกับของเดิม
[อ่านต่อ](lib/ml/Knowledge%20Distillation)

นอกจากนี้ งานนี้ได้ย้อนไปดู proof ที่ว่า KD ก็ยังคงมีประโยชน์เหมือนกัน แม้จะมี layer/neuron จำนวนเท่าเดิมก็ตาม เนื่องจาก knowledge ข้าม generation (น่าจะหมายถึง epoch) ก็สามารถนำมาใช้เป็น teacher ให้ generation หลังๆได้

งานนี้ใช้ Born-Again Distillation

# Framework

# Mathematical Stuff
## Objective function

# Experiment Details
## Datasets
## Evaluation Metrics