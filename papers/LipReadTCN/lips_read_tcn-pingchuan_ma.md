# General Details
[==Abstract==](https://arxiv.org/abs/2007.06504v3), [==PDF File==](https://arxiv.org/pdf/2007.06504v3.pdf), [==Code Repository==](https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks)

## งานนี้โดยหลักแล้ว
พยายามที่จะเพิ่ม performance ของโมเดลโดยการลด computational complexity ลง และลดขนาดของ state-of-the-art model  ให้กลายเป็น lightweight model.

## Related work
### Base Architecture
มีการใช้ base architecture จากงาาน ___ มาปรับปรุง ซึ่งงานดังกล่าวได้ state-of-the-art ของ dataset LRW กับ LRW1000

### Efficient Backbone
เพื่อเพิ่มประสิทธิภาพของ model เลยพยายามหา Backbone ตัวใหม่สำหรับแทนที่ ResNet-18 ซึ่งมีตัวนึงชื่อ ShuffleNet v2 ($\beta \times$ โดยที่ $\beta$ คือ Width Multiplier)
#### เกี่ยวกับ ShuffleNet v2
- ใช้ Separable Convolutions แทน CNN ธรรมดา ซึ่งช่วยลดจำนวน parameters  และ computation complexity ลง 

### Depthwise Separable TCN

### Knowledge Distillation
เป็นหนึ่งในเทคนิคของการ Transfer learning

# Framework

# Mathematical Stuff
## Objective function

# Experiment Details
## Datasets
## Evaluation Metrics