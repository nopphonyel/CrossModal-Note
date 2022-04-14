# General Details
[==Abstract==](https://arxiv.org/abs/2007.06504v3), [==PDF File==](https://arxiv.org/pdf/2007.06504v3.pdf), [==Code Repository==](https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks)

## งานนี้โดยหลักแล้ว
พยายามที่จะเพิ่ม performance ของโมเดลโดยการลด computational complexity ลง และลดขนาดของ state-of-the-art model  ให้กลายเป็น lightweight model.

> [!Note]
> งานนี้จะเป็นการเอา Video มา Predict word โดยตรง ไม่ได้ทำ Cross modal ระหว่าง Text กับ Video Frame แต่อย่างใด

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

นอกจากนี้ งานนี้ได้ย้อนไปดู proof ที่ว่า KD ก็ยังคงมีประโยชน์เหมือนกัน แม้จะมี layer/neuron จำนวนเท่าเดิมก็ตาม เนื่องจาก knowledge ข้าม generation (น่าจะหมายถึงข้าม epoch) ก็สามารถนำมาใช้เป็น teacher ให้ generation หลังๆได้

# Framework

# Mathematical Stuff
## Objective function
งานนี้ใช้ Born-Again Distillation (จากที่ดูผ่านๆคร่าวๆคือ Teacher กับ Student มี Arch ที่เหมือนกันเลย และจะมี Objective function 2 อย่างคือ Predict correct label และ Matching output distribution กับของ Teacher) แต่ก็ใช้ Standard Distillation (Distillation ที่ลดจำนวน Parameter ใน Student ลง) ร่วมด้วย

KD ทั้ง 2 แบบ มีการใช้ Cross-entropy loss สำหรับ Minimize Hard target (ผลของการ Predict class) และใช้ KL divergence loss สำหรับ Soft target (พวก Feature ระหว่าง layer)

$$
\mathcal{L} = \mathcal{L}_{CE}(y, \delta(z_s;\theta_s)) + \alpha \
\mathcal{L}_{KD}(\delta(z_s;\theta_s), \delta(z_t;\theta_t))
$$
- $\mathcal{L}_{CE}$ คือ Cross-Entropy Loss

- $\mathcal{L}_{KD}$  คือ Kullback-Leibler Divergence Loss

- $y$ คือ Label (Hard Target)

- $\theta_s$ กับ $\theta_t$  คือ Student model กับ Teacher model ตามลำดับ

- $z_s$ กับ $z_t$ คือ Prediction ของ Student กับ Teacher ตามลำดับ

- $\delta(\cdot)$ คือ Soft max Function 

- $\alpha$ คือ Hyper parameter สำหรับ Balance term ทั้งสอง

# Experiment Details
## Datasets
- LRW (Lips Reading in the Wild)
- LRW-1000

## Pre-processing
สำหรับ Dataset LRW 
- มีการใช้ dlib ในการ aligned face
- ตัดแต่ละ frame ออกมาเป็นขนาด 96x96 เอาเฉพาะส่วนปาก

ส่วน LRW-1000 มีการตัดปากเรียบร้อยแล้ว จึงไม่จำเป็นต้องทำอะไรเพิ่มเติม

## Training
มีการอ้างถึงงาน 21 (เดะไปดูเพิ่มว่าคืองานอะไร) ว่า Training parameter ใช้เหมือนงานนี้ทุกอย่าง ยกเว้นเปลี่ยนตัว Optimizer คือ Adam with decoupled Weight decay (AdamW)
- $\beta_1=0.9$

- $\beta_2=0.999$

- $\epsilon = 10^{-8}$

- Weight decay = 0.01

- Epoch = 80

- LR = 0.0003 

	- Learning rate decay: *Cosine annealing schedule*  ไม่มี Warm-up steps


## Data Augmentation
แต่ละ Sequence (น่าจะหมายถึง 1 วิดีโอ) 
- มีการสุ่มพลิกตามแนวนอน 
- สุ่มตัดให้เหลือขนาด 88x88
- มีการทำ Variable-length augmentation เหมือนกับงาน 21

## Result
งานนี้ใช้ Eval metrics คือ Accuracy ของการ classify คำจาก Video Sequence

### 1. บน LRW Dataset
**ตารางที่ 1**: Train บน *LRW* Dataset

Method | Top-1 Acc. %
---|---
3D-CNN | 61.1
Seq-to-Seq  | 76.2
ResNet34 + BLSTM | 83.0
ResNet34 + BGRU | 83.4
2-stream 3D-CNN + BLSTM | 84.1
ResNet18 + BLSTM | 84.3
ResNet18 + BGRU + Cutout | 85.0
ResNet18 + MS-TCN | 85.3

**ตารางที่ 1.1**: เอา ResNet18 + MS-TCN มาทำ KD ต่อ

Method | Top-1 Acc. %
---|---
ResNet18 + MS-TCN - Teacher| 85.3
ResNet18 + MS-TCN - Student 1 | 87.4
ResNet18 + MS-TCN - Student 2 | 87.8
ResNet18 + MS-TCN - Student 3 | 87.9
ResNet18 + MS-TCN - Student 4 | 87.7
Ensemble | **88.5**

### 2. บน LRW-1000 Dataset

**ตารางที่ 2**: Train บน *LRW-1000* Dataset

Method | Top-1 Acc. %
---|---
ResNet34 + DenseNet52 + ConvLSTM | 36.9
ResNet34 + BLSTM | 38.2
ResNet18 + BGRU | 38.6
ResNet18 + MS-TCN | 41.4
ResNet18 + BGRU + Cutout | 45.2 [^1]

[^1]:งานนี้เอาใบหน้าทั้งหน้าเข้าไป Train model ซึ่งไม่ใช่ Public dataset

**ตารางที่ 2.1**: เอา ResNet18 + MS-TCN มาทำ KD ต่อ

Method | Top-1 Acc. %
---|---
ResNet18 + MS-TCN - Teacher | 43.2
ResNet18 + MS-TCN - Student 1 | **45.3**
ResNet18 + MS-TCN - Student 2 | 44.7
Ensemble | **46.6**

### สรุปผลการทดลอง
- จากตารางทั้ง 2 ในนี้ไม่ได้บอกว่าทำ Ensemble แบบไหน ต้องไปตามอ่านงานที่ [14]
- ถ้าให้สรุป สิ่งเหล่านี้ช่วยให้สามารถเอาชนะ State-of-the-art ได้
	- การทำ KD
	- ลด Parameter ด้วยการใช้ Depthwise convolution
	- การทำ Ensemble
- จุดที่น่าสนใจอีกอย่างคือ **ResNet18** <ins>ทำงานได้ดีกว่า</ins> **ResNet34** ทั่ง 2 dataset มีความเป็นไปได้เนื่องจาก Parameter ที่น้อยกว่าทำให้มีความ Generalize มากกว่า
