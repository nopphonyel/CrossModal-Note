# General Details
[==Abstract==](https://arxiv.org/abs/2011.07557v1), [==PDF File==](https://arxiv.org/pdf/2011.07557v1.pdf), [==Code Repository==](https://github.com/VIPL-Audio-Visual-Speech-Understanding/learn-an-effective-lip-reading-model-without-pains)
## งานนี้โดยหลักแล้ว
เป็นงานที่ศึกษาว่า ทำยังไงถึงจะ Improve Lip Reading Model ได้ ซึ่งสิ่งที่งานนี้ Contribute จะเป็นการทำ Refinement เบื้องต้น ซึ่งเพียงเท่านี้ เขาก็เคลมมาแล้วว่า สามารถเอาชนะพวก state-of-the-art Model ได้

## Related work
ยุคแรกเริ่มของ Lip reading มักจะใช้พวก hand-craft features (น่าจะอารมณ์เหมือนหา wavelet เอง) กับ shallow-model เข้าด้วยกัน เช่น Hidden Markov Model, Discrete Cosine Transform, Active Appearance Model

พอมาถึงยุคหลัง Stafylakis et al. ได้ลองใช้ ResNet ในส่วนหน้าหรือ Front-end ของ Model และได้ผลที่ดีขึ้น จากนั้นเป็นต้นมา Model หรืองานใหม่ๆมักจะถูกแบ่งออกเป็น 2 ส่วนนั่นคือ
1. **Front-end Module** ส่วนนี้จะเน้นการประมวลผลพวก local motion patterns (น่าจะแค่ส่วนนึงของ Sequence) รวมถึง frame-level และ clip-level features 
2. **Back-end Module** จะเน้นการประมวลผลทั้ง Sequence และบางครั้งจะเรียนรู้พวก Temporal dynamics 

งานนี้ได้ไปหาเทคนิคที่จะเพิ่มประสิทธิภาพต่างๆจาก Computer Vision Field ด้วยเพราะ Lip reading ก็อาศัยส่วนหนึ่งของ Computer Vision เช่นกัน ยกตัวอย่างงานของ 
- He et al. ที่รวบรวมเทคนิคการ train และอื่นๆเพื่อเพิ่มความแม่นยำของ Image classification
- Luo et al. ที่ปรับแต่งการ train นิดหน่อยและได้ค้นพบ strong กับ simple baseline ในการทำ Person Re-identification

# Framework

![[effectiveLipReadNoPainModel.png|460]]

ส่วน Front-end ของ Paper นี้เป็น ResNet-18 ที่มีการปรับเปลี่ยนบางอย่าง
- 2D Convolution ใน layer แรกเปลี่ยนไปเป็น &rarr;  **3D Convolution** ด้วย Kernel 5$\times$ 7 $\times$ 7 
- มีการใช้ Global Average Pooling กับ Output ของแต่ละ Residual Block

หลังจากผ่านส่วน Front-end แล้ว ก็จะเอา Feature จาก Residual block เหล่านี้ส่งต่อไปที่ Back-end แล้วตามด้วย Fully Connected (FC) layer เพื่อระบุ class ของ word จากปากคนพูด 

> [!My Thought]
> ตอนแรกเข้าใจว่า Recon เป็นเสียงออกมา แต่จริงๆแล้ว Classify แต่ละ Frame ออกมาเป็น Text นี่หว่า

# Experiment Details

## Initialization
- **CNN Layer** init ด้วย

  - ค่า Random จาก Uniform Distribution
  
  - Uniform จากช่วง $[-a, a]$

  - $a=\sqrt{2/(d_{in}+d_{out})}$

  - $d_{in}, d_{out}$ คือ Input size และ Output size ของ CNN layer นั้นๆ 

- **Batch-Norm layer** init ด้วย
	- Weight $\gamma = 1$
	
	- Bias $\beta = 0$

- **GRU layer** (Back-end) 

	- Parameter ทั้งหมด sample จาก $(-1,1)$ 

- **FC layer**
	- Parameter sample จาก Uniform Distribution ของ $[-1,1]$ 

## Data Processing
ตรงนี้ค่อนข้างสำคัญเพราะเราคงต้องเล่นกับ Dataset LRW, LRW-1000
- Shuffle order ของ input video
- Resize ขนาด video เป็น &rarr; 96 $\times$ 96
- Random crop เป็น &rarr; 88 $\times$ 88 ก่อน
- Random horizontal flip ด้วย $p$ = 0.5
- เปลี่ยนเป็น Gray scale
- Normalize input tensor เป็น &rarr; $[0,1]$ แล้วเอาเข้า Model

เพิ่มเติมสำหรับ Dataset LRW-1000
- มีการเลือก 40 frame ของแต่ละคำ ถ้าหากคำนั้นสั้นกว่า 40 frame เช่นอาจจะซัก 20 frame Input data ก็จะเป็น  40 frame ที่มี 20 frame อยู่ตรงกลาง หรือพูดสั้นๆ *"เอา Target Word ไว้ตรงกลางของ Input data frame"*
	- การทำแบบนี้ เป็นการช่วยเพิ่ม Context เข้าไปในตัว Training Data ซึ่ง **ช่วยเพิ่มประสิทธิภาพ** ของ lip reading อีกด้วย

 ## Loss
 - หลังจากผ่านส่วนของ Back-end มีการ Average Output ของ Temporal Dimension (ยุบ Sequence ให้เหลือ Time step เดียวด้วยการเฉลี่ย) ก่อนจะโยนเข้า FC Layer ไป Classify ว่าเป็นคำอะไร
 - ดังนั้น จึงใช้ **Cross Entropy Loss** สำหรับการ Optimize model

## Optimizer
- ใช้ **Adam Optimizer** ด้วย config

	- $LR=3\times10^{-4}$
	
	- $Decay_{weight} = 1\times10^{-4}$

- ตอน Train model
	- ถ้า train ด้วย GPU ตัวเดียว
		- Batch size = 32
	- ถ้าหลายตัว
		- Batch size = ??? งงๆ อ่านไม่เข้าใจอ่ะ

## Datasets
หลักๆแล้ว งานนี้ใช้ Dataset อยู่ 2 ตัวคือ
- **LRW** ซึ่งเป็น Dataset video คนพูดภาษาอังกฤษ มีจำนวน 500 word class ซึ่งอัดมาจาก BBC Program
- **LRW-1000** เป็น Dataset คนพูด Chinese Mandarin มีจำนวนคำทั้งหมด 1000 คำ/phrase มีคนพูดมากกว่า 2000 คน


# Results
ผลลัพธ์หลังจากการทดลองสารพัด จนสามารถสรุปเป็น Tips&Tricks ที่จะช่วยพัฒนา Model ให้ดีขึ้นได้
## 1. Model Refinement
จากที่กล่าวไว้แต่ต้น มันจะแบ่งแยกโมเดลออกเป็น 2 ส่วน
### Front-end Network

Front-end | Backend | LRW | LRW-1000
-- | -- | --| -- 
VGGM* | - | 61.1% | 25.7%
ResNet-18* | 3 Layers GRU | 83.0% | 38.2%
ResNet-34* | " | 83.5% | -
ResNet-18 (rerun) | " | 83.7% | 46.5%
SE-ResNet-18 | " | **84.1%** | **46.8%**

<ins>*หมายเหตุ*</ins> ตัวที่ติด * คือไม่ได้รันใหม่ในงานนี้

ได้มีการเปรียบเทียบหลายๆ Model กับ GRU เป็น Baseline ทำให้ได้ข้อสรุปว่า
- Deep convolution network ชนะ Shallow convolution network เสมอใน general task
- ResNet-34 ดูจะทำได้ดีกว่า ResNet-18 เล็กน้อย
- ส่วน ResNet-18 (rerun) คือตัวที่ run ผ่านการเตรียม Data แบบพิเศษตามที่ได้บอกไว้ [==ตรงนี้==](#Experiment%20Details##Data%20Processing) ซึ่งจะเห็นว่าตรง LRW-1000 perform ได้ดีขึ้นอย่างมาก
- SE-ResNet-18 คือตัวที่รวม Squeeze and Extract เข้าไปด้วย ดูเหมือนว่า จะช่วยอยู่นิดนึง

### Back-end Network

Front-end | Backend | LRW | LRW-1000
-- | -- | --| -- 
ResNet-18 | 3 Layers GRU |**83.7%** | **46.5%**
"| GRU w/o dropout | 83.1% | 45.5%
" | MS-TCN | 83.4% | 43.0%
" | Transformer* | 76.2% | 44.5%

> [!My though]
> ส่วนตัวแปลกใจนิดๆว่าทำไม GRU ถึงชนะ model อื่นๆไปได้

## 2. Data Processing

Data Processing | LRW | LRW-1000
-- | -- | --
Baseline | 83.7% | 46.5%
Aligned Lip | 84.2% | -
Word Boundary Input | 86.5% | 53.6%

เทคนิคที่ใช้ก็จะมี 
- **Face Alignment &rarr; Aligned Lip**  งานของ Zhang et al. ได้ confirm Face alignment ว่าสามารถช่วยเพิ่ม accuracy ของการทำ Visual Speech Recognition ได้ 
	- งานนี้เลยเอาเทคนิคนี้มาใช้บ้างโดย Alignment Face ก่อนการทำ Lip region extraction
	- ใช้เครื่องมือ **dlib toolkit** ในการจับ feature หน้าต่างๆ
	- จากนั้นใช้เครื่องมือเดิมในการ center crop ส่วนของริมฝีปาก ออกมาเป็น 4 เหลี่ยมจตุรัส
- **Word Boundary** 
	- ไอเดียคือการใช้ Tensor ขนาด 1 เป็น Indicator ว่าแต่ละคำสิ้นสุดตรงไหน ซึ่งคล้ายๆกับ สัญญาณ clock ใน CPU
	- Stafylakis et al. ได้เชื่อว่า RNN ซึ่งมี Gating Mechanism นั้น จะสามารถเอา Indicator พวกนี้ไปใช้ประโยชน์ได้

## 3. Training Tweaks

Training Tweaks | LRW | LRW-1000
-- | -- | --
Baseline | 83.7% | 46.5%
MixUp | 84.0% | 47.3%
Label Smooth | 84.2% | 47.0%
Cosine Scheduler | 84.2% | 46.6%
Expo Scheduler | 83.2% | 45.6%

- **MixUp** เป็นการทำ Data augmentation โดยการนำ data sample A $(x_A, y_A)$ กับ data sample B $(x_B, y_B)$ มาผสมกันเป็น $(\hat{x}, \hat{y})$
  
  $\hat{x} = \lambda x_{A} + (1-\lambda)x_{B}$
  
  $\hat{y} = \lambda y_{A} + (1-\lambda)y_{B}$
  
	- $\lambda$ เป็นค่านึงที่ sample มาจาก $Beta(\alpha, \alpha)$ Distribution ซึ่งงานนี้ใช้ค่า $\alpha = 0.2$

	- $x_A$ และ $x_B$ sample มาจาก batch $S$ และ $S'$ โดยที่ $S' =  Shuffle(S)$
	
	ซึ่งการทำแบบนี้ จะทำให้ Model มีความเป็น Linearity มากขึ้นซึ่ง Linearity เป็นหนึ่งใน Inductive Bias ที่ดีจากหลักการของ Occam's razor (ซึ่งกล่าวไว้ว่า: "เราไม่ควรสร้างข้อสมมุติฐานเพิ่มเติมโดยไม่จำเป็น" หรือ "ทฤษฎีไม่ควรซับซ้อนเกินความจำเป็น")

- **Label Smoothing**
  โดยปกติแล้ว การคำนวณ Cross Entropy เมื่อ<ins>ไม่ได้ทำ Label Smoothing</ins> จะคำนวนแบบนี้
  $$
  L = -\sum^N_{i=1}q_i\log(p_i)
  \begin{cases}
  q_i=0 & ,y \neq i \\
  q_i = 1 &,y = i
  \end{cases}
  $$
  โดยที่ $i$ คือ Class ของ Word, $N$ คือจำนวน Class ทั้งหมดที่มี, $p_i$ คือค่า Prediction จาก Model และ $y$ คือ annotated word label (ยังไม่ค่อยแน่ใจว่าคืออะไร) แต่เมื่อทำ Label Smoothing แล้ว <ins>จะมีการ**เปลี่ยนแปลงการคำนวณ**ของ $q_i$</ins> เป็น
  $$
  q_i = 
  \begin{cases}
	  \epsilon / N & , y \neq i \\ 
	  1 - \frac{N-1}{N}\epsilon & ,y=i
  \end{cases}
  $$
  โดยที่ $\epsilon$ เป็นค่าคงที่ซึ่งงานนี้กำหนดค่าไว้ที่ 0.1 ซึ่งการทำแบบนี้ทำให้ Model มีความ Generalize มากขึ้น Confident ไม่สูงเกิดจงช่วยลดการ Over-fitting ได้

### Learning Rate Scheduling
- **Expo Scheduler** 
  คือการเอาค่าคงที่มาคูณกับ Learning Rate $\eta_t$ (LR) ตัวเดิมเรื่อยๆ ซึ่งงานนี้ใช้ 0.95 มาคูณเพื่ออัพเดทค่า LR ทุกๆครับที่จบ 1 epoch
  $$
  \eta_t =  0.95 \cdot \eta_{t-1}
  $$

- **Cosine Scheduler**
  มีการอัพเดท Learning Rate $\eta_t$ ดังนี้
  $$
  \eta_t = \frac{1}{2}(1+\cos(\frac{t\cdot\pi}{T}))\eta
  $$
  โดยที่ $\eta$ คือค่า LR เริ่มต้น ส่วน $T$ คือจำนวน Epoch ที่ใช้เทรนซึ่งงานนี้มี 80 การทำงานของสูตรนี้จะทำให้ LR ลดน้อยๆในช่วงแรก เนื่องจากว่า ช่วงแรกของการ Train โมเดลจะมีการปรับ parameter ที่สำคัญๆและมีการปรับอย่างมาก พอช่วงกลาง LR จะลดเกือบๆ Linear จากนั้นพอใกล้จบ จะเป็นการ Fine tune สักนิดสักหน่อย อัตราการลดของ LR ก็จะลดลงในช่วงท้ายอีกรอบ

## The Final Pipeline
สุดท้ายนี้ งานนี้เลยเอา 
- **SE-ResNet-18** มาใช้
- ทำ Data Augmentation แบบ **MixUp**
- เลือกใช้ **Cosine Learning Rate Scheduling**
- ทำ **Label Smoothing**
- และทำ **Word Boundary**

เลยชนะ State-of-the-art ทั้ง Dataset LRW และ LRW-1000 ไปอย่างสบายๆ