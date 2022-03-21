# General Details
[Abstract](https://dl.acm.org/doi/abs/10.1145/3123266.3123326) [PDF File](https://dl.acm.org/doi/pdf/10.1145/3123266.3123326)
## งานนี้โดยหลักแล้ว
- พยายามลด Modality Gap โดยการเรียนรู้ผ่าน Representation Learning เพื่อ Project data แต่ละ modal มาบน common feature representation space

## สิ่งที่งานนี้แตกต่าง
- งานอื่นให้ความสำคัญเฉพาะ คู่ data ที่ต่าง modality แต่งานนี้ไม่ได้รองรับแค่คู่ แต่รองรับ data หลาย modality ด้วย

## Related work
- จากที่ Wang et al. ไปศึกษางานด้านนี้มาพบว่า ด้าน Cross modal retrieval แบ่งตาม Represent vector ได้ 2 แบบ
	- **Real-valued**
	- **Binary representation** (aka. Cross-modal hashing) ดูเหมือนว่า ตัวนี้จะดีกว่าในแง่ของ Computation Retrieval Efficiency แต่จะมี data loss นิดหน่อย
- และยังสามารถแบ่งตามลักษณะการ Train ได้ 4 แบบ
	-  **Unsupervised**
	-  **Pair-wised**
	-  **Ranking-based**
	-   **Supervised** (งานนี้อยู่ในประเภทนี้)
- งานนี้ได้มีการใช้ Triplet ranking loss ซึ่ง inspired มาจาก Ranking-based**

# Framework
![[acmr_framework.png]]
- มี **Feature Projector** สำหรับแต่ละ modal มีหน้าที่ project feature vector จากแต่ละ modal ไปยัง **Common Representation Space** 
- จากนั้น จะมี **Modality Classifier** เป็น **Discriminator** ที่พยายามแยกว่าแต่ละ **Common Representation Vector** นั้นมาจาก modal ไหนบ้าง ซึ่ง **Feature Projector** ต้องหลอก **Modality Classifier** ให้ได้… เพื่อที่จะ preserve data ใน Representation Vector เลยมี loss สำหรับ Label Prediction ด้วย

# Mathematics Stuff
## Objective function
ซึ่งจาก paper นี้จะทดลองบน text กับ image modal เป็นหลัก
$$
\DeclareMathOperator*{\argmin}{argmin}
(\hat{\theta}_V,\hat{\theta}_T, \hat{\theta}_{imd})
= \argmin_{\theta_V,\theta_T, \theta_{imd}}
(\mathcal{L}_{emb}(\theta_V,\theta_T, \theta_{imd}) - \mathcal{L}_{adv}(\hat{\theta}_D))
$$
- Objective function นี้เพื่อทำให้ feature projector ทั้ง 2 modal สามารถสร้าง represent vector ที่ยังคง preserve data ได้ และยังสามารถ confuse Modality Classifier ได้อีกด้วย

$$
\DeclareMathOperator*{\argmax}{argmax}
\hat{\theta}_{D} = 
\argmax_{\theta_{D}}(\mathcal{L}_{emb} (\hat{\theta}_V,\hat{\theta}_T, \hat{\theta}_{imd}) - \mathcal{L}_{adv}(\theta_{D}) )
$$
- ส่วนอันนี้ทำให้ Modality Classifier สามารถจำแนกได้ว่า Representation Vector นี้มาจาก Modal ไหน

**Notation** ของสมการเล็กน้อย 
  
 Symbol          | ความหมาย 
 --------------- | ----------------------------------
 $\theta_{V}$    | Image feature projector parameters.
 $\theta_{T}$    | Text feature projector parameters. 
 $\theta_{imd}$  | Label predictor parameters.
 $\theta_{D}$    | Modality classifier parameters.

### Smaller terms ($\mathcal{L}$ ทั้งหลาย)
#### Embedding Loss ($\mathcal{L}_{emb}$)
สำหรับ loss นี้จะประกอบด้วย 3 term ด้านใน
$$
\mathcal{L}_{emb}(\theta_{V},\theta_{T},\theta_{imd}) = \alpha \cdot \mathcal{L}_{imi} + \beta \cdot \mathcal{L}_{imd} + \mathcal{L}_{reg}
$$
- **$\mathcal{L}_{imi}$  (Inter-Modal Invariance Loss)**
  $$\mathcal{L}_{imi}=\mathcal{L}_{imi,V}(\theta_{V})+\mathcal{L}_{imi,T}(\theta_{T}) 
$$  
  - $\mathcal{L}_{imi,V}(\theta_{V})=\sum_{i,j,k}(l_2(v_i,t^{+}_j)+\lambda\cdot\max(0,\mu-l_2(v_i,t_k^-)))$
    
  - $\mathcal{L}_{imi,T}(\theta_{T})=\sum_{i,j,k}(l_2(t_i,v^{+}_j)+\lambda\cdot\max(0,\mu-l_2(t_i,v_k^-)))$
    
  - $l_2(v,t) = ||f_V(v;\theta_V) - f_T(t;\theta_T)||_2$

  - $(v,t^+)$ คือ text feature $t$ ที่มี semantic label เหมือนกับ image feature $v$ แต่ถ้าเป็น $t^-$ คือ semantic label ไม่เหมือนกับ $v$ ส่วน $(t,v^+)$ เหมือนกันแต่แค่สลับ image กับ text กัน ตัวที่ไม่มี $+,-$ ใน paper จะเรียกว่า anchor
  - $\mathcal{L}_{imi}$ พยายามที่จะลด Distance ของ vector ที่เป็น class เดียวกันและผลักคนละ class ออกจากกัน
- **$\mathcal{L}_{imd}$ (Intra-Modal Discrimination Loss)**
$$\mathcal{L}_{imd}(\theta_{imd}) = -\frac{1}{n} \sum^n_{i=1}(y_i \cdot (\log(\hat{p}_i(v_i))+ \log(\hat{p}_i(t_i))))$$
  - เป็น Cross Entropy Loss ซึ่งให้ค่า Classification Error ของ Label Predictor
- **$\mathcal{L}_{reg}$ (Regularization Loss)**
$$\mathcal{L}_{reg} = \sum^L_{l=1}(||W^l_v||_F + ||W^l_t||_F)$$
  - $L$ คือจำนวน layer ทั้งหมดของ model
  - ใช้สำหรับกัน Over-fitting ตอนนี้คิดว่า function นี้สำหรับ Feature Projector ของทั้ง 2 modal
  - ใช้ [Frobenius Norm](utils/Mathematics%20Function#Normalization##Frobenius%20Norm) ในการทำ Weight Normalization
#### Adversarial Loss ($\mathcal{L}_{adv}$)
$$
\mathcal{L}_{adv}=-\frac{1}{n}\sum^{n}_{i=1}(m_i\cdot(\log D(v_i;\theta_D) + log(1-D(t_i;\theta_D)))
$$
คิดว่าเป็น Cross entropy loss (เดะลองถามพี่สิทธิ์อีกทีนึง) ที่วัดว่า Modality Classifier D สามารถจำแนก represent vector ได้ว่ามาจาก modality ไหน ได้ดีขนาดไหน โดยที่ v และ t คือ Representation vector จาก image modality และ Representation vector จาก text modality ตามลำดับ

# Algorithm
- **Init:** Image batch $V$, Text batch $T$, label batch $Y$
- **Update** until convergence
  - **for** $k$ **steps** (ดูเหมือนว่า ปรับพวก feature projector กับ label predictor ไปก่อน $k$ รอบ)
  -  Update parameters $\theta_V$ (Image feature projector), $\theta_T$ (Text feature projector), $\theta_{imd}$ (Label predictor) via **SGD**
    $$ \DeclareMathOperator*{\argmin}{argmin}(\hat{\theta}_V,\hat{\theta}_T, \hat{\theta}_{imd}) = \argmin_{\hat{\theta}_V,\hat{\theta}_T, \hat{\theta}_{imd}}(\mathcal{L}_{emb}(\theta_V,\theta_T, \theta_{imd}) - \mathcal{L}_{adv}(\hat{\theta}_D))$$
   - Update parameters $\theta_D$ (Modality Classifier) by **ascending its stochastic gradients** through [Gradient Reversal Layer](utils/Machine%20Learning#Special%20Layer%20Definition%20##Gradient%20Reversal%20Layer%20(GRL)) $$
\DeclareMathOperator*{\argmax}{argmax} \hat{\theta}_{D} = \argmax_{\theta_{D}}(\mathcal{L}_{emb} (\hat{\theta}_V,\hat{\theta}_T, \hat{\theta}_{imd}) - \mathcal{L}_{adv}(\theta_{D}) )
$$

# Experiment Details
## Datasets
- Wikipedia
- NUS-WIDE-10K
- Pascal Sentence
- MS-COCO (Multiple class label)

## Evaluation Metrics
- Mean Average Precision