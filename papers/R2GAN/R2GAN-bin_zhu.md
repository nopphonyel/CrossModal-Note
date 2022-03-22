# General Details
[==Abstract==](), [==PDF File==](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhu_R2GAN_Cross-Modal_Recipe_Retrieval_With_Generative_Adversarial_Network_CVPR_2019_paper.pdf)

## งานนี้โดยหลักแล้ว
พยายามจะทำ cross modal ระหว่าง Food Image กับ Recipe ที่เป็น Text

## Challenge
เนื่องจาก Recipe เป็น text ที่มีความยาวค่อนข้างมาก ไม่ใช่แค่ประโยคสั้นๆ จึงต้องมีโมเดลเฉพาะ นอกจากนี้ Recipe ไม่ได้อธิบายลักษณะของภาพโดยตรง แต่เป็นการบอกสาเหตุของอาหารว่าทำไมมันมีหน่าตาแบบนี้

## Related Work

### ด้าน Food Retrieval
-   **Stack Attention Model** โดยเอาสูตรส่วนผสมแปลงเป็น Binary vector แล้วโยนให้ Model แต่ทว่างานนี้ยังไม่ได้แก้ Bug: อาหารบางอย่างมีส่วนผสมเหมือนกันแต่วิธีทำต่างกัน
-   **Joint Neural Embedding (JNE):** จาก bug งานที่แล้ว เลยแก้โดยเอา cooking procedure เข้าไปด้วย โดยมี bi-directional LSTM encode ส่วนผสม และ Hierarchical LSTM encode cooking procedure นอกจากนี้มีส่วน sematic loss ที่ train โดยให้ model predict category ของอาหาร และดูเหมือนว่าจะสำคัญ    
-   มีการใช้ **AdaMine** (Adaptive Mining Embedding) โดยใช้ double-triplet learning และ adaptive strategy  (ยังไม่ค่อยแน่ใจว่ามันคืออะไร) สำหรับ informative triplet mining ซึ่งให้ผลดีมาก งานนี้เลย adaptive strategy ด้วย

### ด้าน Cross-modal GAN
- [**<ins>ACMR</ins>**](papers/ACMR/ACMR-bokun_wang) Modal Discriminator สำหรับทาย modal 1 ตัวและมี Feature projector ที่จะพยายามหลอก Modal Discriminator ให้ได้
- **GXN** 2 Generator, 2 Discriminator. แต่ละ generator gen features แล้วให้ discriminator แต่ละตัวทายว่า fake/real
- **CM-GANs**  2 Generator, 2 Discriminator ทั่วๆไปแหละมั้ง

# Framework
![[r2gan_framework.jpg]]
- แต่ละส่วนของ recipe จะเข้าคนละ model แล้วเอามา concat กันเป็น embedding $E$ เพื่อให้มั่นใจว่า embedding ยังคง preserve ข้อมูลไว้ จึงมี Semantic Loss เข้ามาช่วยไว้
- ส่วน image ก็เอามาผ่าน CNN เพื่อสร้าง Embedding จากนั้นจะมี Discriminator $D_1$ ทาย real fake กับ $D_2$ ทาย modal (ไอเดียคล้าย [ACMR](papers/ACMR/ACMR-bokun_wang)), จากนั้นมี Two-level Ranking Loss มาเพื่อ… ออกแนวตัวช่วยมากกว่า คิดว่าถ้าตัดออก อาจจะยัง work แต่ใช้เวลา train นานขึ้นมากๆ?

# Mathematical Stuff
## Objective Function
ซึ่งมีอยู่ 3 สมการย่อย
### $\mathcal{L}_{full}$
$$
\mathcal{L}_{full} = \mathcal{L}_{rank} + \gamma\cdot \mathcal{L}_{recon} + \lambda \cdot \mathcal{L}_{sem}
$$
$\gamma,\lambda$ เป็น constant ใช้สำหรับปรับ balance ของแต่ละ term ซึ่งได้อธิบายไว้ที่ด้านล่างนี้

- **Ranking Loss $(\mathcal{L}_{rank})$ **
    $$ \begin{aligned}
  \mathcal{L}_{rank} = &\max\{d(E_q, E_p)-d(E_q, E_n)+\alpha_1, 0\}+ \\
  & \mu\max\{d(E_q, E_p)-d(E_q, E_n)+\alpha_1, 0\}
  \end{aligned} $$
  - โดย $d$ คือ distance function วัด similarity distance (Cosine similarity) ระหว่าง $E$ (Embedding ของ recipe หรือไม่ก็ image) ซึ่งจะมีการจับคู่วัดที่เป็น Positive $E_p$ (เป็นของ item เดียวกันไม่ก็ class เดียวกัน) และ Negative $E_n$
  - ส่วนอีก term ก็มีการหา distance เหมือนกัน (Pixel-wise Euclidean) แต่เป็นระหว่าง image… จากที่ดูจาก framework เหมือนว่าจะหา distance เฉพาะ real กับ real หรือ fake กับ fake เท่านั้น
  - **สรุป:** ลด distance คู่ Positive และเพิ่ม distance คู่ Negative ทั้ง Embedding และ image
- **Reconstruction Loss $(\mathcal{L}_{recon})$**

  $$ \begin{aligned} \mathcal{L}_{recon}= & \frac{1}{2}(||\Phi(v_{real}) - \Phi(v_f^I)||^2_2+||\Phi(v_f^I) - \Phi(v_f^R)||^2_2 \\
  & +\beta(||v_{real} - v^I_f||^2_2 + ||v_f^I - v_f^R||^2_2))
  \end{aligned}
  $$
    - $\Phi$  คือ output ของ Discriminator ก่อน layer สุดท้าย, $v^I_f$ กับ $v^R_f$ คือ reconstructed image จาก image กับ recipe -> เลยสรุปได้ว่าที่เอา $v$ มาลบกันคือ image level loss ส่วนที่เอา $\Phi$ มาลบกันคือ feature level loss
    -  จากการดูแบบคร่าวๆแล้ว คือการทำ pixel wise loss ทั้งฝั่ง image และ feature ก่อนที่จะผ่าน layer สุดท้ายของ discriminator

- **Semantic Loss? $(\mathcal{L}_{sem})$**
$$\mathcal{L}_{sem}=-\log{\frac{\exp(E_c)}{\sum_i \exp(E_{c_{i}})}}$$
  - $E_c$ คือ Embedding category ($E$ นี้มาจาก modal ไหนก็ได้)
  - ใน paper บอกว่านี่คือ cross entropy loss ดังนั้นใน **```pytorch```** implementation ก็เป็น label ได้เลย

### Generator's Loss function $(\mathcal{L}_{G_{full}})$
$$\mathcal{L}_{G_{Full}}=\mathcal{L}_G + \delta \cdot \mathcal{L}_{recon}$$
และเช่นกัน $\delta$ เป็น constant สำหรับปรับ balance

### Discriminator's Loss function $(\mathcal{L}_{D1},\mathcal{L}_{D2})$
คงเป็น loss function แบบ Discriminator ทั่วไป

# Experiments Details
## Datasets
ใช้แค่ dataset เดียวคือ **Recipe1M** แต่ว่าเป็นpre-processed Recipe 1M ซึ่ง data จะลดลงมานิดนึง
## Evaluation Metrics
- **Median Rank (MedR)**: refers median rank position of true positive for all the testing queries. (Lower is better) ไม่ค่อยเข้าใจความหมายแฮะ
- **Recall rate at top K (R@K)**: measure the fraction of true positive being ranked at top K returned results (Higher is better) งงๆความหมายนิดหน่อย