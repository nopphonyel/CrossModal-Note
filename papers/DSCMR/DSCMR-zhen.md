# General Details
[==Abstract==](), [==PDF File==](), [==Code Repository==](https://github.com/penghu-cs/DSCMR)
## งานนี้โดยหลักแล้ว
ใช้ Deep learning ในการแก้ปัญหา modality gap

## Contribution
แก้ปัญหาที่งานก่อนหน้าไม่ได้ใช้ประโยชน์ของ semantic information อย่างเต็มที่
- งานเก่าใช้ classification information ในการเรียนรู้ discriminative feature ระหว่าง modal หรือไม่ก็ ข้าม modal ไปเลยแค่อย่างใดอย่างนึงเท่านั้น เข้าใจว่างานนี้จะใช้ร่วมกัน

## Related work
- จากเปเปอร์นี้ แยกประเภทงานที่เกี่ยวข้องเหมือน [ACMR](papers/ACMR/ACMR-bokun_wang#General%20Details##Related%20work) แต่เปเปอร์นี้ขยายความขึ้นนิดนึงว่า **Binary value** ค่อนข้างมี computation efficiency ที่ดี แต่ว่า retrieval acc ก็จะลดลงนิดนึงเพราะการสูญเสียข้อมูล
- ด้วยเหตุนี้งานนี้จึงทำเป็นแบบ Real value representation
- แบ่งลงไปอีก จะได้ 3 ประเภทย่อย (ต่างจาก ACMR นิดนึง งานนั้นบอกมี 4 ประเภท)
  - **Unsupervised method** เรียนรู้จากคู่ modal
    - CCA
    - Deep CCA
    - Correspondence AE
    - Deep Canonically Correlated AE
  - **Pairwise-based method** utilise more similar pairs to learn a meaning metric for comparing sample from different modal -> Modality-Specific Deep Structure
  - **Supervised method** พยายามดึง modal ที่อยู่ class เดียวกันให้มี representation vector เหมือนกัน

# Framework
![[dscmr_framework.jpg]]
- ไอเดียคือ มี sub network 2 สำหรับ image กับ text modal
  - image มี arch แบบ VGG19 + FC
  - text เป็น Word2Vec + Sentence CNN + FC (Word2Vec เอาไปทำ embedding) 
- layer สุดท้ายมีการ share weights กัน จากนั้นใช้ common latent มาคิด J3 loss ที่ลด distance ระหว่าง modal, J2 ที่ลด Discrimination Loss บน common latent ซึ่งมีการไขว้ modal กันด้วย, J1 ที่ลด Discrimination loss เหมือนกัน แต่ไม่ได้ทำบน common latent space แต่จะทำบน Label space

# Mathematical Stuff
## Objective function
คล้ายๆเดิม มี 3 term และมี co-efficient เป็น hyper parameter
$$
\mathcal{J}=\mathcal{J}_1 + \lambda\mathcal{J}_2 + \eta\mathcal{J}_3
$$
### Smaller terms (term ย่อยๆต่างๆ)
#### Label discirmination loss $\mathcal{J}_1$
$$ \mathcal{J}_1 = \frac{1}{n}||P^TU-Y||_F + ||P^TV-Y||_F$$
- Loss นี้ไว้สำหรับ minimize discrimination loss ใน label space หรือพูดอีกนัยหนึ่งคือ ทำให้โมเดลเข้าใจ ลักษณะของแต่ละ class มากขึ้น?
- $P$ คือ projection ของ Linear classifier?
- $U$ คือ vector ที่รวม image representation matrix เข้าด้วยกันไว้
- $V$ เป็น vector รวมของ text representation
- $Y$ vector ที่รวม label เข้าไว้ด้วยกัน
- $||?||_F$ คือ [Frobenius Norm](lib/math/norm/Frobenius%20Norm)

> [!Note]
> - $P^TU, P^TV$ คือ การรวม Projection vector เข้ากับแต่ละ Representation vector ของ modal นั้นๆ
> - $P^T? - Y$ เป็นการหา classification loss
>   - แสดงว่า vector $P$ ต้องเป็น vector ที่คูณกับ representation ใดๆแล้วออกมาเป็น class ได้เลย

#### Common Representation loss $\mathcal{J}_2$
อันนี้จะค่อนข้างยาวเพราะมี 3 term ย่อยในสมการ (แต่ใน paper มันเขียนยาวไปเลย)
$$\mathcal{J}_2 = InterModal + ImageModal + TextModal$$

- $InterModal = \frac{1}{n^2}\sum^n_{i,j=1}(\log(1+e^{\Gamma_{ij}}) - S^{\alpha\beta}_{ij}\Gamma_{ij})$
    - โดยมี $\Gamma$ เป็นค่า [Cosine Similarity](lib/math/sim/Cosine%20Similarity) ระหว่าง Image กับ Text modality $\Gamma_{ij}=\frac{1}{2}S_c(u_i,v_j)$
    - มี $S^{\alpha\beta}_{i,j}$ เป็น Indicator function (อารมณ์แบบ Switch เปิดปิด) $S^{\alpha\beta}_{i,j}=1\{\mathrm{u}_i,\mathrm{v}_j\}$
      - ไอเดียคือ ถ้า $\mathrm{u}_i$  กับ $\mathrm{v}_j$ อยู่ใน class เดียวกัน function จะให้ค่า 1 และ 0 ในทางตรงข้าม
      - จะได้ว่า  $S^{\alpha\beta}_{i,j}$ จะปิดการลบของ $\Gamma_{ij}$ เมื่อ $\mathrm{u}_i$  กับ $\mathrm{v}_j$ อยู่คนละ class
    - ยังไม่แน่ใจว่า $\frac{1}{n^2}$ มีไว้แล้วช่วยอะไรบ้าง
    - **สรุปภาษาคน:**  ถ้าคนละ class แล้ว vector $u, v$ คล้ายกัน loss จะมีค่ามากเนื่องจากมี term ของ $e$ ของ $\Gamma$ อยู่ แต่ถ้า class เดียวกันแล้วคล้ายกัน loss จะน้อยเพราะโดนลบออก

- $ImageModal = \frac{1}{n^2}\sum^n_{i,j=1}(\log(1+e^{\Phi_{ij}})-S^{\alpha\alpha}_{ij}\Phi_{ij})$
  - โดยมี $\Phi$ เป็นค่า [Cosine Similarity](lib/math/sim/Cosine%20Similarity) ระหว่าง Image ด้วยกัน $\Phi_{ij}=\frac{1}{2}S_c(\mathrm{u}_i,\mathrm{u}_j)$
  - มี $S^{\alpha\alpha}_{i,j}$ เหมือนเดิมแต่ระหว่าง image ด้วยกันเอง $S^{\alpha\alpha}_{i,j}=1\{\mathrm{u}_i,\mathrm{u}_j\}$
  - **สรุปภาษาคน:**  ถ้า image vector ทั้งสองอันอยู่ class เดียวกัน แล้ว vector เหมือนๆกัน loss จะออกมาน้อย เพราะโดน term $S^{\alpha\alpha}$ ลบออก แต่ถ้าคนละ class แล้วดันเหมือนกัน loss จะสูงเพราะ term $e$ ของ $\Phi$
   
- $TextModal = \frac{1}{n^2}\sum^n_{i,j=1}(\log(1+e^{\Theta_{ij}})-S^{\beta\beta}_{ij}\Theta_{ij})$
  - โดยมี $\Theta$ เป็นค่า [Cosine Similarity](lib/math/sim/Cosine%20Similarity) ระหว่าง Text ด้วยกัน $\Theta_{ij}=\frac{1}{2}S_c(\mathrm{v}_i,\mathrm{v}_j)$
  - มี $S^{\beta\beta}_{i,j}$ เหมือนเดิมแต่ระหว่าง text ด้วยกันเอง $S^{\beta\beta}_{i,j}=1\{\mathrm{v}_i,\mathrm{v}_j\}$
  - **สรุปภาษาคน:**  text vector เหมือนกันและอยู่ class เดียวกัน -> loss ต่ำ... แต่ถ้า text vector เหมือนกันแต่อยู่คนละ class -> loss สูง

#### ?? loss $\mathcal{J}_3$
อันนี้ไม่ค่อยมีไรล้ะ อันที่จริง หน้าที่มันดูซ้ำซ้อนกับ $\mathcal{J}_2$ ด้วยซ้ำ
$$ \mathcal{J}_3 = \frac{1}{n}||\mathrm{U}-\mathrm{V}||_F $$
- มีหน้าที่ลด Distance ของ representation ระหว่าง modal โดยการเอามาลบกันแล้วทำ [Frobenius Norm](lib/math/norm/Frobenius%20Norm) เหมือนเดิม

# Experiment Details
## Datasets
- Wikipedia Datasets
-   Pascal Sentence
-   NUS-WIDE-10k
-   X-MediaNet

## Model Features
-   มีการใช้ weight ของ VGG19 pre-trained ด้วย ImageNet
  -   ใช้ VGG19 เพื่อนที่จะเอามาสร้าง 4,096-dim representation vector ของ image
-   ใช้ weight ของ Sentence CNN เพื่อสร้าง 300-dim representation vector ของ text

## Evaluation Metrics
- [mAP (mean Average Precision)](lib/math/metrics/Mean%20Average%20Precision)