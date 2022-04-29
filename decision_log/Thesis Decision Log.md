# Decision Log
จัดเรียงลำดับตาม Recent
## 2022-04-

## 2022-04-26
### AutoEncoder-Experiment 01 Results
**Model Objective ในการทดลอง**
- AE + Dis(recon/real)
- AE + Dis(recon/real) + 0.5PixelWise
- AE + Dis(recon+real/real+real)
- AE + Dis(recon+real/real+real) + 0.5PixelWise

![[exp01_eval_pixelwise_mnist.png|500]]

**Discussion**
จะเห็นได้ว่า Discriminator ทำให้ AutoEncoder  พยายามที่จะ Gen รูปที่แยกไม่ออกจนเกินไป โดยไม่สนเรื่องการทำ Encoding แล้ว ทำให้ค่า Loss สูงขึ้นในตอนหลังอย่างมาก 

**What's Next**
- ถ้าลองใช้ Discriminator ออกแนว Regularizer จะดีขึ้นมั้ย โดยการให้ PixelWise loss ยังคงมี Coefficient เป็น 1 เหมือนเดิม แต่ Discriminator จะเป็น 0.1, 0.2 อะไรทำนองนี้

### AutoEncoder-Experiment 02 Results
**Model Objective**
- AE + PixelWise + 0.1Dis(recon/real)
- AE + PixelWise + 0.2Dis(recon/real)
- AE + PixelWise + 0.1Dis(recon+real/real+real)
- AE + PixelWise + 0.2Dis(recon+real/real+real)

![[exp02_eval_pixelwise_mnist.png|500]]

**Discussion**
จากกราฟ พวก Objective function ที่ Coefficient ของ Discriminator term ค่อนข้างต่ำ จะมีแนวโน้มที่ loss ต่ำด้วย แสดงว่า Discriminator ตอนนี้ถ้าให้มาใช้ปรับ AE ไวไป มันอาจจะทำให้แย่กว่าเดิม 

**What's Next**
- ลองลด Coefficient ของ Discriminator ลงอีก
- หรืออาจจะลองให้ AE train ไปก่อนสักพัก จากนั้นค่อยเริ่ม train Discriminator 

## 2022-04-21 to 2022-04-22
เนื่องจากว่า ที่คิดไว้ มันไม่ตรงกัน ดังนั้นเลยขอทำ Experiment นิดหน่อยว่า Auto-encoder with Discriminator ของเรามัน work มั้ย

### AutoEncoder-Experiment Details
- Dataset: MNIST
- Model
	- **AE** + **Dis**(recon or real)
	- **AE** + **Dis**(recon+real or real+real)
	- **AE** + **Dis**(latent or Normal Dist)
	- **AE** + **Dis**(All input type) + $\alpha \cdot$**PixelWise**
- Metrics: Pixel wise loss 2(MSE)


## 2022-04-20
อ่านไปมา เริ่มสงสัยว่า Adversarial Auto-encoder จริงๆแล้ว มันเหมือนกับที่เราคิดไว้มั้ย เลยลองไปหา paper กับพวกอธิบายในบทความไว้ ปรากฏว่าไม่ใช่
> [!Actual  adversarial Auto-encoder]
> ไม่ได้มีการเอา Discriminator มาตัดสินว่าภาพไหน decode มาหรือภาพไหนจริง แต่สิ่งที่เอาไปใช้คือ *เอา Discriminator ไปเทียบ Distribution ของ latent กับ Distribution ที่เรากับหนด*  เช่น
> - เอา latent ไปให้ Discriminator ทายว่า <ins>latent นี้ใช่มาจาก Normal Distribution หรือไม่?</ins>

## 2022-04-17
วันนี้ลองคิดหาวิธีที่จะทำยังไงให้แตกต่าง เลยได้ไอเดียมาว่า ถ้าเกิดใช้ Auto-encoder แล้วเอา decoded image กับ real image ไปให้ discriminator ตัดสินหล่ะ เลยลองหาใน Google Scholar ปรากฏว่า *ไม่เจอว่ามีใครใช้ Adversarial Auto-encoder มาใน field นี้เลย* เลยคิดว่าจะลองอ่าน paper ของ [Changchong et al.](ADC_SSL.md) ดูก่อน

##  2022-04-16
ในวันนี้ได้คิดวิธีการประยุกต์ [ACMR](papers/ACMR/ACMR-bokun_wang) เข้ากับ Lips Reading ได้แต่พบปัญหาคือ มีงานที่ทำอยู่แล้ว แม้จะไม่ได้ cite ACMR ก็ตาม ซึ่งงานนั้นคือ
- [**Cross-modal Self-Supervised Learning for Lip Reading: When Contrastive Learning meets Adversarial Training** by Changchong Sheng et al.](ADC_SSL.md)