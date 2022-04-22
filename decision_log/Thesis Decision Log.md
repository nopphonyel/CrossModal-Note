# Decision Log
จัดเรียงลำดับตาม Recent

## 2022-04-21 to 2022-04-22
เนื่องจากว่า ที่คิดไว้ มันไม่ตรงกัน ดังนั้นเลยขอทำ Experiment นิดหน่อยว่า Auto-encoder with Discriminator ของเรามัน work มั้ย

Experiment นี้จะใช้ Dataset MNIST เป็นหลัก

Model | Some Metrics
--- | ---
**AE** + **Dis**(recon or real) | ??
**AE** + **Dis**(recon+real or real+real) | ??
**AE** + **Dis**(latent or Normal Dist) | ??


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