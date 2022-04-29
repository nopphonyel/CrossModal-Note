# Meeting Log

## Meeting on 2022-04-29
### สถานการณ์ตั้งต้น
- ไล่อ่าน Decision Log จนถึงวันล่าสุด
- แนวทาง(ที่จะทำจริงๆ)ที่คิดไว้
	- based on hypo: Discriminator could improve the cross modal performance
		- Investigation of discriminator effect on lips reading
		- เอา Model จากงาน [**Towards Practical Lipreading with Distilled and Efficient Models** by Pingchuan Ma et al.](lips_read_tcn-pingchuan_ma.md), [**Learn an Effective Lip Reading Model Without Pains** by Dalu Feng et al.](papers/EffectiveLipReadWithNoPain/EffectiveLipReadWithNoPain) ไปเปลี่ยนเป็น AAE (Contribution จะน้อยไปมั้ยน้ะ)
		- เปลี่ยน Model ของงาน [**Cross-modal Self-Supervised Learning for Lip Reading: When Contrastive Learning meets Adversarial Training** by Changchong Sheng et al.](ADC_SSL.md) มาเป็น AAE
	- based on hypo: Adversarial could not always improve the performance of A-V cross modal retrieval
		- Compare the performance of many adversarial technique (GANs stuff also included) ก็อาจจะเอาพวก GANs, CGANs, CycleGANs, AAE, บลาๆ มาเปรียบเทียบกัน เพราะว่ายังไม่เห็นงานไหนที่เปรียบเทียบหลายๆ Adversarial Technique อย่างจริงจัง 
			- Experiment Design อาจจะต้องขอเวลาในการออกแบบเพิ่ม เพราะยังไม่แน่ใจว่าจะเอาอะไรเป็น Baseline บ้าง
### Short-note จากการคุยกัน