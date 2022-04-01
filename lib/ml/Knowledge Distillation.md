# Knowledge Distillation
[==Original Source== by Neptune AI](https://neptune.ai/blog/knowledge-distillation)
[==Alternate Source== by Prakhar Ganesh](https://towardsdatascience.com/knowledge-distillation-simplified-dd4973dbc764)

#ModelCompression #TransferLearning

คือการทำ Model Compression เพื่อลดขนาดของ Weight model ของพวก state-of-the-art model เนื่องจากว่ามันมีปัญหาขนาดใหญ่เกินไป จึงต้องทำการ compress มันลงเพื่อที่จะไป deploy ลงพวกอุปกรณ์ Edge Computing 

### ไอเดียโดยคร่าว
- ให้ model $M$ เป็น model ที่มี layer และ weight จำนวนมาก (Uncompressed Model) เป็น **Teacher Model**
- และให้ Model ที่ยังไม่ได้เรียนรู้และลดจำนวน layer กับ parameters แล้ว $\hat{M}$  (Compressed Model) มาเป็น **Student Model**
- จากนั้นเราจะมี **Distillation Loss** เป็น Function เพื่อให้ Student Model ได้ลอกเลียนพฤติกรรมของ Teacher Model 
	- Student Model จะทำการลอกเลียนพฤติกรรมของ Teacher Model ในหลายๆระดับตั้งแต่ Feature ของแต่ละ Layer จนถึง Output Layer

### อธิบายแบบละเอียดขึ้น
Knowledge Distillation มี 4 ประเด็นหลักที่ต้องพูดถึง

#### Knowledge
หรือพูดอีกอย่างง่ายคือ Weight ของ Teacher Model แต่ถ้าเอาเป๊ะๆหน่อย จะเป็นพฤติกรรมของแต่ละส่วนของ Teacher ซึ่ง Student จะเรียนรู้พฤติกรรมเหล่านั้นมาเป็น Knowledge ให้ตัวมันเอง ซึ่งจะถูกแบ่งย่อยออกไปอีก 3 ส่วน
- **Responded-Based Knowledge**  
  
  ![[KD_Responsed_knowledge.png]]
  
  หรือก็คือ Knowledge ที่ได้จากการสังเกต Output layer ของ Teacher วิธีการ Distilling ด้วย Knowledge ส่วนนี้คือคิด Loss ของ Output layer ระหว่าง Teacher กับ Student แล้ว minimize loss นี้ลงด้วยการปรับ Student

- **Feature-based Knowledge** 
  
  ![[KD_feature_knowledge.png]]
  
  เป็น Knowledge ที่ได้จากการสังเกตพฤติกรรมของ layer ด้านใน Teacher (Intermediate layers) วีธี Distilling คือ minimize loss ของ Feature activation (Intermediate layers output) ระหว่าง Teacher และ Student

- **Relation-based Knowledge**
  
  ![[KD_relation_knowledge.png]]
  
  ทีนี้ 2 Knowledge ก่อนหน้า อาจจะไม่เพียงพอ เลยมีไอเดียว่า ลองหา Relation ของพวก Output layer หรือ Feature activation จาก Teacher ดูหน่อย ซึ่ง Relation พวกนี้อาจจะบันทึกในรูปแบบ PDF, Graphs, Similarity matrix, Feature maps และอื่นๆ จากนั้นเอา Relation ของในตัว Teacher เอง กับ ของในตัว Student เองมาคิด Loss แล้ว minimize loss เพื่อปรับ Student ต่อไป

#### รูปแบบการ Train
หลักๆแล้ว มีอยู่ 3 แบบ
1. **Offline distillation** 
   - คือการที่ Teacher ได้ pre-trained มาก่อนแล้ว 
   - เป็นแบบที่เจอได้ค่อนข้างบ่อย
2. **Online distillation** 
	- Teacher จะ train ไปพร้อมๆกับ distillation Student 
	- แบบ Online จะเอามาแก้ปัญหากรณีที่หา pre-trained Teacher ไม่ได้ หรือขี้เกียจรอ train Teacher เพราะมันใหญ่จั๊ด
	- ปกติแล้ว นิยม run แบบ parallel เพื่อลดเวลาในการ train
3. **Self-distillation** 
	- Student ก็คือตัว Teacher เอง 
	- ที่ทำแบบนี้เชื่อว่า Knowledge จาก epoch เก่าๆ อาจจะช่วยให้ตัว Student มันดีขึ้นไปได้อีก 
	- หรืออาจจะเอา Knowledge จาก Intermediate layer ลึกๆ มา train layer ต้นๆ (shallow layer) ก็อาจจะช่วยให้โมเดลดีขึ้นได้เหมือนกัน

#### Student Model Arch
น่าจะเป็นประเด็นที่เวียนหัวที่สุดแล้ว เพราะมันไม่มีอะไรตายตัวในการออกแบบตัว Student แต่โดยปกติแล้ว จะนิยมอยู่ 5 รูปแบบ
1. ลดจำนวน layer และ neuron แต่ละ layer ลง
2. a quantized version of the teacher model (คือไรวะ??)
3. a smaller network with efficient basic operations
4. a smaller networks with optimized global network architecture
5. เหมือน Teacher

แต่ถ้าออกแบบไม่ได้จริง เหมือนจะมีตัวช่วยอยู่บ้าง ยกตัวอย่างเช่น <ins>Neural architecture search</ins> << มันจะช่วยหา architecture ที่เหมาะสมให้ เมื่อเราใส่ Input เป็น Teacher's architecture เข้าไป

#### Distillation Algorithm
มันก็จะมีอีกหลายวิธีที่จะทำให้ Student ได้เรียนรู้พฤติกรรมของ Teacher ได้
- **Adversarial Distillation**
  จะมีการใช้ concept ของ GANs เข้ามาประยุกต์ 
  - ใช้ Generator ในการทำ Data augmentation 
  - ใช้ Discriminator ในการแยกแยะว่า Output data หรือ Feature activation ตัวนี้มาจาก Teacher หรือ Student แล้วก็ให้ Student ปรับตัวเองไปเรื่อยๆจน Discriminator แยกไม่ออก

- **Multi-Teacher Distillation**
  
  ![[KD_multi_teacher.png]]
  
  - เอา Output ของหลายๆ Teacher มาเฉลี่ยกัน แล้ว Minimize loss ระหว่าง Student กับค่าเฉลี่ย Teacher
  - กำลังคิดว่า เส้น Knowledge Transfer อาจจะมีอีกหลายวิธีนอกจากเฉลี่ย

- **Cross-modal Distillation**
  
  ![[KD_cross_modal.png]]
  
  - อันนี้ก็น่าจะตามภาพเลย คือข้าม Modal ไปเลย

- **Born-Again Distillation**
  อยากได้รายละเอียดเพิ่งเติม ลองเข้าไปอ่าน *External PDF Source* Slide หน้าที่ 7 ดูน้ะ มันพูดถึงอะไรอย่างอื่นหลายอย่างมาก
  
  [==External PDF Source==](https://www.csc.kth.se/cvap/cvg/rg/materials/christos_001_slides.pdf)
  
  จาก *External PDF Source* สรุปได้ดังนี้
  - **ไอเดีย:** 
	  - Student มี parameters ที่เหมือนกับ Teacher
	  - Student outperform Teacher
  
  