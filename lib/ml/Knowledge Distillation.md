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
1. **Offline distillation** คือการที่ Teacher ได้ Pre-trained มาก่อนแล้ว
2. **Online distillation** Teacher จะ train ไปพร้อมๆกับ distillation Student
3. **Self-distillation** Student ก็คือ Teacher เอง

#### Model Arch

#### Distillation Algorithm