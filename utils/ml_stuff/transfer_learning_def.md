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

### องค์ประกอบ
Knowledge Distillation มี 3 องค์ประกอบหลัก

#### Knowledge
หรือพูดอีกอย่างคือ Weight ของ Teacher Model ซึ่งจะถูกแบ่งย่อยออกไปอีก 3 ส่วน
- **Responded-Based Knowledge**  
  
  ![[KD_Responsed_knowledge.png]]
  
  หรือก็คือ Knowledge ที่ได้จากการสังเกต Output layer โดยวิธีการนำ Knowledge ส่วนนี้ไปคิด Loss ระหว่าง Student Model กับ Teacher Model

- **Feature-based Knowledge** 
  เป็น Knowledge ที่ได้จากการสังเกตพฤติกรรมของ Feature ใน Teacher Model
#### Distillation Algorithm
#### Model Arch