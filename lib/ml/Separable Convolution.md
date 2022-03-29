# Separable Convolution
[==Original Source==](https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728)

#ModelCompression

Convolution แบบนี้จะช่วยลด computation complexity ลง เนื่องจากจำนวน parameter ที่ลดลง ซึ่งจะประกอบด้วย 2 ขั้นตอนหลักๆคือ Depthwise Convolution และ Pointwise Convolution
## Depthwise Convolution
**นิยามสไตล์ Mathematical**
- กำหนดให้ Input tensor $I$ มี dimension เป็น $(C\times H\times W)$ 
- เราจะใช้ set ของ Kernel tensor $K=\{K_0, K_1, ..., K_C\}$ ในการทำ Convolution แยกแต่ละ Channel $C$ ของ $I$  โดยที่ Dimension ของ $K_i$ คือ $(J \times K)$ โดยที่ $J \leq H ; K\leq W$

![[depthwise_conv.png]]

จากตัวอย่าง Kernel Tensor $K$ มีขนาด $(5\times 5)$ และมีจำนวน 3 แผ่นเท่ากับ Channel ของ Input Tensor $I$ แล้วทำ Convolution แยกแต่ละ Channel กัน

## Pointwise Convolution
**นิยามสไตล์ Mathematical**
- กำหนดให้ Input tensor $I$ มี dimension เป็น $(C\times H\times W)$ 
- กำหนดให้ $\hat{C}$ คือจำนวน Output channel
- เราจะใช้ set ของ Kernel tensor $K=\{K_0, K_1, ..., K_{\hat{C}}\}$  ในการทำ Convolution โดยที่ $K_i$ มี Dimension เป็น $(C \times 1 \times 1)$  จากนั้นทำ Convolution ตามปกติ

![[pointwise_conv.png]]

จากรูปตัวอย่างคือการทำ Pointwise Convolution โดยที่ Output channel $\hat{C} = 256$  และ Input tensor $I$ มีขนาด $(3\times8\times8)$... หากสังเกตให้ดี จะเห็นว่าค่า $H,W$ ของ Output tensor และ Input tensor $I$ <ins>ไม่เปลี่ยนแปลง</ins>