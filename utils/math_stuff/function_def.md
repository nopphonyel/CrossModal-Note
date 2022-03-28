# Normalization
## Frobenius Norm
[==Original  Source==](https://mathworld.wolfram.com/FrobeniusNorm.html)

คือการเอาแต่ละ element ใน matrix มายกกำลัง 2 แล้วหาผลรวม (Summation) จากนั้นถอด square root (Frobenius Norm เรียกอีกอย่างว่า เป็นการทำ Norm 2 ของทุก element ใน matrix)

- กำหนด Matrix $A$ มีขนาด $(m \times n)$
$$
\begin{flalign}
||A||_F = \sqrt{\sum_{i=1}^{m}\sum_{j=1}^{n}|a_{ij}|^2}
\end{flalign}
$$

# Distance Calculation
## Cosine Similarity
[==Original Source==](https://en.wikipedia.org/wiki/Cosine_similarity)

เป็นสูตรหา Distance ระหว่าง Vector 2 ตัว โดยที่ทั้งคู่ต้องมีขนาดเท่ากัน ซึ่งมีนิยามดังนี้

- กำหนด Vector $A,B$ มีขนาด $(n)$
$$
\begin{flalign}
\DeclareMathOperator*{\cosinesim}{cosine similarity}
\cosinesim &= S_c(A,B) = \frac{A\cdot B}{||A||\ ||B||}
		\\
		\\&= \frac{\sum^n_{i=1}A_iB_i}{\sqrt{\sum^n_{i=1}A_i^2}\sqrt{\sum^n_{i=1}B_i^2}}
\end{flalign}
$$


# Metrics Calculation
## Mean Average Precision