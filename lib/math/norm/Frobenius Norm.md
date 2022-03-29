# Frobenius Norm
[==Original  Source==](https://mathworld.wolfram.com/FrobeniusNorm.html)

คือการเอาแต่ละ element ใน matrix มายกกำลัง 2 แล้วหาผลรวม (Summation) จากนั้นถอด square root (Frobenius Norm เรียกอีกอย่างว่า เป็นการทำ Norm 2 ของทุก element ใน matrix)

- กำหนด Matrix $A$ มีขนาด $(m \times n)$
$$
\begin{flalign}
||A||_F = \sqrt{\sum_{i=1}^{m}\sum_{j=1}^{n}|a_{ij}|^2}
\end{flalign}
$$
