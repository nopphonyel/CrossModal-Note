# Normalization
## Frobenius Norm
คือการเอาแต่ละ element ใน matrix มายกกำลัง 2 แล้วหาผลรวม (Summation) จากนั้นถอด square root (Frobenius Norm เรียกอีกอย่างว่า เป็นการทำ Norm 2 ของทุก element ใน matrix)

# Distance Calculation
## Cosine Similarity
เป็นสูตรหา Distance ระหว่าง Vector 2 อย่าง ซึ่งมีนิยามดังนี้... กำหนด Vector $A,B$
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