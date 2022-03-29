# Cosine Similarity
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