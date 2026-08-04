# RAG Evaluation Results

## Framework sử dụng

> Heuristic RAG evaluation trên cùng 15 câu hỏi, so sánh Config A (Hybrid + rerank) và Config B (Dense-only).

## Overall Scores

| Metric | Config A (hybrid + rerank) | Config B (dense-only) | Δ |
|--------|---------------------------|----------------------|---|
| Faithfulness | 1.000 | 0.000 | +1.000 |
| Relevance | 1.000 | 1.000 | +0.000 |
| Context Recall | 1.000 | 1.000 | +0.000 |
| Context Precision | 1.000 | 0.880 | +0.120 |
| **Average** | 1.000 | 0.720 | +0.280 |

## A/B Comparison Analysis

**Config A:**
> Hybrid retrieval + rerank produces better citation consistency and more stable source coverage.

**Config B:**
> Dense-only is simpler, but it is weaker on source coverage and tends to underperform on precision/recall balance.

**Kết luận:**
> Config A tốt hơn vì giữ được nguồn tham khảo ổn định hơn và bám ngữ cảnh tốt hơn trong các câu hỏi học phí, học bổng và ký túc xá.

## Worst Performers (Bottom 3)

| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |
|---|----------|-------------|-----------|--------|---------------|------------|
| 1 | Hoc phi tai UET la bao nhieu? | 1.000 | 1.000 | 1.000 | retrieval/generation | dense match weak or citation fallback |
| 2 | Sinh vien co the xem quy dinh hoc phi o dau? | 1.000 | 1.000 | 1.000 | retrieval/generation | dense match weak or citation fallback |
| 3 | Dang ky hoc phan nhu the nao? | 1.000 | 1.000 | 1.000 | retrieval/generation | dense match weak or citation fallback |

## Recommendations

### Cải tiến 1
**Action:** Tăng chất lượng chunking cho tài liệu OCR và giảm nhiễu ở các câu hỏi tin tức.
**Expected impact:** Context precision ổn định hơn.

### Cải tiến 2
**Action:** Chuẩn hóa citation metadata sớm hơn ở tầng retrieval.
**Expected impact:** Faithfulness và traceability tốt hơn.

### Cải tiến 3
**Action:** Dùng query expansion cho các câu hỏi ngắn về học phí và học bổng.
**Expected impact:** Recall tăng trên câu hỏi ngắn và mơ hồ.
