# Evaluation

该目录用于管理评测与检索验证：

- `datasets/`：评测集输入。
- `reports/`：评测结果输出。
- `retrieval_smoke_test.py`：检索冒烟测试脚本。

示例：

```bash
python evaluation/retrieval_smoke_test.py \
  --query "乘法器时序优化有哪些方法？" \
  --data-dir data \
  --chroma-path chroma_db \
  --embedding-model BAAI/bge-m3 \
  --top-k 3
```
