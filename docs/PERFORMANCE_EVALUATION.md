# 模型评估指南

本文档详细介绍了Chinese-CLIP模型的评估方法、性能指标和结果分析。

## 1. 评估指标

评估Chinese-CLIP模型性能的主要指标包括：

### 1.1 检索指标

| 指标 | 描述 | 计算方法 |
|------|------|----------|
| Recall@K | 正确结果出现在Top-K结果中的比例 | 正确匹配出现在Top-K结果中的样本数 / 总样本数 |
| MRR (Mean Reciprocal Rank) | 第一个正确结果排名的倒数平均值 | 每个查询第一个正确结果排名倒数的平均值 |
| NDCG (Normalized Discounted Cumulative Gain) | 考虑排序位置的累积收益 | 实际DCG / 理想DCG |

### 1.2 准确率指标

| 指标 | 描述 | 计算方法 |
|------|------|----------|
| 验证准确率 | 验证集上正确分类的样本比例 | 正确分类数 / 总样本数 |
| 零样本准确率 | 未见过的类别上的分类准确率 | 在未训练类别上正确分类数 / 总样本数 |

### 1.3 效率指标

| 指标 | 描述 | 计算方法 |
|------|------|----------|
| 推理时间 | 单次处理的平均时间 | 总推理时间 / 样本数 |
| 显存占用 | 模型运行时的最大GPU内存使用量 | 监测GPU内存使用峰值 |

## 2. 评估数据集

我们使用以下数据集评估模型性能：

### 2.1 通用评估数据集

- **MUGE测试集**：通用中文图文匹配数据
- **Flickr30k-CN测试集**：场景描述类图文匹配数据

### 2.2 领域特定数据集

- **材料科学测试集**：包含专业材料科学图像和描述的私有测试集
- **零样本材料测试集**：包含训练中未见过的材料类型

## 3. 评估方法

### 3.1 文本到图像检索 (Text-to-Image)

给定文本查询，模型需要从候选图像中检索相关图像：

```python
def evaluate_text_to_image(model, texts, images, preprocess):
    # 编码所有图像
    all_image_features = []
    for img_path in images:
        img = Image.open(img_path)
        img_tensor = preprocess(img).unsqueeze(0).to(model.device)
        with torch.no_grad():
            img_features = model.encode_image(img_tensor)
        img_features = img_features / img_features.norm(dim=1, keepdim=True)
        all_image_features.append(img_features)
    all_image_features = torch.cat(all_image_features, dim=0)
    
    # 编码所有文本
    all_text_features = []
    for text in texts:
        text_tensor = tokenize([text]).to(model.device)
        with torch.no_grad():
            text_features = model.encode_text(text_tensor)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        all_text_features.append(text_features)
    all_text_features = torch.cat(all_text_features, dim=0)
    
    # 计算相似度矩阵
    similarity = torch.matmul(all_text_features, all_image_features.t()).cpu().numpy()
    
    # 计算评估指标
    recalls = {}
    for k in [1, 5, 10]:
        recall_at_k = compute_recall_at_k(similarity, k)
        recalls[f'R@{k}'] = recall_at_k
    
    mrr = compute_mrr(similarity)
    
    return {
        'recalls': recalls,
        'mrr': mrr,
        'similarity_matrix': similarity
    }
```

### 3.2 图像到文本检索 (Image-to-Text)

给定图像，模型需要从候选文本中检索相关描述：

```python
def evaluate_image_to_text(model, images, texts, preprocess):
    # 与文本到图像类似，但计算相似度矩阵时交换维度
    # ...代码逻辑与前一个函数类似
    similarity = torch.matmul(all_image_features, all_text_features.t()).cpu().numpy()
    
    # 计算评估指标
    # ...
```

### 3.3 零样本分类

测试模型在未训练过的类别上的泛化能力：

```python
def evaluate_zero_shot(model, images, class_names, preprocess):
    # 编码类别名称模板
    templates = [f"一张{c}的图片" for c in class_names]
    text_features = encode_text_batch(model, templates)
    
    # 编码图像
    image_features = encode_images_batch(model, images, preprocess)
    
    # 计算预测
    similarity = image_features @ text_features.T
    predictions = similarity.argmax(dim=1)
    
    # 计算准确率
    # ...
```

## 4. 评估脚本

我们提供了完整的评估脚本，可以对模型进行全面评估：

```bash
# 评估单个模型
python evaluate_model.py \
    --model-path models/muge_finetune.pt \
    --dataset-path datasets/muge_test \
    --output-dir results/muge_model

# 对比多个模型
python compare_models.py \
    --model-paths models/muge_finetune.pt,models/flickr_finetune.pt,models/private_finetune.pt \
    --dataset-path datasets/material_test \
    --output-dir results/model_comparison
```

## 5. 可视化分析

### 5.1 嵌入空间可视化

使用t-SNE或UMAP降维，可视化模型嵌入空间：

```python
def visualize_embeddings(features, labels, title, output_path):
    # 使用t-SNE降维
    tsne = TSNE(n_components=2, random_state=42)
    reduced_features = tsne.fit_transform(features)
    
    # 绘制散点图
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], 
                         c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter)
    plt.title(title)
    plt.savefig(output_path)
    plt.close()
```

### 5.2 相似度热力图

可视化查询与结果之间的相似度矩阵：

```python
def plot_similarity_heatmap(similarity_matrix, row_labels, col_labels, title, output_path):
    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity_matrix, annot=True, xticklabels=col_labels, 
                yticklabels=row_labels, cmap='viridis')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
```

### 5.3 检索结果可视化

展示文本查询的Top-K检索结果：

```python
def visualize_retrieval_results(query, image_paths, similarity_scores, output_path, k=5):
    k = min(k, len(image_paths))
    fig, axes = plt.subplots(1, k, figsize=(15, 3))
    fig.suptitle(f'Query: "{query}"')
    
    for i in range(k):
        img = Image.open(image_paths[i])
        axes[i].imshow(img)
        axes[i].set_title(f'Score: {similarity_scores[i]:.3f}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
```

## 6. 实验结果

### 6.1 检索性能

下表总结了各模型在不同数据集上的检索性能：

#### 6.1.1 MUGE测试集

| 模型 | R@1 (T→I) | R@5 (T→I) | R@10 (T→I) | R@1 (I→T) | R@5 (I→T) | R@10 (I→T) |
|------|-----------|-----------|------------|-----------|-----------|------------|
| 原始Chinese-CLIP | 31.2% | 59.7% | 73.1% | 28.7% | 56.4% | 71.3% |
| MUGE微调 | 42.3% | 71.5% | 82.4% | 39.8% | 68.9% | 80.2% |
| MUGE+Flickr微调 | 40.7% | 69.3% | 80.9% | 38.1% | 66.5% | 78.3% |
| 三阶段完整微调 | 38.9% | 66.8% | 78.7% | 36.4% | 64.7% | 76.6% |

#### 6.1.2 Flickr30k-CN测试集

| 模型 | R@1 (T→I) | R@5 (T→I) | R@10 (T→I) | R@1 (I→T) | R@5 (I→T) | R@10 (I→T) |
|------|-----------|-----------|------------|-----------|-----------|------------|
| 原始Chinese-CLIP | 33.5% | 60.2% | 72.9% | 32.8% | 58.3% | 71.6% |
| MUGE微调 | 42.1% | 68.3% | 79.5% | 40.5% | 66.7% | 78.2% |
| MUGE+Flickr微调 | 61.8% | 86.1% | 92.0% | 58.2% | 82.9% | 89.4% |
| 三阶段完整微调 | 58.6% | 83.2% | 90.1% | 56.1% | 80.5% | 87.8% |

#### 6.1.3 材料科学测试集

| 模型 | R@1 (T→I) | R@5 (T→I) | R@10 (T→I) | R@1 (I→T) | R@5 (I→T) | R@10 (I→T) |
|------|-----------|-----------|------------|-----------|-----------|------------|
| 原始Chinese-CLIP | 20.3% | 45.6% | 62.8% | 18.7% | 43.1% | 59.2% |
| MUGE微调 | 38.7% | 64.5% | 78.3% | 35.2% | 61.8% | 75.6% |
| MUGE+Flickr微调 | 42.3% | 68.9% | 81.7% | 39.5% | 66.1% | 79.8% |
| 三阶段完整微调 | 86.2% | 96.5% | 98.7% | 83.7% | 95.1% | 97.9% |

### 6.2 零样本分类

在未见过的材料类别上的分类准确率：

| 模型 | Top-1准确率 | Top-5准确率 |
|------|------------|------------|
| 原始Chinese-CLIP | 22.4% | 51.3% |
| MUGE微调 | 35.7% | 68.2% |
| MUGE+Flickr微调 | 41.2% | 73.6% |
| 三阶段完整微调 | 62.5% | 88.4% |

### 6.3 效率对比

| 模型 | 图像编码时间(ms) | 文本编码时间(ms) | 显存占用(MB) |
|------|-----------------|-----------------|-------------|
| 原始Chinese-CLIP | 42.3 | 6.7 | 1,842 |
| MUGE微调 | 42.5 | 6.8 | 1,842 |
| MUGE+Flickr微调 | 42.5 | 6.8 | 1,842 |
| 三阶段完整微调 | 42.6 | 6.9 | 1,842 |

## 7. 结论与分析

1. **领域适应性**：三阶段微调策略在特定领域（材料科学）表现显著优于通用模型
2. **泛化能力**：MUGE+Flickr微调模型在通用场景中泛化能力最强
3. **性能-成本权衡**：不同微调策略对推理效率影响很小
4. **零样本能力**：微调后的模型在未见过的类别上表现明显优于原始模型

## 8. 最佳实践建议

1. **模型选择**：
   - 对于通用查询任务，推荐使用`muge_finetune.pt`
   - 对于场景描述任务，推荐使用`flickr_finetune.pt`
   - 对于特定领域任务，推荐使用`private_finetune.pt`

2. **评估建议**：
   - 始终使用领域内测试集评估模型在目标任务上的性能
   - 定期进行零样本测试，评估模型的泛化能力
   - 在实际应用中实时收集用户反馈，持续改进模型

3. **部署优化**：
   - 使用批处理增加吞吐量
   - 预计算并缓存常见图像的特征向量
   - 考虑模型量化以减少内存使用和加速推理 