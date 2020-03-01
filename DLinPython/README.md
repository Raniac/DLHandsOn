# DEEP LEARNING IN PYTHON

## 1 PyTorch Geometric

### 1.1 Docker Image ```imcomking/pytorch_geometric```

```bash
docker run -it --rm -v /c/Users/Benny/Documents/Projects/DLHandsOn/DLinPython/PyTorchGeometric:/workspace imcomking/pytorch_geometric:latest /bin/bash
```

### 1.2 Graph Attention Network

- [Graph Attention Networks](https://arxiv.org/abs/1710.10903)
- [浅谈 Attention 机制的理解](https://www.cnblogs.com/ydcode/p/11038064.html)

### 1.3 Graph Isomorphism Network

- [How Powerful are Graph Neural Networks?](https://arxiv.org/abs/1810.00826)
- [Understanding Graph Isomorphism Network for Brain MR Functional Connectivity Analysis](http://cn.arxiv.org/abs/2001.03690)

### 1.4 Differentiable Pooling

- [Hierarchical Graph Representation Learning with Differentiable Pooling](https://arxiv.org/abs/1806.08804)

## 2 TIANCHI

### 2.1 [FreshCompOffline](https://tianchi.aliyun.com/competition/entrance/231522/introduction)

- [协同过滤（Collaborative Filtering）学习笔记](https://www.jianshu.com/p/d15ba37755d1)
- [推荐系统中召回策略](https://www.cnblogs.com/graybird/p/11393511.html)
- [CIKM 2019 EComm AI：超大规模推荐之用户兴趣高效检索 赛题解读及阿里深度树匹配技术实践](https://tianchi.aliyun.com/course/video?spm=5176.12586971.1001.45.4ee274a3LPTFrh&liveId=41072)
- [CIKM 2019 EComm AI：用户行为预测 赛题解读与阿里GNN推荐结合实践分享](https://tianchi.aliyun.com/course/video?spm=5176.12586971.1001.50.6fc147c4o8k4tC&liveId=41071)
- [《文章推荐系统》系列之1、推荐流程设计](http://thinkgamer.cn/2019/12/05/%E6%8E%A8%E8%8D%90%E4%B8%8E%E6%8E%92%E5%BA%8F/%E6%96%87%E7%AB%A0%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F/%E3%80%8A%E6%96%87%E7%AB%A0%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E3%80%8B%E7%B3%BB%E5%88%97%E4%B9%8B1%E3%80%81%E6%8E%A8%E8%8D%90%E6%B5%81%E7%A8%8B%E8%AE%BE%E8%AE%A1/)
- Docker Image

```bash
docker run -it --rm -v /c/Users/Benny/Documents/Projects/DLHandsOn/DLinPython/TIANCHI/FreshCompOffline:/workspace imcomking/pytorch_geometric:latest /bin/bash
```

## Appendix

- [```torch_geometric.nn.GATConv```](PyTorchGeometric/docs/GATConv.png)
- [```torch_geometric.nn.GINConv```](PyTorchGeometric/docs/GINConv.png)
- [```torch_geometric.nn.dense_diff_pool```](PyTorchGeometric/docs/dense_diff_pool.png)
