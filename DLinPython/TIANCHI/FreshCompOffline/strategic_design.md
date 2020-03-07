# STRATEGIC DESIGN

02-28

- 根据用户行为数据对用户, 品类和商品建立三部图(Tripartite Graph), 然后使用图神经网络学习商品节点嵌入.

03-01

- Propositions:
  - 同一位用户, 对一类商品的查看量, 收藏量, 加购量越大, 购买该类商品的可能性越高;
  - 同类商品中, 查看量, 收藏量, 加购量, 购买量越大,被购买的可能性越高.

- Strategies:
  - 可以基于行为时间和类别划分训练集和验证集, 最后根据购买行为作为测试集评估模型;
  - 首先构建模型预测用户对某一品类商品的购买行为, 然后对该类商品进行排序, 挑选排名前十作为预测值;
  - 模型: 用户-品类预测 & 品类-商品排序.

- Features:
  - 对用户行为进行one-hot编码, 特征维度(N = Behavior Type = 4), 标签数(N = Category Size = 1054);

03-03

- Target to predict: Buy or Not?
  - Feature: history of a certain category, including view, favorite, add2cart;
  - Label: given a certain category, buy or not?

- Feature Engineering:
  - Extract history of a 'bought' categroy as positive samples;
  - Extract the rest of the history a certain user as negative samples.

03-05

- 使用groupby根据用户ID和品类ID对数据集进行分组;
- 将每个分组重组为一行特征，然后构建数据集;

03-06

- 由于正样本相对负样本来说较少，需要进行数据均衡;
- 根据正样本数随机抽取负样本;
