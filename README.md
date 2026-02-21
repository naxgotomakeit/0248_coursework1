# 0248_coursework1
## Project Structure

```text
project_<studentno>_<surname>/
├── dataset/
│   └── <studentno>_<surname>/      # 只放你自己录制+标注的数据
├── src/
│   ├── dataloader.py               # 数据读取 + 预处理
│   ├── model.py                    # 检测+分割+分类模型 (torch.nn.Module)
│   ├── train.py                    # 训练循环
│   ├── evaluate.py                 # 指标 (det/seg/cls)
│   ├── visualise.py                # (可选) 可视化：mask/box/confusion matrix
│   └── utils.py                    # 工具函数 (metrics, 可视化 helper 等)
├── weights/                        # 保存模型权重 (.pt/.pth)
├── results/                        # 预测结果、log、图、confusion matrix 等
├── requirements.txt                # 依赖
└── README.md                       # 环境 + 如何运行训练/评估
