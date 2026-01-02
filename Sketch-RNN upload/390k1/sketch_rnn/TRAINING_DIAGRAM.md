# TensorFlow 训练示意图

文件：`assets/training_diagram.svg`

快速说明：

- 打开（在本目录）：

```bash
xdg-open assets/training_diagram.svg    # 在 Linux 桌面环境中打开
# 或使用 VS Code 打开：
code assets/training_diagram.svg
```

- 导出 PNG（如果需要）：

```bash
# 使用 rsvg-convert（需要安装 librsvg）
rsvg-convert -w 1200 assets/training_diagram.svg -o training_diagram.png

# 或使用 inkscape:
inkscape assets/training_diagram.svg --export-filename=training_diagram.png --export-width=1200
```

说明：该 SVG 描述了一个典型的 TensorFlow 训练流程：数据加载 → 输入管线 → 模型 → 训练循环（前向、计算 loss、反向、更新）→ 检查点与日志（TensorBoard）→ 可选采样/生成。

如果你希望将图改为更具体的实现（例如加入 learning-rate scheduler、分布式训练、混合精度、梯度累积等），告诉我我可以直接替你修改 SVG 或用 Graphviz/Diagrams 生成更复杂的图。
