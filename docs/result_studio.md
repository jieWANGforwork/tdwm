# World Model Result Studio

只读 Streamlit 实验工作台：O25/O50/O100 的全部测试样本、真实初始/目标图片、
完整参考轨迹、方法执行/预测轨迹和逐条成功标签。方法不限数量，轨迹画面固定
224 × 224，不因少选方法而拉伸。状态/表征和 action 分析单独管理。

## 独立环境启动

在仓库根目录运行（Python 3.12）：

```sh
python3.12 -m venv .venv-result-studio
.venv-result-studio/bin/python -m pip install -r configs/result_studio/requirements.txt
export RESULT_STUDIO_MANIFEST=/absolute/path/to/manifest.json
bash scripts/run_result_studio.sh
```

打开 http://127.0.0.1:8501。可用 `RESULT_STUDIO_PORT` 修改端口，
`RESULT_STUDIO_PYTHON` 指定已有的独立展示环境。
该环境不安装训练依赖、不导入训练模块，不更改固定的 stable-worldmodel 版本。
依赖唯一声明为根 pyproject.toml 的 `dependency-groups.result-studio`；
requirements.txt 是针对 Python 3.12 生成的锁定安装清单，更新命令：

```sh
uv pip compile --group result-studio --python-version 3.12 --universal \
  -o configs/result_studio/requirements.txt
```

未指定数据清单时，默认检查仓库内
`data/result_studio/real_preview_20260930/manifest.json`。
真实数据缺失会明确显示未接入，绝不自动换成合成图片。合成演示必须主动选择。

## AutoDL 与本地浏览器

网站应在服务器使用同样的独立环境运行，并只监听 127.0.0.1。
在本机建立 SSH 转发（替换 SSH_PORT、SERVER_HOST）：

```sh
ssh -N -L 18501:127.0.0.1:8501 -p SSH_PORT root@SERVER_HOST
```

保持 SSH 连接，然后本机浏览器打开 http://127.0.0.1:18501。
18501 与本机原有的 8501 预览分开。服务器负责读取图片和计算分析；浏览器只接收
当前展示内容，不需要先下载整个数据集。不要开放未认证的公网 Streamlit 端口。
密码和 SSH 私钥不进入代码、GitHub 或数据清单。

## 数据接入与当前边界

- 示例：`configs/result_studio/manifest.example.json`。路径相对清单目录，
  不能逃出该目录；数组为不含 pickle 的数值 `[T,D]` .npy。
- identity = (offset, episode, start, goal)，不能按行号盲目对齐不同方法。
- reference 必须是 start 到 goal 的实际图像序列；初始/目标图取它的首/末帧。
- methods 下 executed 与 predicted 分开保存，成功标签必须为 boolean 或 null。
- 图像/状态/表征帧数必须一致，action 长度可为 T 或 T−1。
- 同步预览最多均匀选取 201 帧；单帧检查可读取所有帧。视频使用独立时间轴。
- 当前导入的预览是 O25/O50/O100 各 50 条真实参考片段、参考状态/action、
  f_only/f_plus_g 成功标签，共 5,138 张去重 JPEG，保留数据集中编码字节。
  JPEG100 不是无损原始像素。方法执行/预测轨迹及真实 embedding 尚未接入，
  不能用参考轨迹或合成坐标代替。
- 数据集、预览图片/视频、数组、日志、密码不提交 GitHub；本仓库只保存程序、
  格式示例、测试和文档，具体运行数据保存在服务器或个人本机。

可选的「历史评测记录」用于仅加载标签。设置 `RESULT_STUDIO_HISTORY_CONFIG`
指向 JSON（见 `configs/result_studio/history.example.json`），其中每个 offset
指向含 f_only/f_plus_g 子目录的结果目录，各目录包含 results.json 和
episode_selection.json。目录相对于该配置文件，不包含机器私有路径默认值。

`scripts/import_result_studio_preview.py --help` 提供按这些测试选择读取参考帧的
导入入口；显式传入历史配置、SSH 控制连接、服务器解释器、数据集和新输出目录。
它沿用已核验的 Cube 固定 201 帧/episode 行布局，并逐行检查 episode_idx/step_idx；
不适配任意数据集布局。只读服务器数据、流式传输并去重，不保存密码或中间压缩包，
已有输出目录会拒绝覆盖。此导入工具不运行评测，也不能补出缺失的方法 rollout。

## 扩展与验证

核心模块在 `src/tdwm/result_studio/`。`data.py` 读文件，`models.py` 定义数据，
`player.py` 播放，`endpoints.py` 展示端点，`analysis.py` 是独立纯函数，
`views.py` 渲染分析；添加分析函数并注册到 REGISTRY 即可出现在界面。

分布对有效时间步等权，所有系列统一分箱、独立归一化。PCA 联合拟合并抽样，
不是显著性检验；不同编码器的表征不可直接混合比较。

```sh
PYTHONPATH=src .venv-result-studio/bin/python -m unittest discover \
  -s tests/unit -p test_result_studio.py -v
```

测试使用合成文件 fixture，不依赖私人镜像或网络；如果配置了真实预览清单，
额外核对三组各 50 条、每条全部参考帧、224×224 JPEG 与状态/action 形状。
