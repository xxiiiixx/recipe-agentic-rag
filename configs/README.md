# configs 约定

- `base.yaml`：完整默认配置（本地/冒烟；Milvus + bge-small）。
- `server.yaml`：服务器差异，`extends: base.yaml`，只写改动的字段。
- `ablation/*.yaml`（待建）：每份 `extends: ../base.yaml`，只改一个变量，文件名即实验名。
- 路径相对当前 YAML 所在目录解析；子配置覆盖父配置，字典递归合并，列表整体替换。
- 密钥只放 `.env`，不进 YAML；模型 ID、维度、collection 等可复现配置放 YAML。
