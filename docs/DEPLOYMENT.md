# 部署指南

## 方案 A：部署到 Streamlit Cloud（推荐 · 5 分钟）

### 步骤 1：在 GitHub 创建仓库（2 分钟）

```
1. 浏览器访问 github.com
2. 登录（如果没有账号先注册，免费）
3. 右上角「+」→「New repository」
4. 填写：
   - Repository name: civil-engineering-dashboard
   - Description: 中国工业 PPI 跨行业分析平台
   - Public（公开）+ 不勾选 Add README
5. 点「Create repository」
```

### 步骤 2：本地初始化 Git + Push 代码（3 分钟）

打开终端，执行：

```bash
# 1. 切到项目目录
cd ~/Desktop/Claude\ code/civil-engineering-dashboard

# 2. 初始化 Git
git init

# 3. 配置身份（如没配置过）
git config user.name "你的 GitHub 用户名"
git config user.email "你的邮箱"

# 4. 添加所有文件
git add .

# 5. 第一次提交
git commit -m "feat: 初始化项目 · 4 行业 PPI 跨年度分析平台"

# 6. 添加远程仓库（替换成你的 GitHub 用户名）
git remote add origin https://github.com/你的用户名/civil-engineering-dashboard.git

# 7. 推送
git branch -M main
git push -u origin main
```

如果推送时弹出 GitHub 登录窗口，输入用户名 + Personal Access Token（不是密码）。

### 步骤 3：在 Streamlit Cloud 部署（5 分钟）

```
1. 浏览器访问 share.streamlit.io
2. 点「Sign in with GitHub」登录
3. 点「New app」
4. 填写：
   - Repository: 你的用户名/civil-engineering-dashboard
   - Branch: main
   - Main file path: app/streamlit_app.py
5. 点「Deploy」
6. 等待 2-5 分钟，部署完成
7. 复制应用 URL（类似 https://xxx.streamlit.app）
```

### 步骤 4：测试部署

打开应用 URL，应该能看到：
- 标题：中国工业 PPI 跨行业分析平台
- 5 个 Tab（趋势 / 相关性 / 同比 / 预测 / 数据说明）
- 4 个行业的数据可视化

---

## 方案 B：本地部署（不推荐 · 仅用于本地测试）

```bash
cd ~/Desktop/Claude\ code/civil-engineering-dashboard
pip3 install streamlit
streamlit run app/streamlit_app.py
```

浏览器会自动打开 http://localhost:8501

但这只是本地访问，校招面试官看不到。

---

## 方案 C：临时公网演示（用 ngrok · 1 分钟）

如果暂时不想建 GitHub 仓库，可以用 ngrok 临时暴露本地端口：

```bash
# 1. 安装 ngrok
brew install ngrok

# 2. 注册 ngrok 账号（免费）
# 访问 ngrok.com 注册 + 复制 authtoken

# 3. 配置 ngrok
ngrok config add-authtoken 你的token

# 4. 启动 streamlit
streamlit run app/streamlit_app.py &

# 5. 暴露到公网
ngrok http 8501
```

会得到一个公网 URL（免费版 8 小时有效）。

---

## 推荐路径

**秋招简历项目**：方案 A（部署到 Streamlit Cloud）
**本地测试**：方案 B
**临时演示**：方案 C（如果 GitHub 建仓失败）

---

## 部署后简历链接填写

```
GitHub: https://github.com/你的用户名/civil-engineering-dashboard
Demo: https://你的应用名.streamlit.app
```

---

## 部署后检查清单

- [ ] GitHub 仓库是 public（公开）
- [ ] README 显示正确
- [ ] Streamlit 应用 URL 可访问
- [ ] 4 个 Tab 都正常显示
- [ ] 中文显示正常（如果乱码，告诉我）
- [ ] 4 个行业数据都加载到

---

## 常见问题

**Q：GitHub push 时认证失败？**
A：用 Personal Access Token 代替密码。GitHub → Settings → Developer settings → Personal access tokens → Generate new token。权限选 repo。

**Q：Streamlit Cloud 部署失败？**
A：检查 requirements.txt 是否齐全。看错误日志（Deploy 页面有 Logs 标签）。

**Q：中文显示乱码？**
A：matplotlib 用 PNG 时会乱码（Plotly 默认正常）。如果出现，告诉我加中文字体配置。

---

## 部署完成后告诉我

1. GitHub 仓库 URL
2. Streamlit Cloud 应用 URL
3. 是否正常显示

我帮你：
- 完善 README
- 写面试讲稿
- 准备校招简历投递清单