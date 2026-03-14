# AI Reader 云部署指南

## 🚀 一键部署到 Railway（推荐）

### 第一步：注册 GitHub 账号

1. 打开 https://github.com
2. 点击 Sign Up 注册账号
3. 验证邮箱

### 第二步：创建 GitHub 仓库

1. 登录 GitHub
2. 点击右上角 **+** → **New repository**
3. 填写：
   - Repository name: `ai-reader`
   - 选择 **Public**
   - 点击 **Create repository**

### 第三步：上传代码

**方法一：网页上传（最简单）**

1. 在新建的仓库页面点击 **uploading an existing file**
2. 把以下文件全部拖进去：
   ```
   app.py
   requirements.txt
   Procfile
   railway.json
   .gitignore
   templates/
     ├── index.html
     └── login.html
   ```
3. 点击 **Commit changes**

**方法二：命令行上传**

```bash
cd 你的项目文件夹

git init
git add .
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/你的用户名/ai-reader.git
git push -u origin main
```

### 第四步：部署到 Railway

1. 打开 https://railway.app
2. 点击 **Login** → **Login with GitHub**
3. 授权 Railway 访问你的 GitHub
4. 点击 **New Project**
5. 选择 **Deploy from GitHub repo**
6. 选择 `ai-reader` 仓库
7. 等待部署完成（约 2-3 分钟）

### 第五步：获取访问地址

1. 部署完成后，点击你的项目
2. 点击 **Settings** → **Networking**
3. 点击 **Generate Domain**
4. 复制生成的地址，如 `ai-reader-xxx.up.railway.app`

### 第六步：访问你的应用

在浏览器或 iPad 上打开生成的地址，注册账号即可使用！

---

## 🔧 环境变量配置（可选）

在 Railway 项目的 **Variables** 中可以添加：

| 变量名 | 说明 | 示例 |
|--------|------|------|
| `SECRET_KEY` | Session 加密密钥 | 随机字符串 |
| `OPENAI_API_KEY` | OpenAI API 密钥（支持 ChatGPT、GPT-4、Codex） | sk-xxx |
| `AI_MODEL` | 使用的 AI 模型（可选）| gpt-3.5-turbo 或 gpt-4 或 code-davinci-002（Codex）|

---

## 📱 iPad/手机访问

部署完成后，在任何设备的浏览器中输入你的 Railway 地址即可访问！

**添加到主屏幕（iOS）：**
1. Safari 打开网址
2. 点击分享按钮
3. 选择"添加到主屏幕"

---

## ❓ 常见问题

### Q: 部署失败怎么办？
A: 检查 Railway 的 Deployments 日志，看具体错误信息。

### Q: 免费额度用完怎么办？
A: Railway 每月免费 $5 额度，超出后可以：
   - 升级付费（$5/月起）
   - 换用 Render.com（也有免费额度）

### Q: 数据会丢失吗？
A: Railway 的免费版本数据可能会丢失。如需持久化存储，建议：
   - 升级 Railway 付费版
   - 或使用外部数据库（如 Supabase）

---

## 🔄 更新应用

修改代码后，只需：

```bash
git add .
git commit -m "update"
git push
```

Railway 会自动重新部署！
