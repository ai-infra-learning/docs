# AI 读书会文档项目

## About this project

- 这是一个 AI 读书会网站，基于 [Mintlify](https://mintlify.com) 构建
- 每本书对应一个独立的 tab 和 `books/` 下的一个子目录
- 页面使用 MDX 格式，带 YAML frontmatter
- 配置文件为 `docs.json`
- 运行 `mint dev` 本地预览
- 运行 `mint broken-links` 检查链接

## Site structure

- `index.mdx` — 读书会首页
- `about.mdx` — 关于读书会
- `how-to-join.mdx` — 如何参与
- `books/<book-slug>/` — 每本书一个目录
  - `index.mdx` — 书籍概览
  - `schedule.mdx` — 阅读计划
  - `chapters/ch01.mdx` — 各章节笔记
  - `resources.mdx` — 延伸资源

## Adding a new book

1. 在 `books/` 下创建新目录，如 `books/new-book/`
2. 创建 `index.mdx`、`schedule.mdx`、`resources.mdx` 和 `chapters/` 目录
3. 在 `docs.json` 的 `navigation.tabs` 中添加新 tab

## Terminology

- 使用"读书会"而非"学习小组"
- 使用"章节笔记"而非"读书笔记"
- 使用"阅读计划"而非"时间表"

## Style preferences

- 中文内容为主
- Use active voice and second person ("你")
- Keep sentences concise
- Use sentence case for headings
- Bold for UI elements
- Code formatting for file names, commands, paths
