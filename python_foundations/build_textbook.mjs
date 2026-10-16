// 教材构建工具：不执行 Markdown 中的代码，只嵌入已验证的文件和输出。
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = path.dirname(fileURLToPath(import.meta.url));
const { marked } = await import(process.argv[2] ? pathToFileURL(path.resolve(process.argv[2])).href : 'marked');
const verification = JSON.parse(fs.readFileSync(path.join(root, 'verification.json'), 'utf8'));
if (verification.status !== 'passed') throw new Error('请先运行 verify_examples.py');
const esc = (s) => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const code = (text, language = '') => '\n```' + language + '\n' + text.trimEnd() + '\n```\n';

function include(markdown) {
  return markdown
    .replace(/<!-- EXAMPLE: (.+?) -->/g, (_, relative) => {
      const record = verification.runs[relative];
      if (!record || record.exit_code !== 0) throw new Error('缺少验证：' + relative);
      const source = fs.readFileSync(path.join(root, relative), 'utf8');
      return '**可运行例子：** [' + relative + '](' + relative + ')\n\n' +
        code(source, 'python') + '\n**运行方式：**\n' +
        code(verification.executable + ' ' + path.join(root, relative), 'bash') +
        '\n**本机实际输出：**\n' + code(record.stdout, 'text');
    })
    .replace(/<!-- OUTPUT: (.+?) -->/g, (_, name) => {
      const record = verification.runs[name];
      if (!record || record.exit_code !== 0) throw new Error('缺少输出：' + name);
      return '**本机实际输出：**\n' + code(record.stdout, 'text');
    })
    .replace(/<!-- SOURCE: (.+?) -->/g, (_, relative) => {
      const source = fs.readFileSync(path.join(root, relative), 'utf8');
      return code(source, relative.endsWith('.py') ? 'python' : 'json');
    });
}

const names = fs.readdirSync(path.join(root, 'chapters')).filter(n => /^\d\d_.*\.md$/.test(n)).sort();
if (names.length !== 17) throw new Error('应当有 17 个教学单元');
const chapters = names.map(name => {
  const markdown = include(fs.readFileSync(path.join(root, 'chapters', name), 'utf8'));
  const title = markdown.match(/^## (.+)$/m)[1];
  return { id: 'unit-' + name.slice(0, 2), title, markdown, number: Number(name.slice(0, 2)) };
});
const intro = include(fs.readFileSync(path.join(root, 'chapters', 'intro.md'), 'utf8'));
const appendix = include(fs.readFileSync(path.join(root, 'chapters', 'appendix.md'), 'utf8'));
const markdown = [intro, ...chapters.map(c => '<a id="' + c.id + '"></a>\n\n' + c.markdown), appendix].join('\n\n---\n\n');
fs.writeFileSync(path.join(root, 'textbook.md'), markdown);

marked.setOptions({ gfm: true, breaks: false });
const level = n => [0, 2, 11].includes(n) ? {id:'reference', label:'速查'} : n >= 13 ? {id:'practice', label:'实战'} : {id:'core', label:'重点'};
const stage = n => n <= 6 ? '语言语义与数据' : n <= 12 ? '调用、协议与工程接口' : n === 13 ? '真实源码阅读' : n <= 15 ? 'PyTorch 语义边界' : '综合接口审查';
const nav = chapters.map(c => '<a class="toc-link" href="#' + c.id + '" data-level="' + level(c.number).id + '" data-search="' + esc(c.title.toLowerCase()) + '"><span>' + String(c.number).padStart(2, '0') + '</span><span>' + esc(c.title.replace(/^\d+ · /, '')) + '</span><span class="level-tag">' + level(c.number).label + '</span></a>').join('');
const content = chapters.map((c, i) => {
  const previous = i ? chapters[i - 1] : {id: 'start', title: '阅读说明'};
  const next = i < chapters.length - 1 ? chapters[i + 1] : {id: 'appendix', title: '速查与参考'};
  return '<section class="chapter" id="' + c.id + '" data-chapter="' + c.number + '"><div class="chapter-kicker"><span class="chapter-level">' + level(c.number).label + '</span>' + stage(c.number) + '</div>' + marked.parse(c.markdown) +
    '<nav class="chapter-pager" aria-label="章节切换"><a href="#' + previous.id + '"><small>← 上一单元</small>' + esc(previous.title) + '</a><a href="#' + next.id + '"><small>下一单元 →</small>' + esc(next.title) + '</a></nav></section>';
}).join('\n');

const css = fs.readFileSync(path.join(root, 'reader.css'), 'utf8');
const js = fs.readFileSync(path.join(root, 'reader.js'), 'utf8');
const html = `<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light"><title>Python 迁移教材 · Java / C++ / Dart 开发者版</title><style>${css}</style></head>
<body><a class="skip-link" href="#main">跳到正文</a>
<aside class="sidebar" id="sidebar"><a class="brand" href="#start"><span class="brand-mark" aria-hidden="true">Py</span><span>Python 迁移<span class="brand-sub">面向大模型工程</span></span></a>
<p class="sidebar-intro">Java · C++ · Dart → Python</p><label class="search-label" for="toc-search">查找单元</label><input type="search" id="toc-search" placeholder="例如：函数、类、张量" autocomplete="off">
<div class="toc-filters" role="group" aria-label="目录筛选"><button type="button" data-filter="all" aria-pressed="true">全部</button><button type="button" data-filter="core" aria-pressed="false">重点</button><button type="button" data-filter="practice" aria-pressed="false">实战</button></div><p class="toc-counter" id="toc-counter" aria-live="polite">17 / 17 个单元</p><nav aria-label="教材目录"><a class="toc-meta" href="#start">阅读说明与路线</a>${nav}<p id="toc-empty" hidden>没有匹配的单元</p><a class="toc-meta" href="#appendix">速查与官方参考</a></nav>
<div class="sidebar-footer"><span class="status-dot"></span>离线阅读 · CPU 示例<br><span>Python 3.9 · PyTorch 2.2</span></div></aside>
<div class="workspace"><header class="toolbar"><button class="menu-button" id="menu-toggle" aria-expanded="false" aria-controls="sidebar">目录</button><div class="breadcrumb">学习补充 <span>/</span> <b id="current-label">阅读说明</b></div><div class="tools"><button id="font-smaller" aria-label="缩小正文字号">A−</button><button id="font-larger" aria-label="放大正文字号">A＋</button><button id="print-book">打印 / PDF</button></div></header>
<div class="reading-track" aria-hidden="true"><div id="reading-progress"></div></div>
<main id="main"><section class="intro" id="start"><div class="eyebrow">资深开发者版 / 语义对照与实跑验证 / 17 个单元</div>${marked.parse(intro)}<div class="start-action"><a href="#unit-01">进入迁移路线 <span aria-hidden="true">→</span></a><a href="textbook.md" class="secondary-link">Markdown 版 ↗</a></div></section>${content}<section class="chapter appendix" id="appendix"><div class="chapter-kicker">随时回来查阅</div>${marked.parse(appendix)}</section></main>
<footer class="page-footer">Java / C++ / Dart → Python · 以语义差异为起点，回到真实模型代码。</footer></div><button class="mobile-shade" id="mobile-shade" aria-label="关闭目录" hidden></button><script>${js}</script></body></html>`;
fs.writeFileSync(path.join(root, 'textbook.html'), html);
console.log(JSON.stringify({units: chapters.length, markdown_chars: markdown.length, html_bytes: Buffer.byteLength(html)}));
