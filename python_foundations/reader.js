// 所有交互均在本地执行；正文在禁用 JavaScript 时也可阅读。
(() => {
  const sidebar = document.querySelector('#sidebar');
  const menu = document.querySelector('#menu-toggle');
  const shade = document.querySelector('#mobile-shade');
  const links = Array.from(document.querySelectorAll('.toc-link'));
  const chapters = Array.from(document.querySelectorAll('[data-chapter]'));
  function setMenu(open) {
    sidebar.classList.toggle('open', open);
    menu.setAttribute('aria-expanded', String(open));
    shade.hidden = !open;
  }
  menu.addEventListener('click', () => setMenu(!sidebar.classList.contains('open')));
  shade.addEventListener('click', () => setMenu(false));
  sidebar.querySelectorAll('a[href^="#"]').forEach(link => link.addEventListener('click', () => setMenu(false)));
  document.addEventListener('keydown', event => { if (event.key === 'Escape') setMenu(false); });
  let selectedLevel = 'all';
  const search = document.querySelector('#toc-search');
  const filterButtons = Array.from(document.querySelectorAll('[data-filter]'));
  function filterNavigation() {
    const query = search.value.trim().toLowerCase();
    let visible = 0;
    links.forEach(link => {
      link.hidden = !(link.dataset.search.includes(query) && (selectedLevel === 'all' || link.dataset.level === selectedLevel));
      if (!link.hidden) visible += 1;
    });
    document.querySelector('#toc-counter').textContent = visible + ' / ' + links.length + ' 个单元';
    document.querySelector('#toc-empty').hidden = visible !== 0;
    filterButtons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.filter === selectedLevel)));
  }
  search.addEventListener('input', filterNavigation);
  filterButtons.forEach(button => button.addEventListener('click', () => {
    selectedLevel = button.dataset.filter;
    filterNavigation();
  }));

  let size = window.innerWidth < 821 ? 17 : 18;
  function font(delta) {
    size = Math.max(15, Math.min(22, size + delta));
    document.documentElement.style.setProperty('--font-size', size + 'px');
    document.querySelector('#font-smaller').disabled = size === 15;
    document.querySelector('#font-larger').disabled = size === 22;
  }
  document.querySelector('#font-smaller').addEventListener('click', () => font(-1));
  document.querySelector('#font-larger').addEventListener('click', () => font(1));
  document.querySelector('#print-book').addEventListener('click', () => window.print());
  const closedBeforePrint = [];
  window.addEventListener('beforeprint', () => {
    document.querySelectorAll('details:not([open])').forEach(item => { closedBeforePrint.push(item); item.open = true; });
  });
  window.addEventListener('afterprint', () => { closedBeforePrint.splice(0).forEach(item => { item.open = false; }); });

  let queued = false;
  function updateReading() {
    const total = document.documentElement.scrollHeight - window.innerHeight;
    document.querySelector('#reading-progress').style.width = (total > 0 ? Math.max(0, Math.min(100, window.scrollY / total * 100)) : 100) + '%';
    let current = null;
    chapters.forEach(chapter => { if (chapter.getBoundingClientRect().top <= 160) current = chapter; });
    links.forEach(link => {
      const active = current && link.hash === '#' + current.id;
      link.classList.toggle('active', Boolean(active));
      if (active) link.setAttribute('aria-current', 'location'); else link.removeAttribute('aria-current');
    });
    const appendixVisible = document.querySelector('#appendix').getBoundingClientRect().top <= 160;
    document.querySelector('#current-label').textContent = appendixVisible ? '速查与参考' : current ? current.querySelector('h2').textContent : '阅读说明';
    queued = false;
  }
  window.addEventListener('scroll', () => { if (!queued) { queued = true; requestAnimationFrame(updateReading); } }, { passive: true });
  window.addEventListener('resize', updateReading);
  updateReading();

  // file:// 场景可能禁用剪贴板 API；失败时保留文本选择方式。
  document.querySelectorAll('pre').forEach(pre => {
    const button = document.createElement('button');
    button.className = 'copy-button';
    button.textContent = '复制';
    button.setAttribute('aria-label', '复制这段代码或输出');
    button.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(pre.querySelector('code').textContent);
        button.textContent = '已复制';
      } catch {
        const selection = window.getSelection();
        const range = document.createRange();
        range.selectNodeContents(pre.querySelector('code'));
        selection.removeAllRanges(); selection.addRange(range);
        button.textContent = '已选中，请复制';
      }
      setTimeout(() => { button.textContent = '复制'; }, 1800);
    });
    pre.appendChild(button);
  });
})();
