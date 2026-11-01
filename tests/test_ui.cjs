// Dependency-free UI regressions: execute the actual embedded application in a
// small DOM harness. Real layout/painting is validated separately in a browser.
// Run with: node --test tests/test_ui.cjs
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const template = fs.readFileSync(path.join(__dirname, '../vespa_retrieval/templates/ui_template.html'), 'utf8');
const source = template.match(/<script>([\s\S]*?)<\/script>/)[1];
const flush = () => new Promise(resolve => setImmediate(resolve));

function harness(data = {}) {
  const ids = new Map();
  const images = [];
  const requests = [];
  const errors = [];
  let observer;
  let mobile = false;
  class Element {
    constructor(tag = 'div') {
      this.tagName = tag.toUpperCase(); this.children = []; this.style = {};
      this.dataset = {}; this.attributes = {}; this.events = {};
      this.className = ''; this.value = ''; this.checked = false;
      this.clientWidth = 300; this.clientHeight = 400;
      this.scrollWidth = 0; this.scrollHeight = 0;
      this.classList = {
        contains: c => this.className.split(/\s+/).includes(c),
        add: c => { if (!this.classList.contains(c)) this.className += ' ' + c; },
        remove: c => { this.className = this.className.split(/\s+/).filter(x => x !== c).join(' '); },
        toggle: (c, force) => {
          const on = force === undefined ? !this.classList.contains(c) : force;
          this.classList[on ? 'add' : 'remove'](c); return on;
        },
      };
    }
    appendChild(child) { child.parentElement = this; this.children.push(child); return child; }
    setAttribute(key, value) { this.attributes[key] = String(value); }
    getAttribute(key) { return this.attributes[key] ?? null; }
    removeAttribute(key) { delete this.attributes[key]; }
    addEventListener(key, callback) { this.events[key] = callback; }
    focus() { document.activeElement = this; }
    set textContent(value) { this._text = String(value); this.children = []; }
    get textContent() { return (this._text || '') + this.children.map(c => c.textContent).join(''); }
    set innerHTML(value) {
      this._html = value; this._text = ''; this.children = [];
      // Only table structure is needed by styleTableNode; this is not a browser.
      if (/<table[\s>]/i.test(value)) {
        const table = this.appendChild(new Element('table'));
        for (const row of value.matchAll(/<tr[^>]*>([\s\S]*?)<\/tr>/gi)) {
          const tr = table.appendChild(new Element('tr'));
          for (const cell of row[1].matchAll(/<(td|th)[^>]*>([\s\S]*?)<\/(?:td|th)>/gi)) {
            tr.appendChild(new Element(cell[1])).textContent = cell[2];
          }
        }
      }
    }
    get innerHTML() { return this._html || ''; }
    matches(selector) {
      const match = selector.match(/^(?:([\w-]+))?(?:#([\w-]+))?((?:\.[\w-]+)*)$/);
      return !!match && (!match[1] || this.tagName === match[1].toUpperCase()) &&
        (!match[2] || this.id === match[2]) &&
        (!match[3] || match[3].slice(1).split('.').every(c => this.classList.contains(c)));
    }
    querySelectorAll(selector) {
      const result = [];
      for (const child of this.children) {
        if (selector.split(',').some(s => child.matches(s.trim()))) result.push(child);
        result.push(...child.querySelectorAll(selector));
      }
      return result;
    }
    querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
  }
  const root = new Element();
  const document = {
    activeElement: null,
    createElement: tag => new Element(tag),
    getElementById: id => ids.get(id),
    querySelectorAll(selector) {
      const scoped = selector.match(/^#([\w-]+) (.*)$/);
      return scoped ? ids.get(scoped[1]).querySelectorAll(scoped[2]) : root.querySelectorAll(selector);
    },
  };
  for (const [, id] of template.matchAll(/\bid="([^"]+)"/g)) {
    const node = root.appendChild(new Element()); node.id = id; ids.set(id, node);
  }
  for (const id of ['orig-inner', 'layout-inner', 'render-inner']) {
    const node = ids.get(id); root.children = root.children.filter(x => x !== node);
    ids.get('viewer').appendChild(node);
  }
  class Image extends Element {
    constructor() { super('img'); this.complete = false; images.push(this); }
    set src(value) { this._src = value; }
    get src() { return this._src; }
  }
  const context = vm.createContext({
    document, Image,
    window: {matchMedia: () => ({matches: mobile}), addEventListener() {}},
    ResizeObserver: class {constructor(callback) {observer = callback;} observe() {}},
    console: {error: (...args) => errors.push(args), warn: (...args) => errors.push(args)},
    fetch: url => new Promise((resolve, reject) => requests.push({url, resolve, reject})),
  });
  vm.runInContext(source.replace('__LAYOUTS_JSON__', JSON.stringify(data)), context);
  return {
    ids, images, requests, errors, context,
    eval: code => vm.runInContext(code, context),
    mobile: value => {mobile = value;},
    resize: width => {
      ids.get('viewer').querySelectorAll('.viewport').forEach(v => {v.clientWidth = width;});
      observer();
    },
    load: (name, width = 1000, height = 1400) => {
      images.filter(i => i.src === 'originals/' + name && !i.complete).forEach(i => {
        i.complete = true; i.naturalWidth = width; i.naturalHeight = height; i.onload();
      });
    },
    fail: name => images.filter(i => i.src === 'originals/' + name && !i.complete).forEach(i => {i.complete = true; i.onerror();}),
    search: query => {ids.get('page-search').value = query; ids.get('page-search').events.input();},
  };
}
const records = {
  'scan_10.png': {elements: [{label: 'table', bbox: [10,20,150,100], table: {html_inline: '<table><tr><td>Torque&nbsp; limits</td></tr></table>'}}]},
  'scan_2.png': {elements: [{label: 'text', bbox: [20,30,240,80], text: 'Engine MAINTENANCE'}, {label: 'figure', bbox: [40,90,200,210], crop: 'outputs/crops/figure.png'}, {label: 'abandon', bbox: [0,0,20,20], text: 'discard'}]},
};

test('template script compiles, markup has unique IDs and a single doctype', () => {
  new vm.Script(source.replace('__LAYOUTS_JSON__', '{}'));
  const ids = [...template.matchAll(/\bid="([^"]+)"/g)].map(m => m[1]);
  assert.equal(new Set(ids).size, ids.length);
  assert.equal((template.match(/<!doctype html>/gi) || []).length, 1);
  assert.match(template, /<html lang="en">/);
  assert.match(template, /name="viewport"/);
});

test('empty dataset gives an accessible empty state and disabled controls', () => {
  const h = harness();
  assert.equal(h.ids.get('page-count').textContent, '0');
  assert.match(h.ids.get('viewer-status').textContent, /No document pages/);
  assert.equal(h.images.length, 0);
});

test('numeric ordering, automatic selection, page stats and previous/next', async () => {
  const h = harness(records); h.load('scan_2.png'); await flush();
  assert.equal(h.ids.get('active-name').textContent, 'scan_2.png');
  assert.equal(h.ids.get('page-position').textContent, '1 / 2');
  assert.equal(h.ids.get('stat-elements').textContent, '3');
  assert.equal(h.ids.get('stat-figures').textContent, '1');
  assert.equal(h.ids.get('prev-page').disabled, true);
  assert.equal(h.eval("pageButtons.get('scan_2.png').getAttribute('aria-current')"), 'page');
  h.ids.get('next-page').onclick(); h.load('scan_10.png'); await flush();
  assert.equal(h.ids.get('stat-tables').textContent, '1');
  assert.equal(h.ids.get('next-page').disabled, true);
  h.ids.get('prev-page').onclick();
  assert.equal(h.ids.get('active-name').textContent, 'scan_2.png');
});

test('OCR search handles normalized case/whitespace, tables, filenames and no matches', () => {
  const h = harness(records);
  h.search('  ＭＡＩＮＴＥＮＡＮＣＥ  ');
  assert.equal(h.eval("pageButtons.get('scan_2.png').hidden"), false);
  assert.equal(h.eval("pageButtons.get('scan_10.png').hidden"), true);
  h.search('TORQUE limits');
  assert.equal(h.eval("pageButtons.get('scan_10.png').hidden"), false);
  h.search('scan_2');
  assert.equal(h.ids.get('search-info').textContent, '1 of 2 pages match');
  h.search('not-in-any-page');
  assert.equal(h.ids.get('empty-search').hidden, false);
  h.ids.get('clear-search').onclick();
  assert.equal(h.ids.get('empty-search').hidden, true);
  assert.equal(h.eval("Array.from(pageButtons.values()).filter(b=>!b.hidden).length"), 2);
});

test('latest selection wins when image loads complete out of order', async () => {
  const h = harness(records);
  h.eval("show('scan_10.png')"); h.load('scan_10.png'); await flush();
  h.load('scan_2.png'); await flush();
  assert.equal(h.ids.get('active-name').textContent, 'scan_10.png');
  assert.equal(h.ids.get('orig-inner').querySelectorAll('.content').length, 1);
  assert.equal(h.ids.get('orig-inner').querySelector('img').alt, 'scan_10.png');
  assert.equal(h.ids.get('viewer').getAttribute('aria-busy'), 'false');
});

test('stale errors cannot overwrite a ready page; current failures are announced', async () => {
  const h = harness(records);
  h.eval("show('scan_10.png')"); h.load('scan_10.png'); await flush();
  h.fail('scan_2.png'); await flush();
  assert.match(h.ids.get('viewer-status').textContent, /Page 2 ready/);
  h.eval("show('scan_2.png')"); h.fail('scan_2.png'); await flush();
  assert.match(h.ids.get('viewer-status').textContent, /Unable to load/);
  assert.equal(h.ids.get('zoom-in').disabled, true);
});

test('fit and zoom update scroll surfaces, preserve manual zoom and clamp extremes', async () => {
  const h = harness(records); h.load('scan_2.png'); await flush();
  const surface = h.ids.get('orig-inner').querySelector('.page-surface');
  assert.equal(surface.style.width, '300px');
  assert.equal(surface.style.height, '420px');
  h.resize(180); assert.equal(surface.style.width, '180px');
  h.ids.get('zoom-in').onclick();
  const manual = surface.style.width; h.resize(250);
  assert.equal(surface.style.width, manual);
  h.ids.get('zoom-reset').onclick(); assert.equal(surface.style.width, '250px');
  h.eval('applyScale(999)'); assert.equal(surface.style.width, '4000px');
  assert.equal(h.ids.get('zoom-in').disabled, true);
  h.eval('applyScale(0)'); assert.equal(surface.style.width, '10px');
  assert.equal(h.ids.get('zoom-out').disabled, true);
});

test('bbox coordinates, figure crop paths and display toggles survive redesign', async () => {
  const h = harness(records); h.load('scan_2.png'); await flush();
  const boxes = h.ids.get('layout-inner').querySelectorAll('.box');
  assert.equal(boxes.length, 3, 'exactly one overlay per detection');
  assert.equal(boxes[0].style.left, '20px');
  assert.equal(boxes[0].style.width, '220px');
  assert.equal(boxes[2].style.display, 'none');
  const crop = h.ids.get('render-inner').querySelector('.figure-crop');
  assert.equal(crop.src, 'crops/figure.png');
  const element = h.ids.get('render-inner').querySelector('.element');
  assert.equal(element.classList.contains('no-bg'), true);
  h.ids.get('show-bg').checked = true; h.ids.get('show-bg').onchange();
  assert.equal(element.classList.contains('no-bg'), false);
  h.ids.get('show-abandon').checked = true; h.ids.get('show-abandon').onchange();
  assert.equal(boxes[2].style.display, '');
  h.ids.get('show-orig').checked = true; h.ids.get('show-orig').onchange();
  assert.equal(h.ids.get('render-inner').querySelector('.render-bg').style.display, 'block');
});

test('late external-table fetches do not mutate a newly selected page', async () => {
  const h = harness({...records, 'scan_1.png': {elements: [{label:'table', bbox:[0,0,200,100], table:{html:'tables/a.html'}}]}});
  h.load('scan_1.png'); await flush();
  const oldNode = h.ids.get('render-inner').querySelector('.table-el');
  h.eval("show('scan_2.png')"); h.load('scan_2.png'); await flush();
  h.requests[0].resolve({ok:true, text:async () => '<table><tr><td>old</td></tr></table>'});
  await flush();
  assert.equal(oldNode.innerHTML, '');
  assert.equal(h.ids.get('render-inner').querySelector('.table-el'), null);
});

test('search Enter selects a result and mobile navigation restores focus', () => {
  const h = harness(records); h.mobile(true);
  h.ids.get('toggle-sidebar').onclick();
  assert.equal(h.ids.get('toggle-sidebar').getAttribute('aria-expanded'), 'true');
  h.search('torque'); h.ids.get('page-search').events.keydown({key:'Enter'});
  assert.equal(h.ids.get('active-name').textContent, 'scan_10.png');
  assert.equal(h.ids.get('toggle-sidebar').getAttribute('aria-expanded'), 'false');
});

test('figure preprocessing caches results and switching it off restores the source', async () => {
  const h = harness(records); h.load('scan_2.png'); await flush();
  const crop = h.ids.get('render-inner').querySelector('.figure-crop');
  crop.complete = true; crop.naturalWidth = 3; crop.naturalHeight = 3;
  h.eval(`
    let canvasCalls = 0;
    const createElement = document.createElement;
    document.createElement = tag => tag !== 'canvas' ? createElement(tag) : {
      getContext: () => ({drawImage(){}, getImageData: () => ({data:new Uint8ClampedArray(36).fill(255)}), putImageData(){}}),
      toDataURL: () => { canvasCalls++; return 'data:image/png;base64,white'; }
    };
  `);
  h.ids.get('binarize').checked = true; h.ids.get('binarize').onchange();
  assert.equal(crop.src, 'data:image/png;base64,white');
  h.ids.get('binarize').checked = false; h.ids.get('binarize').onchange();
  assert.equal(crop.src, 'crops/figure.png');
  h.ids.get('binarize').checked = true; h.ids.get('binarize').onchange();
  assert.equal(h.eval('canvasCalls'), 1);
  assert.equal(h.errors.length, 0);
});

test('raw HTML tables retain bbox styling and table/crop deduplication', async () => {
  const table = '<table><tr><td>A</td></tr><tr><td>B</td></tr></table>';
  const h = harness({'scan_0.png': {elements: [
    {bbox:[1,2,201,102], text:table},
    {bbox:[1,110,201,210], table:{html_inline:table}},
    {bbox:[1,220,201,320], table:{html_inline:table}},
    {bbox:[1,330,201,430], crop:'outputs/crops/a.png', label:'figure'},
    {bbox:[1,440,201,540], crop:'outputs/crops/a.png', label:'figure'},
  ]}});
  h.load('scan_0.png'); await flush();
  const node = h.ids.get('render-inner').querySelector('.table-el');
  assert.equal(node.style.left, '1px');
  assert.equal(node.style.height, '100px');
  assert.equal(node.querySelector('table').style.width, '100%');
  assert.equal(node.querySelector('tr').style.height, 'auto');
  assert.equal(node.querySelector('td').style.overflow, 'visible');
  assert.match(node.querySelector('td').style.border, /solid #68717d/);
  assert.equal(h.ids.get('render-inner').querySelectorAll('.figure-crop').length, 1);
  assert.equal(h.ids.get('render-inner').querySelectorAll('.table-el').length, 2);
});

test('page-list keyboard navigation moves focus within visible search results', () => {
  const h = harness(records);
  h.eval("pageButtons.get('scan_2.png').focus()");
  h.ids.get('list').events.keydown({key:'ArrowDown',preventDefault(){}});
  assert.equal(h.eval("document.activeElement === pageButtons.get('scan_10.png')"), true);
  h.ids.get('list').events.keydown({key:'Home',preventDefault(){}});
  assert.equal(h.eval("document.activeElement === pageButtons.get('scan_2.png')"), true);
  h.search('torque');
  h.ids.get('list').events.keydown({key:'End',preventDefault(){}});
  assert.equal(h.eval("document.activeElement === pageButtons.get('scan_10.png')"), true);
});