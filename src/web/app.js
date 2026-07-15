const $ = (selector) => document.querySelector(selector);
const state = { ready: false, busy: false, conversationId: null, documentId: null };

const els = {
  file: $('#fileInput'), dropzone: $('#dropzone'), selectFile: $('#selectFile'),
  kb: $('#knowledgeBase'), progress: $('#uploadProgress'), progressBar: $('#progressBar'),
  progressValue: $('#progressValue'), progressLabel: $('#progressLabel'), detail: $('#processingDetail'),
  documentCard: $('#documentCard'), documentName: $('#documentName'), documentMeta: $('#documentMeta'),
  question: $('#question'), send: $('#sendButton'), messages: $('#messages'), empty: $('#emptyState'),
  clear: $('#clearChat'), charCount: $('#charCount'), systemState: $('#systemState'),
  systemStateText: $('#systemStateText'), chatSubtitle: $('#chatSubtitle'), toast: $('#toast'),
};

function toast(message, error = false) {
  els.toast.textContent = message;
  els.toast.className = `toast show${error ? ' error' : ''}`;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => els.toast.className = 'toast', 2800);
}

function setReady(ready) {
  state.ready = ready;
  els.question.disabled = !ready;
  els.send.disabled = !ready;
  els.question.placeholder = ready ? '针对文档提出问题...' : '知识库尚未就绪，可等待恢复或上传 PDF...';
  els.systemState.classList.toggle('ready', ready);
  els.systemStateText.textContent = ready ? '知识库已就绪' : '等待文档';
  els.chatSubtitle.textContent = ready ? `正在使用 #${els.kb.value} 知识库` : '上传完成后即可开始';
}

function safeText(value) {
  const node = document.createElement('div');
  node.textContent = value ?? '';
  return node.innerHTML;
}

function addMessage(role, content = '') {
  els.empty.classList.add('hidden');
  const node = document.createElement('article');
  node.className = `message ${role}`;
  node.innerHTML = `<div class="message-label">${role === 'user' ? 'You' : 'AgentRAG'}</div><div class="bubble"></div>`;
  node.querySelector('.bubble').textContent = content;
  els.messages.appendChild(node);
  scrollMessages();
  return node;
}

function scrollMessages() { els.messages.scrollTop = els.messages.scrollHeight; }

function showThinking(label = 'Agent 正在分析问题与检索证据') {
  let node = $('#thinking');
  if (!node) {
    node = document.createElement('div');
    node.id = 'thinking'; node.className = 'thinking';
    node.innerHTML = '<span class="thinking-dots"><i></i><i></i><i></i></span><span></span>';
    els.messages.appendChild(node);
  }
  node.querySelector('span:last-child').textContent = label;
  scrollMessages();
}

function hideThinking() { $('#thinking')?.remove(); }

async function upload(file) {
  if (!file || !file.name.toLowerCase().endsWith('.pdf')) return toast('请选择 PDF 文件', true);
  if (file.size > 20 * 1024 * 1024) return toast('文件不能超过 20 MB', true);
  const kb = els.kb.value.trim();
  if (!/^[a-zA-Z0-9_-]{1,64}$/.test(kb)) return toast('知识库标识只能包含字母、数字、- 和 _', true);

  setReady(false); state.busy = true; els.progress.classList.remove('hidden');
  els.documentCard.classList.add('hidden'); els.progressBar.style.width = '12%';
  els.progressValue.textContent = '12%'; els.progressLabel.textContent = '正在上传';
  els.detail.textContent = '正在安全传输 PDF 文档...';
  const form = new FormData(); form.append('file', file); form.append('knowledge_base_id', kb);
  try {
    const response = await fetch('/api/v1/documents/upload', { method: 'POST', body: form });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || '上传失败');
    state.documentId = data.document_id;
    els.progressBar.style.width = '46%'; els.progressValue.textContent = '46%';
    els.progressLabel.textContent = '正在解析'; els.detail.textContent = '正在分块、生成向量并构建混合索引...';
    await pollStatus(data.document_id, file.name);
  } catch (error) {
    state.busy = false; els.progress.classList.add('hidden'); toast(error.message, true);
  }
}

async function restoreKnowledgeBase(showToast = false) {
  const kb = els.kb.value.trim();
  if (!/^[a-zA-Z0-9_-]{1,64}$/.test(kb)) return setReady(false);
  setReady(false);
  els.systemStateText.textContent = '正在恢复本地知识库...';
  try {
    const response = await fetch(`/api/v1/health?knowledge_base_id=${encodeURIComponent(kb)}`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || '知识库状态检查失败');
    if (data.status !== 'ready') {
      els.documentCard.classList.add('hidden');
      return;
    }
    setReady(true);
    els.documentName.textContent = `${kb} 知识库`;
    els.documentMeta.textContent = `${data.indexed_chunks || 0} 个语义片段 · 已从本地索引恢复`;
    els.documentCard.classList.remove('hidden');
    if (showToast) toast('本地知识库已恢复');
  } catch (error) {
    setReady(false);
    if (showToast) toast(error.message, true);
  }
}

async function pollStatus(id, filename) {
  for (let attempt = 0; attempt < 360; attempt++) {
    await new Promise(resolve => setTimeout(resolve, 1500));
    const response = await fetch(`/api/v1/documents/${encodeURIComponent(id)}/status`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || '无法获取处理状态');
    const percent = Math.min(92, 46 + attempt * 2);
    els.progressBar.style.width = `${percent}%`; els.progressValue.textContent = `${percent}%`;
    if (data.status === 'completed') {
      els.progressBar.style.width = '100%'; els.progressValue.textContent = '100%';
      els.progressLabel.textContent = '索引完成'; els.detail.textContent = '文档已进入知识库，可以开始提问。';
      setTimeout(() => els.progress.classList.add('hidden'), 900);
      els.documentName.textContent = filename;
      els.documentMeta.textContent = `${data.chunk_count} 个语义片段 · 已完成索引`;
      els.documentCard.classList.remove('hidden'); state.busy = false; setReady(true);
      els.question.focus(); toast('知识库构建完成'); return;
    }
    if (data.status === 'failed') throw new Error(data.error || '文档处理失败');
  }
  throw new Error('文档处理超时，请稍后重试');
}

function parseSSEBlock(block) {
  let event = 'message'; const data = [];
  for (const line of block.split('\n')) {
    if (line.startsWith('event:')) event = line.slice(6).trim();
    if (line.startsWith('data:')) data.push(line.slice(5).trim());
  }
  if (!data.length) return null;
  try { return { event, data: JSON.parse(data.join('\n')) }; }
  catch { return null; }
}

async function ask(question) {
  if (!state.ready || state.busy || !question.trim()) return;
  state.busy = true; els.send.disabled = true; els.question.disabled = true;
  addMessage('user', question.trim()); els.question.value = ''; updateTextarea(); showThinking();
  const answerNode = addMessage('assistant', ''); answerNode.classList.add('hidden');
  try {
    const response = await fetch('/api/v1/chat', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: question.trim(), knowledge_base_id: els.kb.value.trim(), conversation_id: state.conversationId, stream: true }),
    });
    if (!response.ok) { const data = await response.json(); throw new Error(data.detail || '请求失败'); }
    const reader = response.body.getReader(); const decoder = new TextDecoder(); let buffer = ''; let answer = '';
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done }).replace(/\r\n/g, '\n');
      const blocks = buffer.split('\n\n'); buffer = blocks.pop() || '';
      for (const block of blocks) {
        const item = parseSSEBlock(block); if (!item) continue;
        if (item.event === 'thinking') showThinking(item.data.content.replace(/^\[|\].*$/g, '') || 'Agent 正在推理');
        if (item.event === 'tool_call') showThinking(`正在调用 ${item.data.tool} 检索证据`);
        if (item.event === 'token') {
          hideThinking(); answerNode.classList.remove('hidden'); answer += item.data.token;
          answerNode.querySelector('.bubble').textContent = answer; scrollMessages();
        }
        if (item.event === 'done') {
          hideThinking(); answerNode.classList.remove('hidden');
          answer = item.data.answer || answer; answerNode.querySelector('.bubble').textContent = answer;
          state.conversationId = item.data.conversation_id;
          renderSources(answerNode, item.data.sources || []);
        }
        if (item.event === 'error') throw new Error(item.data.error || '回答生成失败');
      }
      if (done) break;
    }
    if (!answer) answerNode.querySelector('.bubble').textContent = '未能生成回答，请换一种问法重试。';
  } catch (error) {
    hideThinking(); answerNode.classList.remove('hidden');
    answerNode.querySelector('.bubble').textContent = `抱歉，本次处理失败：${error.message}`; toast(error.message, true);
  } finally {
    state.busy = false; els.question.disabled = false; els.send.disabled = false; els.question.focus(); scrollMessages();
  }
}

function renderSources(message, sources) {
  if (!sources.length) return;
  const box = document.createElement('div'); box.className = 'sources';
  sources.slice(0, 6).forEach((source) => {
    const card = document.createElement('div'); card.className = 'source-card';
    card.innerHTML = `<strong>第 ${Number(source.page) || 0} 页 · ${(Number(source.score) || 0).toFixed(3)}</strong><p>${safeText(source.text)}</p>`;
    box.appendChild(card);
  });
  message.appendChild(box);
}

function updateTextarea() {
  els.question.style.height = 'auto';
  els.question.style.height = `${Math.min(140, els.question.scrollHeight)}px`;
  els.charCount.textContent = `${els.question.value.length} / 5000`;
}

els.selectFile.addEventListener('click', (e) => { e.stopPropagation(); els.file.click(); });
els.dropzone.addEventListener('click', () => els.file.click());
els.dropzone.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') els.file.click(); });
els.file.addEventListener('change', () => upload(els.file.files[0]));
['dragenter', 'dragover'].forEach(name => els.dropzone.addEventListener(name, (e) => { e.preventDefault(); els.dropzone.classList.add('dragging'); }));
['dragleave', 'drop'].forEach(name => els.dropzone.addEventListener(name, (e) => { e.preventDefault(); els.dropzone.classList.remove('dragging'); }));
els.dropzone.addEventListener('drop', (e) => upload(e.dataTransfer.files[0]));
els.send.addEventListener('click', () => ask(els.question.value));
els.question.addEventListener('input', updateTextarea);
els.question.addEventListener('keydown', (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); ask(els.question.value); } });
document.querySelectorAll('.suggestions button').forEach(button => button.addEventListener('click', () => { els.question.value = button.textContent; updateTextarea(); if (state.ready) ask(button.textContent); else toast('知识库尚未就绪，请等待恢复或上传 PDF'); }));
els.clear.addEventListener('click', () => { state.conversationId = null; els.messages.innerHTML = ''; els.messages.appendChild(els.empty); els.empty.classList.remove('hidden'); toast('对话已清空'); });
els.kb.addEventListener('change', () => {
  state.conversationId = null;
  els.documentCard.classList.add('hidden');
  restoreKnowledgeBase(true);
});

restoreKnowledgeBase();
