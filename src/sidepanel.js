import { generateText, tool, jsonSchema } from 'ai';
import { createGoogleGenerativeAI } from '@ai-sdk/google';
import { createAnthropic } from '@ai-sdk/anthropic';
import { createOpenAI } from '@ai-sdk/openai';
import {
  LANGUAGE_LABELS,
  detectBrowserLanguage,
  resolveLanguage,
  translate
} from './i18n.js';

/**
 * Supported AI Providers
 * Models are queried dynamically from each provider's API using the user's API key.
 */
const PROVIDERS = {
  gemini: {
    label: 'Gemini',
    placeholder: 'AIzaSy...',
    keyUrl: 'https://aistudio.google.com/app/apikey'
  },
  claude: {
    label: 'Claude',
    placeholder: 'sk-ant-api03-...',
    keyUrl: 'https://console.anthropic.com/settings/keys'
  },
  openai: {
    label: 'OpenAI',
    placeholder: 'sk-proj-...',
    keyUrl: 'https://platform.openai.com/api-keys'
  }
};

// State
let currentConfig = {
  provider: 'gemini',
  model: '',
  theme: 'system', // 'system' | 'light' | 'dark'
  language: 'system', // 'system' | 'en' | 'zh' | 'ja' | 'ko' | 'es' | 'fr' | 'de'
  selectedModels: {
    gemini: '',
    claude: '',
    openai: ''
  },
  apiKeys: {
    gemini: '',
    claude: '',
    openai: ''
  }
};

// In-memory cache of fetched models per provider: { gemini: [{ id, name }], claude: [...], openai: [...] }
const providerModelsCache = {
  gemini: [],
  claude: [],
  openai: []
};
let fetchModelsDebounceTimer = null;

let discoveredTools = [];
let pageInfo = { title: '', url: '' };
let conversationHistory = [];
let isGenerating = false;

// Shared prompt history across all sessions & webpages (persisted in chrome.storage.local, max 100)
const MAX_PROMPT_HISTORY = 100;
const PROMPT_HISTORY_STORAGE_KEY = 'webmcpPromptHistory';
let promptHistory = [];
let promptHistoryIndex = -1; // -1 means currently editing draft
let currentPromptDraft = '';

async function recordPromptInHistory(promptText) {
  const cleaned = String(promptText || '').trim();
  if (!cleaned) return;

  // Avoid consecutive duplicate entries at the tail
  if (promptHistory.length === 0 || promptHistory[promptHistory.length - 1] !== cleaned) {
    promptHistory.push(cleaned);
  }
  if (promptHistory.length > MAX_PROMPT_HISTORY) {
    promptHistory = promptHistory.slice(promptHistory.length - MAX_PROMPT_HISTORY);
  }
  promptHistoryIndex = -1;
  currentPromptDraft = '';

  try {
    await chrome.storage.local.set({ [PROMPT_HISTORY_STORAGE_KEY]: promptHistory });
  } catch {
    // Ignore storage quota errors
  }
}

// Cache Gemini thoughtSignature / thought parts so multi-turn functionCall payloads
// include the required `thoughtSignature` field on Gemini 2.5 / 3 thinking models.
const geminiToolCallSignatures = [];

/**
 * Custom fetch wrapper for Google Generative AI that:
 * 1. Injects saved `thoughtSignature` (and preceding `thought` parts) back into
 *    outgoing `role: "model"` `functionCall` parts in multi-step tool loops.
 * 2. Captures `thoughtSignature` and `thought` parts from incoming Gemini responses.
 */
async function geminiFetchWithThoughtSignatures(url, options = {}) {
  if (options?.body && typeof options.body === 'string') {
    try {
      const parsedBody = JSON.parse(options.body);
      if (Array.isArray(parsedBody.contents)) {
        let sigCursor = 0;
        for (const content of parsedBody.contents) {
          if (content.role === 'model' && Array.isArray(content.parts)) {
            const newParts = [];
            for (const part of content.parts) {
              if (part && part.functionCall) {
                // Find matching signature entry from sigCursor forward
                let matchedEntry = null;
                for (let i = sigCursor; i < geminiToolCallSignatures.length; i++) {
                  if (geminiToolCallSignatures[i].name === part.functionCall.name) {
                    matchedEntry = geminiToolCallSignatures[i];
                    sigCursor = i + 1;
                    break;
                  }
                }
                if (matchedEntry) {
                  if (
                    Array.isArray(matchedEntry.thoughtParts) &&
                    matchedEntry.thoughtParts.length > 0 &&
                    !newParts.some((p) => p.thought === true)
                  ) {
                    newParts.push(...matchedEntry.thoughtParts);
                  }
                  if (matchedEntry.thoughtSignature && !part.thoughtSignature) {
                    part.thoughtSignature = matchedEntry.thoughtSignature;
                  }
                  if (matchedEntry.thought_signature && !part.thought_signature) {
                    part.thought_signature = matchedEntry.thought_signature;
                  }
                }
              }
              newParts.push(part);
            }
            content.parts = newParts;
          }
        }
        options = {
          ...options,
          body: JSON.stringify(parsedBody)
        };
      }
    } catch (e) {
      // Ignore request parse errors
    }
  }

  const response = await fetch(url, options);

  try {
    const clone = response.clone();
    const data = await clone.json();
    const candidates = data?.candidates;
    if (Array.isArray(candidates)) {
      for (const cand of candidates) {
        const parts = cand?.content?.parts;
        if (Array.isArray(parts)) {
          const thoughtParts = parts.filter((p) => p && p.thought === true);
          // Also check if a thought part itself carries a thoughtSignature
          const fallbackSig =
            parts.find((p) => p?.thoughtSignature || p?.thought_signature)?.thoughtSignature ||
            parts.find((p) => p?.thoughtSignature || p?.thought_signature)?.thought_signature ||
            null;

          for (const part of parts) {
            if (part && part.functionCall) {
              geminiToolCallSignatures.push({
                name: part.functionCall.name,
                argsJson: JSON.stringify(part.functionCall.args ?? {}),
                thoughtSignature: part.thoughtSignature || fallbackSig || undefined,
                thought_signature: part.thought_signature || undefined,
                thoughtParts
              });
            }
          }
        }
      }
    }
  } catch (e) {
    // Ignore non-JSON or stream responses
  }

  return response;
}

// DOM Elements
const setupView = document.getElementById('setup-view');
const chatView = document.getElementById('chat-view');
const providerTabs = document.querySelectorAll('.provider-tab');
const modelSelect = document.getElementById('model-select');
const refreshModelsBtn = document.getElementById('refresh-models-btn');
const themeSelect = document.getElementById('theme-select');
const langSelect = document.getElementById('lang-select');
const langSystemOption = document.getElementById('lang-system-option');
const apiKeyInput = document.getElementById('api-key-input');
const apiKeyLabel = document.getElementById('api-key-label');
const getApiKeyLink = document.getElementById('get-api-key-link');
const copyKeyBtn = document.getElementById('copy-key-btn');
const setupError = document.getElementById('setup-error');
const activeProviderPill = document.getElementById('active-provider-pill');
const settingsToggleBtn = document.getElementById('settings-toggle-btn');
const refreshToolsBtn = document.getElementById('refresh-tools-btn');

const toolsSummaryToggle = document.getElementById('tools-summary-toggle');
const toolsStatusDot = document.getElementById('tools-status-dot');
const toolsCountLabel = document.getElementById('tools-count-label');
const toolsModalBackdrop = document.getElementById('tools-modal-backdrop');
const toolsModalBadge = document.getElementById('tools-modal-badge');
const closeToolsModalBtn = document.getElementById('close-tools-modal-btn');
const toolsListEl = document.getElementById('tools-list');
const newChatBtn = document.getElementById('new-chat-btn');
const messagesContainer = document.getElementById('messages-container');
const welcomeCard = document.getElementById('welcome-card');
const promptForm = document.getElementById('prompt-form');
const promptInput = document.getElementById('prompt-input');
const sendBtn = document.getElementById('send-btn');

const systemColorMedia = window.matchMedia('(prefers-color-scheme: dark)');

/**
 * Helper to translate strings in the current active language
 */
function t(key, params = {}) {
  return translate(currentConfig.language, key, params);
}

/**
 * Apply Theme (Light / Dark / System)
 */
function applyTheme(themePref = currentConfig.theme) {
  const pref = themePref || 'system';
  const effectiveTheme =
    pref === 'system' ? (systemColorMedia.matches ? 'dark' : 'light') : pref;

  document.documentElement.setAttribute('data-theme', effectiveTheme);

  if (themeSelect && themeSelect.value !== pref) {
    themeSelect.value = pref;
  }
}

systemColorMedia.addEventListener('change', () => {
  if (currentConfig.theme === 'system') {
    applyTheme('system');
  }
});

/**
 * Apply i18n translations to static and dynamic DOM elements
 */
function applyI18n() {
  const resolvedLang = resolveLanguage(currentConfig.language);
  document.documentElement.lang = resolvedLang;

  // Static text nodes
  document.querySelectorAll('[data-i18n]').forEach((el) => {
    const key = el.getAttribute('data-i18n');
    if (key) {
      el.textContent = t(key);
    }
  });

  // Static HTML nodes (containing <strong> or <code>)
  document.querySelectorAll('[data-i18n-html]').forEach((el) => {
    const key = el.getAttribute('data-i18n-html');
    if (key) {
      el.innerHTML = t(key);
    }
  });

  // Browser default option label
  if (langSystemOption) {
    const browserLang = detectBrowserLanguage();
    const browserLangName = LANGUAGE_LABELS[browserLang] || LANGUAGE_LABELS.en;
    langSystemOption.textContent = t('langSystem', { lang: browserLangName });
  }

  if (langSelect && langSelect.value !== (currentConfig.language || 'system')) {
    langSelect.value = currentConfig.language || 'system';
  }

  // Tooltips and placeholders
  if (refreshToolsBtn) refreshToolsBtn.title = t('refreshToolsTitle');
  if (settingsToggleBtn) {
    const isSetupVisible = !setupView.classList.contains('hidden');
    settingsToggleBtn.title = isSetupVisible ? t('homeTitle') : t('settingsTitle');
  }
  if (copyKeyBtn) copyKeyBtn.title = t('copyKeyTitle');
  if (refreshModelsBtn) refreshModelsBtn.title = t('fetchModelsTitle');
  if (newChatBtn) newChatBtn.title = t('newChatTitle');
  if (closeToolsModalBtn) closeToolsModalBtn.title = t('closeDialogTitle');
  if (sendBtn) sendBtn.title = t('sendTitle');
  if (promptInput) promptInput.placeholder = t('promptPlaceholder');

  const providerMeta = PROVIDERS[currentConfig.provider] || PROVIDERS.gemini;
  if (apiKeyLabel) {
    apiKeyLabel.textContent = t('apiKeyLabel', { provider: providerMeta.label });
  }
  if (getApiKeyLink) {
    getApiKeyLink.href = providerMeta.keyUrl;
    getApiKeyLink.textContent = t('getApiKeyLink', { provider: providerMeta.label });
  }

  renderDiscoveredTools();
}

/**
 * Query available models from the selected AI service API
 */
async function fetchModelsFromService(provider, apiKey) {
  const key = (apiKey || '').trim();
  if (!key) {
    return [];
  }

  if (provider === 'gemini') {
    const res = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models?pageSize=100&key=${encodeURIComponent(key)}`
    );
    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData?.error?.message || `Gemini API returned ${res.status}`);
    }
    const data = await res.json();
    const list = Array.isArray(data.models) ? data.models : [];
    return list
      .filter(
        (m) =>
          Array.isArray(m.supportedGenerationMethods) &&
          m.supportedGenerationMethods.includes('generateContent')
      )
      .map((m) => {
        const id = String(m.name || '').replace(/^models\//, '');
        return {
          id,
          name: m.displayName ? `${m.displayName} (${id})` : id
        };
      })
      .filter((m) => m.id);
  }

  if (provider === 'claude') {
    const res = await fetch('https://api.anthropic.com/v1/models?limit=100', {
      headers: {
        'x-api-key': key,
        'anthropic-version': '2023-06-01',
        'anthropic-dangerous-direct-browser-access': 'true'
      }
    });
    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData?.error?.message || `Anthropic API returned ${res.status}`);
    }
    const data = await res.json();
    const list = Array.isArray(data.data) ? data.data : [];
    return list
      .map((m) => ({
        id: m.id,
        name: m.display_name ? `${m.display_name} (${m.id})` : m.id
      }))
      .filter((m) => m.id);
  }

  if (provider === 'openai') {
    const res = await fetch('https://api.openai.com/v1/models', {
      headers: {
        Authorization: `Bearer ${key}`
      }
    });
    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData?.error?.message || `OpenAI API returned ${res.status}`);
    }
    const data = await res.json();
    const list = Array.isArray(data.data) ? data.data : [];
    return list
      .map((m) => ({
        id: m.id,
        name: m.id
      }))
      .filter((m) => m.id)
      .sort((a, b) => a.id.localeCompare(b.id));
  }

  return [];
}

/**
 * Populate the model <select> dropdown with fetched models or status message
 */
function populateModelDropdown(models, preferredModelId, emptyText = t('enterKeyToLoadModels')) {
  modelSelect.innerHTML = '';

  if (!models || models.length === 0) {
    const opt = document.createElement('option');
    opt.value = preferredModelId || '';
    opt.textContent = preferredModelId
      ? `${preferredModelId} ${t('savedSuffix')}`
      : emptyText;
    modelSelect.appendChild(opt);
    modelSelect.disabled = !preferredModelId;
    return;
  }

  modelSelect.disabled = false;
  models.forEach((m) => {
    const opt = document.createElement('option');
    opt.value = m.id;
    opt.textContent = m.name;
    modelSelect.appendChild(opt);
  });

  const hasPreferred = preferredModelId && models.some((m) => m.id === preferredModelId);
  modelSelect.value = hasPreferred ? preferredModelId : models[0].id;
  currentConfig.model = modelSelect.value;
  currentConfig.selectedModels[currentConfig.provider] = modelSelect.value;
  persistConfig();
}

function updateActiveProviderPill() {
  if (hasValidActiveKey() && currentConfig.model) {
    const providerMeta = PROVIDERS[currentConfig.provider] || PROVIDERS.gemini;
    activeProviderPill.textContent = `${providerMeta.label} · ${currentConfig.model}`;
    activeProviderPill.classList.remove('hidden');
  } else {
    activeProviderPill.classList.add('hidden');
  }
}

async function persistConfig() {
  updateActiveProviderPill();
  await chrome.storage.local.set({ webmcpConfig: currentConfig });
}

/**
 * Trigger a live model query for the active provider and key
 */
async function loadModelsForProvider(providerKey, apiKey, preferredModelId, forceRefresh = false) {
  const key = (apiKey || '').trim();
  if (!key) {
    populateModelDropdown([], '', t('enterKeyToLoadModels'));
    return [];
  }

  if (!forceRefresh && providerModelsCache[providerKey]?.length > 0) {
    populateModelDropdown(providerModelsCache[providerKey], preferredModelId);
    return providerModelsCache[providerKey];
  }

  modelSelect.innerHTML = `<option value="">${t('loadingModels')}</option>`;
  modelSelect.disabled = true;
  if (refreshModelsBtn) refreshModelsBtn.disabled = true;
  setupError.classList.add('hidden');

  try {
    const models = await fetchModelsFromService(providerKey, key);
    providerModelsCache[providerKey] = models;
    if (currentConfig.provider === providerKey) {
      populateModelDropdown(models, preferredModelId, t('noModelsReturned'));
    }
    return models;
  } catch (err) {
    if (currentConfig.provider === providerKey) {
      populateModelDropdown([], preferredModelId, t('failedToLoadModels'));
      setupError.textContent = t('couldNotFetchModels', {
        error: err?.message || String(err)
      });
      setupError.classList.remove('hidden');
    }
    return [];
  } finally {
    if (refreshModelsBtn) refreshModelsBtn.disabled = false;
  }
}

/**
 * Initialize extension UI and load saved settings
 */
async function init() {
  const stored = await chrome.storage.local.get(['webmcpConfig', PROMPT_HISTORY_STORAGE_KEY]);
  if (Array.isArray(stored?.[PROMPT_HISTORY_STORAGE_KEY])) {
    promptHistory = stored[PROMPT_HISTORY_STORAGE_KEY]
      .filter((item) => typeof item === 'string' && item.trim().length > 0)
      .slice(-MAX_PROMPT_HISTORY);
  }
  if (stored?.webmcpConfig) {
    currentConfig = {
      ...currentConfig,
      ...stored.webmcpConfig,
      theme: stored.webmcpConfig.theme || 'system',
      language: stored.webmcpConfig.language || 'system',
      selectedModels: {
        ...currentConfig.selectedModels,
        ...(stored.webmcpConfig.selectedModels || {})
      },
      apiKeys: {
        ...currentConfig.apiKeys,
        ...(stored.webmcpConfig.apiKeys || {})
      }
    };
  }

  applyTheme(currentConfig.theme);
  applyI18n();

  await renderSetupForm(currentConfig.provider, currentConfig.model);

  if (hasValidActiveKey() && currentConfig.model) {
    showChatView();
    await refreshPageTools();
  } else {
    showSetupView(false);
  }
}

function hasValidActiveKey() {
  const key = currentConfig.apiKeys?.[currentConfig.provider];
  return Boolean(key && key.trim().length > 0);
}

/**
 * Render provider tabs, API key input, and query models from service
 */
async function renderSetupForm(providerKey, selectedModelId) {
  const providerMeta = PROVIDERS[providerKey] || PROVIDERS.gemini;

  providerTabs.forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.provider === providerKey);
  });

  const savedKey = currentConfig.apiKeys?.[providerKey] || '';
  const preferredModel = selectedModelId || currentConfig.selectedModels?.[providerKey] || '';

  apiKeyLabel.textContent = t('apiKeyLabel', { provider: providerMeta.label });
  if (getApiKeyLink) {
    getApiKeyLink.href = providerMeta.keyUrl;
    getApiKeyLink.textContent = t('getApiKeyLink', { provider: providerMeta.label });
  }
  apiKeyInput.placeholder = providerMeta.placeholder;
  apiKeyInput.value = savedKey;
  if (themeSelect) themeSelect.value = currentConfig.theme || 'system';
  if (langSelect) langSelect.value = currentConfig.language || 'system';
  setupError.classList.add('hidden');

  await loadModelsForProvider(providerKey, savedKey, preferredModel, false);
}

const HOME_ICON_SVG = `<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path><polyline points="9 22 9 12 15 12 15 22"></polyline></svg>`;

function showSetupView() {
  setupView.classList.remove('hidden');
  chatView.classList.add('hidden');
  if (refreshToolsBtn) refreshToolsBtn.classList.add('hidden');
  if (settingsToggleBtn) {
    settingsToggleBtn.innerHTML = HOME_ICON_SVG;
    settingsToggleBtn.title = t('homeTitle');
  }
  updateActiveProviderPill();
}

function showChatView() {
  setupView.classList.add('hidden');
  chatView.classList.remove('hidden');
  if (refreshToolsBtn) refreshToolsBtn.classList.remove('hidden');
  if (settingsToggleBtn) {
    settingsToggleBtn.textContent = '⚙️';
    settingsToggleBtn.title = t('settingsTitle');
  }
  updateActiveProviderPill();
  promptInput.focus();
}

/**
 * Active tab communication & direct MAIN-world execution helpers
 * Uses chrome.scripting.executeScript({ world: 'MAIN' }) to directly call
 * `await document.modelContext.getTools()` and `await document.modelContext.executeTool()`
 * in the active tab's MAIN world (across all frames), bypassing stale content script state.
 */
async function getActiveTab() {
  const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
  return tabs[0] || null;
}

/**
 * Injected directly into the webpage's MAIN world to query `await document.modelContext.getTools()`
 */
async function inPageGetWebMCPTools() {
  function sanitizeSchema(schema) {
    if (!schema) return { type: 'object', properties: {} };
    if (typeof schema === 'string') {
      try {
        return JSON.parse(schema);
      } catch {
        return { type: 'object', properties: {} };
      }
    }
    try {
      return JSON.parse(JSON.stringify(schema));
    } catch {
      return { type: 'object', properties: {} };
    }
  }

  function extractArray(raw) {
    if (!raw) return [];
    if (Array.isArray(raw)) return raw;
    if (typeof raw === 'string') {
      try {
        return extractArray(JSON.parse(raw));
      } catch {
        return [];
      }
    }
    if (Array.isArray(raw.tools)) return raw.tools;
    if (raw instanceof Map) return Array.from(raw.values());
    if (typeof raw[Symbol.iterator] === 'function') return Array.from(raw);
    if (typeof raw === 'object') {
      return Object.entries(raw).map(([k, v]) =>
        v && typeof v === 'object' ? { name: v.name || k, ...v } : { name: k }
      );
    }
    return [];
  }

  const merged = new Map();

  // 1. Primary: await document.modelContext.getTools()
  const mc =
    document.modelContext ||
    window.modelContext ||
    navigator.modelContext ||
    document.modelContextTesting ||
    navigator.modelContextTesting;

  if (mc) {
    const getFn =
      typeof mc.getTools === 'function'
        ? mc.getTools.bind(mc)
        : typeof mc.listTools === 'function'
          ? mc.listTools.bind(mc)
          : null;

    if (getFn) {
      try {
        const rawTools = await getFn();
        const list = extractArray(rawTools);
        for (const t of list) {
          const toolName = t?.name || t?.toolName || t?.id || t?.function?.name;
          if (toolName) {
            const desc =
              t.description || t.toolDescription || t.function?.description || '';
            const schema =
              t.inputSchema ||
              t.parameters ||
              t.schema ||
              t.input_schema ||
              t.function?.parameters ||
              t.function?.inputSchema;
            merged.set(String(toolName), {
              name: String(toolName),
              description: String(desc),
              inputSchema: sanitizeSchema(schema),
              source: 'modelContext'
            });
          }
        }
      } catch (e) {
        console.warn('[webmcp] document.modelContext.getTools() error:', e);
      }
    } else if (Array.isArray(mc.tools) || mc.tools instanceof Map) {
      for (const t of extractArray(mc.tools)) {
        const toolName = t?.name || t?.toolName || t?.id;
        if (toolName && !merged.has(String(toolName))) {
          merged.set(String(toolName), {
            name: String(toolName),
            description: String(t.description || ''),
            inputSchema: sanitizeSchema(t.inputSchema || t.parameters || t.schema),
            source: 'modelContext'
          });
        }
      }
    }
  }

  // 2. Also scan declarative <form toolname="..."> in case page uses declarative forms
  const forms = document.querySelectorAll('form[toolname], [data-webmcp-tool]');
  forms.forEach((form) => {
    const name = form.getAttribute('toolname') || form.getAttribute('data-webmcp-tool');
    if (!name || merged.has(name)) return;
    const description =
      form.getAttribute('tooldescription') ||
      form.getAttribute('data-webmcp-description') ||
      form.getAttribute('aria-label') ||
      `Submit the ${name} form`;
    const properties = {};
    const required = [];
    form.querySelectorAll('input, select, textarea').forEach((field) => {
      const typeAttr = (field.getAttribute('type') || '').toLowerCase();
      if (['submit', 'button', 'reset', 'hidden', 'image'].includes(typeAttr)) return;
      const fieldName = field.getAttribute('name') || field.getAttribute('id');
      if (!fieldName) return;
      properties[fieldName] = {
        type:
          typeAttr === 'number' || typeAttr === 'range'
            ? 'number'
            : typeAttr === 'checkbox'
              ? 'boolean'
              : 'string',
        description:
          field.getAttribute('toolparamdescription') ||
          field.getAttribute('aria-label') ||
          field.getAttribute('placeholder') ||
          `Value for ${fieldName}`
      };
      if (field.hasAttribute('required')) required.push(fieldName);
    });
    merged.set(name, {
      name,
      description,
      source: 'declarative',
      inputSchema: {
        type: 'object',
        properties,
        ...(required.length ? { required } : {})
      }
    });
  });

  return {
    tools: Array.from(merged.values()),
    pageInfo: {
      title: document.title,
      url: location.href
    }
  };
}

/**
 * Injected directly into the webpage's MAIN world to run `await document.modelContext.executeTool(name, args)`
 */
async function inPageExecuteWebMCPTool(toolName, toolArgs) {
  const mc =
    document.modelContext ||
    window.modelContext ||
    navigator.modelContext ||
    document.modelContextTesting ||
    navigator.modelContextTesting;

  if (mc && typeof mc.executeTool === 'function') {
    let rawRes;
    try {
      rawRes = await mc.executeTool(toolName, toolArgs || {});
    } catch (err) {
      // Fallback 1: JSON string second argument
      try {
        const jsonArg =
          typeof toolArgs === 'string' ? toolArgs : JSON.stringify(toolArgs || {});
        rawRes = await mc.executeTool(toolName, jsonArg);
      } catch (err2) {
        // Fallback 2: single object { name, arguments }
        try {
          rawRes = await mc.executeTool({
            name: toolName,
            arguments: toolArgs || {},
            args: toolArgs || {}
          });
        } catch (err3) {
          throw err;
        }
      }
    }

    try {
      return rawRes !== undefined ? JSON.parse(JSON.stringify(rawRes)) : { status: 'ok' };
    } catch {
      return String(rawRes);
    }
  }

  // Fallback to declarative form if present
  const form =
    document.querySelector(`form[toolname="${CSS.escape(toolName)}"]`) ||
    document.querySelector(`[data-webmcp-tool="${CSS.escape(toolName)}"]`);
  if (form) {
    const updatedFields = [];
    for (const [key, value] of Object.entries(toolArgs || {})) {
      const el = form.querySelector(`[name="${CSS.escape(key)}"], #${CSS.escape(key)}`);
      if (!el) continue;
      el.value = String(value);
      el.dispatchEvent(new Event('input', { bubbles: true }));
      el.dispatchEvent(new Event('change', { bubbles: true }));
      updatedFields.push(key);
    }
    if (form.hasAttribute('toolautosubmit')) {
      form.dispatchEvent(new SubmitEvent('submit', { bubbles: true, cancelable: true }));
    }
    return { status: 'ok', tool: toolName, updatedFields };
  }

  throw new Error(`document.modelContext.executeTool is not available for "${toolName}"`);
}

// Track which frameId owns each discovered tool
const toolFrameMap = new Map();

async function refreshPageTools() {
  const tab = await getActiveTab();
  if (
    !tab ||
    !tab.id ||
    tab.url?.startsWith('chrome://') ||
    tab.url?.startsWith('edge://') ||
    tab.url?.startsWith('about:')
  ) {
    discoveredTools = [];
    pageInfo = { title: tab?.title || 'Browser Page', url: tab?.url || '' };
    renderDiscoveredTools();
    return;
  }

  pageInfo = { title: tab.title || '', url: tab.url || '' };
  const mergedTools = new Map();
  toolFrameMap.clear();

  // 1. Primary: Execute directly in the page's MAIN world across all frames
  try {
    const frameResults = await chrome.scripting.executeScript({
      target: { tabId: tab.id, allFrames: true },
      world: 'MAIN',
      func: inPageGetWebMCPTools
    });

    if (Array.isArray(frameResults)) {
      for (const frameRes of frameResults) {
        const res = frameRes?.result;
        if (frameRes?.frameId === 0 && res?.pageInfo) {
          pageInfo = res.pageInfo;
        }
        if (Array.isArray(res?.tools)) {
          for (const tItem of res.tools) {
            if (tItem && tItem.name && !mergedTools.has(tItem.name)) {
              mergedTools.set(tItem.name, tItem);
              toolFrameMap.set(tItem.name, frameRes.frameId ?? 0);
            }
          }
        }
      }
    }
  } catch (err) {
    // Fallback to content script message if scripting.executeScript was restricted
    try {
      const response = await chrome.tabs
        .sendMessage(tab.id, { type: 'WEBMCP_GET_TOOLS' })
        .catch(() => null);
      if (response && response.ok && Array.isArray(response.tools)) {
        for (const tItem of response.tools) {
          if (tItem && tItem.name && !mergedTools.has(tItem.name)) {
            mergedTools.set(tItem.name, tItem);
            toolFrameMap.set(tItem.name, 0);
          }
        }
        if (response.pageInfo) pageInfo = response.pageInfo;
      }
    } catch {}
  }

  discoveredTools = Array.from(mergedTools.values());
  renderDiscoveredTools();
}

async function callPageWebMCPTool(name, args) {
  const tab = await getActiveTab();
  if (!tab || !tab.id) {
    throw new Error('No active browser tab found.');
  }

  const targetFrameId = toolFrameMap.get(name) ?? 0;

  try {
    const execResults = await chrome.scripting.executeScript({
      target: { tabId: tab.id, frameIds: [targetFrameId] },
      world: 'MAIN',
      func: inPageExecuteWebMCPTool,
      args: [name, args || {}]
    });
    return execResults?.[0]?.result;
  } catch (err) {
    // Fallback to content script bridge
    const response = await chrome.tabs.sendMessage(tab.id, {
      type: 'WEBMCP_CALL_TOOL',
      name,
      args
    });
    if (!response || !response.ok) {
      throw new Error(response?.error || err?.message || `Failed to execute WebMCP tool "${name}"`);
    }
    return response.result;
  }
}

/**
 * Render discovered WebMCP tools badge and popout dialog list
 */
function renderDiscoveredTools() {
  const count = discoveredTools.length;
  toolsStatusDot.classList.toggle('ready', count > 0);
  toolsCountLabel.textContent =
    count === 1
      ? t('toolsActiveSingular')
      : count > 1
        ? t('toolsActivePlural', { count })
        : t('noToolsDetected');

  if (toolsModalBadge) {
    toolsModalBadge.textContent = String(count);
  }

  toolsListEl.innerHTML = '';

  if (count === 0) {
    const emptyEl = document.createElement('div');
    emptyEl.className = 'tool-desc';
    emptyEl.textContent = t('noToolsEmptyDesc');
    toolsListEl.appendChild(emptyEl);
    return;
  }

  discoveredTools.forEach((tTool) => {
    const item = document.createElement('div');
    item.className = 'tool-item';

    const header = document.createElement('div');
    header.className = 'tool-item-header';

    const nameSpan = document.createElement('span');
    nameSpan.className = 'tool-name';
    nameSpan.textContent = tTool.name;

    const badgeSpan = document.createElement('span');
    badgeSpan.className = 'tool-badge';
    badgeSpan.textContent = tTool.source || 'webmcp';

    header.appendChild(nameSpan);
    header.appendChild(badgeSpan);

    const desc = document.createElement('div');
    desc.className = 'tool-desc';
    desc.textContent = tTool.description || t('noDescProvided');

    item.appendChild(header);
    item.appendChild(desc);

    const propKeys = Object.keys(tTool.inputSchema?.properties || {});
    if (propKeys.length > 0) {
      const paramsEl = document.createElement('div');
      paramsEl.className = 'tool-params';
      paramsEl.textContent = `params: ${propKeys.join(', ')}`;
      item.appendChild(paramsEl);
    }

    toolsListEl.appendChild(item);
  });
}

/**
 * Instantiate the Vercel AI SDK model for the user's chosen provider & API key
 */
function createLanguageModel() {
  const provider = currentConfig.provider;
  const apiKey = (currentConfig.apiKeys?.[provider] || '').trim();
  const modelId = currentConfig.model;

  if (!apiKey) {
    throw new Error(t('missingKeyError', { provider }));
  }

  if (provider === 'gemini') {
    const google = createGoogleGenerativeAI({
      apiKey,
      fetch: geminiFetchWithThoughtSignatures
    });
    return google(modelId);
  }

  if (provider === 'claude') {
    const anthropic = createAnthropic({
      apiKey,
      headers: {
        'anthropic-dangerous-direct-browser-access': 'true'
      }
    });
    return anthropic(modelId);
  }

  if (provider === 'openai') {
    const openai = createOpenAI({ apiKey });
    return openai(modelId);
  }

  throw new Error(`Unsupported provider: ${provider}`);
}

/**
 * Convert discovered page WebMCP tools into Vercel AI SDK tools
 */
function buildAISDKTools(runTrace) {
  const sdkTools = {};

  for (const webTool of discoveredTools) {
    const rawSchema =
      webTool.inputSchema && typeof webTool.inputSchema === 'object'
        ? webTool.inputSchema
        : { type: 'object', properties: {} };

    const normalizedSchema = {
      type: 'object',
      properties: rawSchema.properties || {},
      ...(Array.isArray(rawSchema.required) && rawSchema.required.length > 0
        ? { required: rawSchema.required }
        : {})
    };

    sdkTools[webTool.name] = tool({
      description:
        webTool.description || `Execute WebMCP tool ${webTool.name} on the active webpage`,
      parameters: jsonSchema(normalizedSchema),
      execute: async (args) => {
        const startedAtMs = Date.now();
        const toolDebugRecord = {
          toolName: webTool.name,
          source: webTool.source || 'modelContext',
          inputSchema: normalizedSchema,
          arguments: args,
          startedAt: new Date(startedAtMs).toISOString(),
          completedAt: null,
          durationMs: null,
          status: 'running',
          response: null,
          error: null,
          pageInfo: { ...pageInfo },
          provider: currentConfig.provider,
          model: currentConfig.model
        };
        if (runTrace && Array.isArray(runTrace.toolCalls)) {
          runTrace.toolCalls.push(toolDebugRecord);
        }

        const card = appendToolCallCard(webTool.name, args, toolDebugRecord, runTrace?.statusEl);
        try {
          const result = await callPageWebMCPTool(webTool.name, args);
          const completedAtMs = Date.now();
          toolDebugRecord.completedAt = new Date(completedAtMs).toISOString();
          toolDebugRecord.durationMs = completedAtMs - startedAtMs;
          toolDebugRecord.status = 'completed';
          toolDebugRecord.response = result ?? { status: 'success' };

          updateToolCallCardResult(card, toolDebugRecord.response, false, toolDebugRecord);
          // Re-scan tools in case page state changed and dynamically registered new tools
          setTimeout(() => refreshPageTools(), 250);
          return toolDebugRecord.response;
        } catch (err) {
          const completedAtMs = Date.now();
          const errMsg = err?.message || String(err);
          const errObj = { error: errMsg };
          toolDebugRecord.completedAt = new Date(completedAtMs).toISOString();
          toolDebugRecord.durationMs = completedAtMs - startedAtMs;
          toolDebugRecord.status = 'failed';
          toolDebugRecord.error = errMsg;
          toolDebugRecord.response = errObj;

          updateToolCallCardResult(card, errObj, true, toolDebugRecord);
          return errObj;
        }
      }
    });
  }

  return sdkTools;
}

const COPY_DEBUG_SVG = `<svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>`;

/**
 * Format compact inline arguments preview for the main tool call line: e.g. `key: "val", count: 2`
 */
function formatInlineArgsPreview(args) {
  if (!args || typeof args !== 'object') return '';
  const entries = Object.entries(args);
  if (entries.length === 0) return '';
  return entries
    .map(([k, v]) => {
      const valStr = typeof v === 'string' ? JSON.stringify(v) : JSON.stringify(v);
      return `${k}: ${valStr}`;
    })
    .join(', ');
}

/**
 * Format a single-line human-readable summary for the sub-line
 */
function formatOneLineResultSummary(result, isError) {
  if (isError) {
    const errText =
      typeof result === 'string'
        ? result
        : result?.error || JSON.stringify(result);
    return `${t('toolFailed')}: ${String(errText).replace(/\s+/g, ' ').trim()}`;
  }

  if (result === null || result === undefined) {
    return t('toolCompleted');
  }

  if (typeof result === 'string') {
    const cleaned = result.replace(/\s+/g, ' ').trim();
    return cleaned || t('toolCompleted');
  }

  try {
    const compact = JSON.stringify(result).replace(/\s+/g, ' ').trim();
    return compact || t('toolCompleted');
  } catch {
    return String(result);
  }
}

/**
 * Copy text to clipboard and show temporary visual feedback on button
 */
async function copyTextWithButtonFeedback(btn, payloadObj, showLabelText = false) {
  const jsonText = JSON.stringify(payloadObj, null, 2);
  try {
    await navigator.clipboard.writeText(jsonText);
  } catch {
    const ta = document.createElement('textarea');
    ta.value = jsonText;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand('copy');
    document.body.removeChild(ta);
  }

  const prevHtml = btn.innerHTML;
  btn.classList.add('copied');
  btn.innerHTML = showLabelText ? `<span>${t('copiedLabel')}</span>` : '<span>✓</span>';
  setTimeout(() => {
    btn.classList.remove('copied');
    btn.innerHTML = prevHtml;
  }, 1200);
}

/**
 * Extract Input, Output, and Cache-Hit token counts across Gemini, Claude, and OpenAI
 */
function extractTokenMetrics(result) {
  let inputTokens = 0;
  let outputTokens = 0;
  let cachedTokens = 0;

  const usage = result?.usage || {};
  if (typeof usage.promptTokens === 'number') inputTokens = usage.promptTokens;
  else if (typeof usage.inputTokens === 'number') inputTokens = usage.inputTokens;

  if (typeof usage.completionTokens === 'number') outputTokens = usage.completionTokens;
  else if (typeof usage.outputTokens === 'number') outputTokens = usage.outputTokens;

  // Sum cached tokens across steps if multi-step, or fallback to top-level providerMetadata
  const steps = Array.isArray(result?.steps) && result.steps.length > 0 ? result.steps : [result];

  let stepInputSum = 0;
  let stepOutputSum = 0;
  let stepCachedSum = 0;

  for (const step of steps) {
    if (!step) continue;
    const u = step.usage || {};
    if (typeof u.promptTokens === 'number') stepInputSum += u.promptTokens;
    else if (typeof u.inputTokens === 'number') stepInputSum += u.inputTokens;

    if (typeof u.completionTokens === 'number') stepOutputSum += u.completionTokens;
    else if (typeof u.outputTokens === 'number') stepOutputSum += u.outputTokens;

    if (typeof u.cachedInputTokens === 'number') {
      stepCachedSum += u.cachedInputTokens;
    }

    const meta = step.providerMetadata || step.experimental_providerMetadata || {};
    // Anthropic cache read tokens
    const anthropicCache =
      meta?.anthropic?.cacheReadInputTokens ??
      meta?.anthropic?.usage?.cache_read_input_tokens ??
      0;
    // OpenAI cached prompt tokens
    const openaiCache =
      meta?.openai?.cachedPromptTokens ??
      meta?.openai?.usage?.prompt_tokens_details?.cached_tokens ??
      0;
    // Google Gemini cached content token count
    const googleCache =
      meta?.google?.cachedContentTokenCount ??
      meta?.google?.usageMetadata?.cachedContentTokenCount ??
      0;

    stepCachedSum +=
      (Number(anthropicCache) || 0) +
      (Number(openaiCache) || 0) +
      (Number(googleCache) || 0);
  }

  if (!inputTokens && stepInputSum > 0) inputTokens = stepInputSum;
  if (!outputTokens && stepOutputSum > 0) outputTokens = stepOutputSum;
  cachedTokens = stepCachedSum;

  return {
    inputTokens: Number.isFinite(inputTokens) ? inputTokens : 0,
    outputTokens: Number.isFinite(outputTokens) ? outputTokens : 0,
    cachedTokens: Number.isFinite(cachedTokens) ? cachedTokens : 0,
    totalTokens:
      (Number.isFinite(inputTokens) ? inputTokens : 0) +
      (Number.isFinite(outputTokens) ? outputTokens : 0)
  };
}

/**
 * Chat UI Helpers
 */
function appendMessage(role, text) {
  if (welcomeCard) {
    welcomeCard.classList.add('hidden');
  }
  const el = document.createElement('div');
  el.className = `message ${role}`;
  el.textContent = text;
  messagesContainer.appendChild(el);
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
  return el;
}

/**
 * Claude Code style tool call block:
 * 1. Main line: in-progress spinner / status dot + `toolName(args)` + duration + copy debug icon
 * 2. Sub-line: `⎿  one-line summary...` (click to toggle details)
 * 3. Collapsible details: full tool_call input and response JSON
 */
function appendToolCallCard(toolName, args, toolDebugRecord, beforeStatusEl = null) {
  if (welcomeCard) {
    welcomeCard.classList.add('hidden');
  }

  const block = document.createElement('div');
  block.className = 'cc-tool-block';

  // 1. Main line
  const mainLine = document.createElement('div');
  mainLine.className = 'cc-tool-main-line';

  const statusIndicator = document.createElement('span');
  statusIndicator.className = 'cc-status-indicator';
  statusIndicator.innerHTML = `<span class="cc-spinner" aria-label="${t('toolRunning')}"></span>`;

  const callText = document.createElement('div');
  callText.className = 'cc-tool-call-text';

  const fnNameSpan = document.createElement('span');
  fnNameSpan.className = 'cc-tool-fn-name';
  fnNameSpan.textContent = toolName;

  const fnArgsSpan = document.createElement('span');
  fnArgsSpan.className = 'cc-tool-fn-args';
  const inlineArgs = formatInlineArgsPreview(args);
  fnArgsSpan.textContent = inlineArgs ? `(${inlineArgs})` : '()';

  callText.appendChild(fnNameSpan);
  callText.appendChild(fnArgsSpan);

  const durationSpan = document.createElement('span');
  durationSpan.className = 'cc-tool-duration';

  const copyBtn = document.createElement('button');
  copyBtn.type = 'button';
  copyBtn.className = 'cc-copy-debug-btn';
  copyBtn.title = t('copyDebugToolTitle');
  copyBtn.innerHTML = COPY_DEBUG_SVG;
  copyBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    copyTextWithButtonFeedback(
      copyBtn,
      {
        type: 'webmcp_tool_call_debug',
        ...toolDebugRecord
      },
      false
    );
  });

  mainLine.appendChild(statusIndicator);
  mainLine.appendChild(callText);
  mainLine.appendChild(durationSpan);
  mainLine.appendChild(copyBtn);

  // 2. Sub line (lighter text, 1 line with ellipsis, click to expand details)
  const subLine = document.createElement('div');
  subLine.className = 'cc-tool-sub-line';
  subLine.title = t('clickToExpand');

  const branchGlyph = document.createElement('span');
  branchGlyph.className = 'cc-sub-branch';
  branchGlyph.textContent = '⎿';

  const summarySpan = document.createElement('span');
  summarySpan.className = 'cc-sub-summary';
  summarySpan.textContent = t('toolRunning');

  const chevronSpan = document.createElement('span');
  chevronSpan.className = 'cc-sub-chevron';
  chevronSpan.textContent = '▸';

  subLine.appendChild(branchGlyph);
  subLine.appendChild(summarySpan);
  subLine.appendChild(chevronSpan);

  // 3. Expandable Details Drawer (tool_call arguments + response)
  const detailsEl = document.createElement('div');
  detailsEl.className = 'cc-tool-details hidden';

  const inputSection = document.createElement('div');
  inputSection.className = 'cc-detail-section';
  const inputLabel = document.createElement('div');
  inputLabel.className = 'cc-detail-label';
  inputLabel.textContent = `${t('toolInputLabel')} (${toolName})`;
  const inputPre = document.createElement('pre');
  inputPre.className = 'cc-detail-pre cc-input-pre';
  inputPre.textContent = JSON.stringify(args ?? {}, null, 2);
  inputSection.appendChild(inputLabel);
  inputSection.appendChild(inputPre);

  const outputSection = document.createElement('div');
  outputSection.className = 'cc-detail-section';
  const outputLabel = document.createElement('div');
  outputLabel.className = 'cc-detail-label';
  outputLabel.textContent = t('toolOutputLabel');
  const outputPre = document.createElement('pre');
  outputPre.className = 'cc-detail-pre cc-output-pre';
  outputPre.textContent = t('toolRunning');
  outputSection.appendChild(outputLabel);
  outputSection.appendChild(outputPre);

  detailsEl.appendChild(inputSection);
  detailsEl.appendChild(outputSection);

  subLine.addEventListener('click', () => {
    const isHidden = detailsEl.classList.toggle('hidden');
    block.classList.toggle('expanded', !isHidden);
  });

  block.appendChild(mainLine);
  block.appendChild(subLine);
  block.appendChild(detailsEl);

  if (beforeStatusEl && beforeStatusEl.parentNode === messagesContainer) {
    messagesContainer.insertBefore(block, beforeStatusEl);
  } else {
    messagesContainer.appendChild(block);
  }
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
  return block;
}

function updateToolCallCardResult(block, result, isError, toolDebugRecord) {
  const statusIndicator = block.querySelector('.cc-status-indicator');
  const durationSpan = block.querySelector('.cc-tool-duration');
  const summarySpan = block.querySelector('.cc-sub-summary');
  const outputPre = block.querySelector('.cc-output-pre');

  if (statusIndicator) {
    statusIndicator.innerHTML = isError
      ? `<span class="cc-dot-err" title="${t('toolFailed')}"></span>`
      : `<span class="cc-dot-ok" title="${t('toolCompleted')}"></span>`;
  }

  if (durationSpan && typeof toolDebugRecord?.durationMs === 'number') {
    durationSpan.textContent =
      toolDebugRecord.durationMs >= 1000
        ? `${(toolDebugRecord.durationMs / 1000).toFixed(2)}s`
        : `${toolDebugRecord.durationMs}ms`;
  }

  if (summarySpan) {
    summarySpan.textContent = formatOneLineResultSummary(result, isError);
    summarySpan.classList.toggle('is-error', Boolean(isError));
  }

  if (outputPre) {
    outputPre.textContent =
      typeof result === 'string' ? result : JSON.stringify(result, null, 2);
  }

  messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

/**
 * Format duration (e.g. "1s", "2.4s")
 */
function formatElapsedSeconds(ms) {
  const sec = Math.max(0.1, ms / 1000);
  return sec >= 10 ? `${Math.round(sec)}s` : `${sec.toFixed(1).replace(/\.0$/, '')}s`;
}

/**
 * Create the Claude Code style bottom status line:
 * - While working: pulsing `✻ Working… · 2.1s · ↑ 120 ↓ 45 tokens`
 * - When finished: `✻ Worked for 2s · done 10:53 PM · 120 in · 45 out · cache 0` + Copy Debug Info
 */
function createRunStatusLine(runTrace) {
  const statusEl = document.createElement('div');
  statusEl.className = 'cc-status-line is-working';

  const leftEl = document.createElement('div');
  leftEl.className = 'cc-status-left';

  const starEl = document.createElement('span');
  starEl.className = 'cc-claude-star';
  starEl.textContent = '✻';

  const textEl = document.createElement('span');
  textEl.className = 'cc-status-text';
  textEl.textContent = `${t('thinking')}… · 0s`;

  const tokenEl = document.createElement('span');
  tokenEl.className = 'cc-token-inline';
  tokenEl.textContent = `· 0 ${t('tokensInput')} · 0 ${t('tokensOutput')}`;

  leftEl.appendChild(starEl);
  leftEl.appendChild(textEl);
  leftEl.appendChild(tokenEl);

  const copyRunBtn = document.createElement('button');
  copyRunBtn.type = 'button';
  copyRunBtn.className = 'cc-copy-debug-btn hidden';
  copyRunBtn.title = t('copyDebugRunTitle');
  copyRunBtn.innerHTML = `${COPY_DEBUG_SVG}<span>${t('copyDebugRunLabel')}</span>`;
  copyRunBtn.addEventListener('click', () => {
    copyTextWithButtonFeedback(copyRunBtn, runTrace, true);
  });

  statusEl.appendChild(leftEl);
  statusEl.appendChild(copyRunBtn);
  messagesContainer.appendChild(statusEl);
  messagesContainer.scrollTop = messagesContainer.scrollHeight;

  return {
    statusEl,
    textEl,
    tokenEl,
    copyRunBtn
  };
}

function updateRunStatusLineProgress(statusRefs, runStartedAt, usage) {
  if (!statusRefs) return;
  const elapsedStr = formatElapsedSeconds(Date.now() - runStartedAt);
  statusRefs.textEl.textContent = `${t('thinking')}… · ${elapsedStr}`;

  const inTok = usage?.inputTokens || 0;
  const outTok = usage?.outputTokens || 0;
  const cacheTok = usage?.cachedTokens || 0;
  if (inTok > 0 || outTok > 0 || cacheTok > 0) {
    statusRefs.tokenEl.innerHTML = `· ${inTok} ${t('tokensInput')} · ${outTok} ${t('tokensOutput')}${
      cacheTok > 0 ? ` · <span class="cache-hit-text">${t('tokensCacheHit')} ${cacheTok}</span>` : ''
    }`;
  }
}

function finalizeRunStatusLine(statusRefs, runTrace) {
  if (!statusRefs) return;
  const { statusEl, textEl, tokenEl, copyRunBtn } = statusRefs;
  statusEl.classList.remove('is-working');

  const durationStr = formatElapsedSeconds(runTrace.durationMs || 0);
  const doneTimeStr = new Date().toLocaleTimeString([], {
    hour: 'numeric',
    minute: '2-digit'
  });

  textEl.textContent = `${t('crunchedFor', { duration: durationStr })} · ${t('doneAt', { time: doneTimeStr })}`;

  const { inputTokens, outputTokens, cachedTokens } = runTrace.tokenUsage || {
    inputTokens: 0,
    outputTokens: 0,
    cachedTokens: 0
  };

  tokenEl.innerHTML = `· ${inputTokens} ${t('tokensInput')} · ${outputTokens} ${t('tokensOutput')} · <span class="${
    cachedTokens > 0 ? 'cache-hit-text' : ''
  }">${t('tokensCacheHit')} ${cachedTokens}</span>`;

  copyRunBtn.classList.remove('hidden');
  messagesContainer.appendChild(statusEl);
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

/**
 * Handle prompt submission and execute AI agent loop with WebMCP tools
 */
async function handlePromptSubmit(userPrompt) {
  const trimmedPrompt = userPrompt.trim();
  if (!trimmedPrompt || isGenerating) return;

  isGenerating = true;
  sendBtn.disabled = true;
  promptInput.value = '';
  recordPromptInHistory(trimmedPrompt);

  appendMessage('user', trimmedPrompt);
  await refreshPageTools();

  const runStartedAt = Date.now();
  const runTrace = {
    type: 'webmcp_run_debug',
    startedAt: new Date(runStartedAt).toISOString(),
    completedAt: null,
    durationMs: null,
    provider: currentConfig.provider,
    model: currentConfig.model,
    pageInfo: { ...pageInfo },
    discoveredTools: discoveredTools.map((toolItem) => ({
      name: toolItem.name,
      description: toolItem.description,
      source: toolItem.source,
      inputSchema: toolItem.inputSchema
    })),
    userPrompt: trimmedPrompt,
    systemPrompt: '',
    toolCalls: [],
    tokenUsage: {
      inputTokens: 0,
      outputTokens: 0,
      cachedTokens: 0,
      totalTokens: 0
    },
    finalResponse: null,
    error: null
  };

  const statusRefs = createRunStatusLine(runTrace);
  Object.defineProperty(runTrace, 'statusEl', {
    value: statusRefs.statusEl,
    enumerable: false
  });

  const liveTimer = setInterval(() => {
    updateRunStatusLineProgress(statusRefs, runStartedAt, runTrace.tokenUsage);
  }, 200);

  try {
    const model = createLanguageModel();
    const tools = buildAISDKTools(runTrace);

    // Reset per-invocation Gemini tool-call signature queue because conversationHistory
    // stores user/assistant text turns while multi-step tool calls happen inside generateText().
    geminiToolCallSignatures.length = 0;

    conversationHistory.push({
      role: 'user',
      content: trimmedPrompt
    });

    const resolvedLang = resolveLanguage(currentConfig.language);
    const langName = LANGUAGE_LABELS[resolvedLang] || 'English';

    const systemPrompt = [
      `You are webmcp, an AI browser assistant that controls webpages through WebMCP (Web Model Context Protocol) tools.`,
      `Active Webpage Title: "${pageInfo.title || 'Unknown'}"`,
      `Active Webpage URL: "${pageInfo.url || 'Unknown'}"`,
      `Preferred UI Language: ${langName} (${resolvedLang}). Respond in the user's prompt language (or ${langName} by default).`,
      discoveredTools.length > 0
        ? `Available WebMCP tools on this page: ${discoveredTools.map((toolItem) => toolItem.name).join(', ')}. Always use the provided tools when the user asks to perform an action or query data on the webpage.`
        : `No WebMCP tools were detected on this page yet. Inform the user if they try to run a page action, or answer their general question.`
    ].join('\n');

    runTrace.systemPrompt = systemPrompt;

    const result = await generateText({
      model,
      system: systemPrompt,
      messages: conversationHistory,
      tools: Object.keys(tools).length > 0 ? tools : undefined,
      maxSteps: 5,
      onStepFinish: (stepResult) => {
        const stepMetrics = extractTokenMetrics(stepResult);
        runTrace.tokenUsage.inputTokens += stepMetrics.inputTokens;
        runTrace.tokenUsage.outputTokens += stepMetrics.outputTokens;
        runTrace.tokenUsage.cachedTokens += stepMetrics.cachedTokens;
        runTrace.tokenUsage.totalTokens =
          runTrace.tokenUsage.inputTokens + runTrace.tokenUsage.outputTokens;
        updateRunStatusLineProgress(statusRefs, runStartedAt, runTrace.tokenUsage);
      }
    });

    clearInterval(liveTimer);

    const finalReply = result.text?.trim() || t('defaultDoneReply');
    const assistantMsgEl = document.createElement('div');
    assistantMsgEl.className = 'message assistant';
    assistantMsgEl.textContent = finalReply;
    messagesContainer.insertBefore(assistantMsgEl, statusRefs.statusEl);

    const completedAt = Date.now();
    runTrace.completedAt = new Date(completedAt).toISOString();
    runTrace.durationMs = completedAt - runStartedAt;
    const finalMetrics = extractTokenMetrics(result);
    if (finalMetrics.totalTokens > 0 || runTrace.tokenUsage.totalTokens === 0) {
      runTrace.tokenUsage = finalMetrics;
    }
    runTrace.finalResponse = finalReply;

    finalizeRunStatusLine(statusRefs, runTrace);

    conversationHistory.push({
      role: 'assistant',
      content: finalReply
    });
  } catch (err) {
    clearInterval(liveTimer);

    const completedAt = Date.now();
    const errMsg = err?.message || String(err);
    runTrace.completedAt = new Date(completedAt).toISOString();
    runTrace.durationMs = completedAt - runStartedAt;
    runTrace.error = errMsg;

    const errEl = document.createElement('div');
    errEl.className = 'message error';
    errEl.textContent = t('errorPrefix', {
      error: errMsg
    });
    messagesContainer.insertBefore(errEl, statusRefs.statusEl);

    finalizeRunStatusLine(statusRefs, runTrace);
  } finally {
    clearInterval(liveTimer);
    isGenerating = false;
    sendBtn.disabled = false;
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }
}

/**
 * Event Listeners
 */
providerTabs.forEach((tab) => {
  tab.addEventListener('click', async () => {
    currentConfig.apiKeys[currentConfig.provider] = apiKeyInput.value.trim();
    if (modelSelect.value) {
      currentConfig.selectedModels[currentConfig.provider] = modelSelect.value;
    }
    currentConfig.provider = tab.dataset.provider;
    currentConfig.model = currentConfig.selectedModels[currentConfig.provider] || '';
    await persistConfig();
    await renderSetupForm(currentConfig.provider, currentConfig.model);
  });
});

modelSelect.addEventListener('change', async () => {
  currentConfig.model = modelSelect.value;
  currentConfig.selectedModels[currentConfig.provider] = modelSelect.value;
  await persistConfig();
});

if (themeSelect) {
  themeSelect.addEventListener('change', async () => {
    currentConfig.theme = themeSelect.value;
    applyTheme(currentConfig.theme);
    await persistConfig();
  });
}

if (langSelect) {
  langSelect.addEventListener('change', async () => {
    currentConfig.language = langSelect.value;
    applyI18n();
    if (!apiKeyInput.value.trim()) {
      populateModelDropdown([], '', t('enterKeyToLoadModels'));
    }
    await persistConfig();
  });
}

apiKeyInput.addEventListener('input', () => {
  const enteredKey = apiKeyInput.value.trim();
  currentConfig.apiKeys[currentConfig.provider] = enteredKey;
  providerModelsCache[currentConfig.provider] = [];
  clearTimeout(fetchModelsDebounceTimer);
  persistConfig();

  if (!enteredKey) {
    populateModelDropdown([], '', t('enterKeyToLoadModels'));
    return;
  }

  fetchModelsDebounceTimer = setTimeout(() => {
    loadModelsForProvider(
      currentConfig.provider,
      enteredKey,
      currentConfig.selectedModels[currentConfig.provider],
      true
    );
  }, 500);
});

if (refreshModelsBtn) {
  refreshModelsBtn.addEventListener('click', () => {
    const enteredKey = apiKeyInput.value.trim();
    if (!enteredKey) {
      setupError.textContent = t('enterKeyFirstError', {
        provider: PROVIDERS[currentConfig.provider].label
      });
      setupError.classList.remove('hidden');
      return;
    }
    loadModelsForProvider(
      currentConfig.provider,
      enteredKey,
      modelSelect.value || currentConfig.selectedModels[currentConfig.provider],
      true
    );
  });
}

let copyKeyFeedbackTimer = null;

if (copyKeyBtn) {
  copyKeyBtn.addEventListener('click', async () => {
    const keyVal = apiKeyInput.value.trim();
    if (!keyVal) return;

    try {
      await navigator.clipboard.writeText(keyVal);
    } catch {
      const prevType = apiKeyInput.type;
      apiKeyInput.type = 'text';
      apiKeyInput.select();
      document.execCommand('copy');
      apiKeyInput.type = prevType;
    }

    clearTimeout(copyKeyFeedbackTimer);
    const svgIcon = copyKeyBtn.querySelector('.copy-icon-svg');
    const checkGlyph = copyKeyBtn.querySelector('.copy-check-glyph');
    copyKeyBtn.classList.add('copied');
    if (svgIcon) svgIcon.classList.add('hidden');
    if (checkGlyph) checkGlyph.classList.remove('hidden');

    copyKeyFeedbackTimer = setTimeout(() => {
      copyKeyBtn.classList.remove('copied');
      if (svgIcon) svgIcon.classList.remove('hidden');
      if (checkGlyph) checkGlyph.classList.add('hidden');
    }, 1200);
  });
}

settingsToggleBtn.addEventListener('click', async () => {
  const isSetupVisible = !setupView.classList.contains('hidden');
  if (isSetupVisible) {
    const enteredKey = apiKeyInput.value.trim();
    if (enteredKey) {
      currentConfig.apiKeys[currentConfig.provider] = enteredKey;
    }
    if (modelSelect.value) {
      currentConfig.model = modelSelect.value;
      currentConfig.selectedModels[currentConfig.provider] = modelSelect.value;
    }
    await persistConfig();

    if (!hasValidActiveKey()) {
      setupError.textContent = t('enterValidKeyError', {
        provider: PROVIDERS[currentConfig.provider].label
      });
      setupError.classList.remove('hidden');
      return;
    }

    if (!currentConfig.model) {
      const models = await loadModelsForProvider(
        currentConfig.provider,
        currentConfig.apiKeys[currentConfig.provider],
        currentConfig.selectedModels[currentConfig.provider],
        true
      );
      if (!currentConfig.model && models[0]?.id) {
        currentConfig.model = models[0].id;
        currentConfig.selectedModels[currentConfig.provider] = models[0].id;
        await persistConfig();
      }
    }

    if (hasValidActiveKey() && currentConfig.model) {
      showChatView();
      await refreshPageTools();
    }
  } else {
    renderSetupForm(currentConfig.provider, currentConfig.model);
    showSetupView();
  }
});

let refreshFeedbackTimer = null;

refreshToolsBtn.addEventListener('click', async () => {
  clearTimeout(refreshFeedbackTimer);
  const glyphEl = refreshToolsBtn.querySelector('.refresh-glyph');
  if (glyphEl) glyphEl.textContent = '↻';
  refreshToolsBtn.classList.remove('refresh-done');
  refreshToolsBtn.classList.add('is-refreshing');

  if (toolsStatusDot) toolsStatusDot.classList.remove('ready');
  if (toolsCountLabel) toolsCountLabel.textContent = t('scanningTools');
  if (toolsSummaryToggle) toolsSummaryToggle.classList.remove('flash-updated');

  const minSpinPromise = new Promise((resolve) => setTimeout(resolve, 350));
  await Promise.all([refreshPageTools(), minSpinPromise]);

  refreshToolsBtn.classList.remove('is-refreshing');
  refreshToolsBtn.classList.add('refresh-done');
  if (glyphEl) glyphEl.textContent = '✓';

  if (toolsSummaryToggle) {
    void toolsSummaryToggle.offsetWidth;
    toolsSummaryToggle.classList.add('flash-updated');
  }

  refreshFeedbackTimer = setTimeout(() => {
    refreshToolsBtn.classList.remove('refresh-done');
    if (glyphEl) glyphEl.textContent = '↻';
    if (toolsSummaryToggle) toolsSummaryToggle.classList.remove('flash-updated');
  }, 1000);
});

function openToolsModal() {
  if (toolsModalBackdrop) {
    toolsModalBackdrop.classList.remove('hidden');
  }
}

function closeToolsModal() {
  if (toolsModalBackdrop) {
    toolsModalBackdrop.classList.add('hidden');
  }
}

toolsSummaryToggle.addEventListener('click', () => {
  openToolsModal();
});

if (closeToolsModalBtn) {
  closeToolsModalBtn.addEventListener('click', () => {
    closeToolsModal();
  });
}

if (toolsModalBackdrop) {
  toolsModalBackdrop.addEventListener('click', (e) => {
    if (e.target === toolsModalBackdrop) {
      closeToolsModal();
    }
  });
}

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && toolsModalBackdrop && !toolsModalBackdrop.classList.contains('hidden')) {
    closeToolsModal();
  }
});

if (newChatBtn) {
  newChatBtn.addEventListener('click', () => {
    conversationHistory = [];
    geminiToolCallSignatures.length = 0;
    promptHistoryIndex = -1;
    currentPromptDraft = '';
    messagesContainer.innerHTML = '';
    if (welcomeCard) {
      welcomeCard.classList.remove('hidden');
      messagesContainer.appendChild(welcomeCard);
    }
    promptInput.value = '';
    promptInput.focus();
  });
}

promptForm.addEventListener('submit', (e) => {
  e.preventDefault();
  handlePromptSubmit(promptInput.value);
});

promptInput.addEventListener('input', () => {
  if (promptHistoryIndex === -1) {
    currentPromptDraft = promptInput.value;
  }
});

promptInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    handlePromptSubmit(promptInput.value);
    return;
  }

  if (e.key === 'ArrowUp' && !e.shiftKey && !e.altKey && !e.metaKey && !e.ctrlKey) {
    const val = promptInput.value;
    const isMultiLine = val.includes('\n');
    const cursorAtFirstLine = !val.slice(0, promptInput.selectionStart).includes('\n');
    if (promptHistory.length > 0 && (!isMultiLine || cursorAtFirstLine || promptHistoryIndex !== -1)) {
      e.preventDefault();
      if (promptHistoryIndex === -1) {
        currentPromptDraft = val;
        promptHistoryIndex = promptHistory.length - 1;
      } else if (promptHistoryIndex > 0) {
        promptHistoryIndex -= 1;
      }
      promptInput.value = promptHistory[promptHistoryIndex];
      const len = promptInput.value.length;
      promptInput.setSelectionRange(len, len);
    }
    return;
  }

  if (e.key === 'ArrowDown' && !e.shiftKey && !e.altKey && !e.metaKey && !e.ctrlKey) {
    if (promptHistoryIndex !== -1) {
      const val = promptInput.value;
      const isMultiLine = val.includes('\n');
      const cursorAtLastLine = !val.slice(promptInput.selectionEnd).includes('\n');
      if (!isMultiLine || cursorAtLastLine) {
        e.preventDefault();
        if (promptHistoryIndex < promptHistory.length - 1) {
          promptHistoryIndex += 1;
          promptInput.value = promptHistory[promptHistoryIndex];
        } else {
          promptHistoryIndex = -1;
          promptInput.value = currentPromptDraft;
        }
        const len = promptInput.value.length;
        promptInput.setSelectionRange(len, len);
      }
    }
  }
});

// Keep shared prompt history synchronized across multiple open sidepanel instances
chrome.storage?.onChanged?.addListener((changes, areaName) => {
  if (areaName === 'local' && changes[PROMPT_HISTORY_STORAGE_KEY]) {
    const newValue = changes[PROMPT_HISTORY_STORAGE_KEY].newValue;
    if (Array.isArray(newValue)) {
      promptHistory = newValue
        .filter((item) => typeof item === 'string' && item.trim().length > 0)
        .slice(-MAX_PROMPT_HISTORY);
      if (promptHistoryIndex >= promptHistory.length) {
        promptHistoryIndex = -1;
      }
    }
  }
});

// Auto-refresh tools when the user switches active tabs or page tools update
chrome.tabs?.onActivated?.addListener(() => {
  refreshPageTools();
});

chrome.tabs?.onUpdated?.addListener((tabId, changeInfo) => {
  if (changeInfo.status === 'complete') {
    refreshPageTools();
  }
});

chrome.runtime.onMessage.addListener((message) => {
  if (message?.type === 'WEBMCP_TAB_TOOLS_UPDATED') {
    refreshPageTools();
  }
});

init();
