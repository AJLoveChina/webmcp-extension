/**
 * MAIN World WebMCP Page Bridge
 *
 * Runs in the page's MAIN execution world and calls `document.modelContext.getTools()`
 * and `document.modelContext.executeTool()` directly (without overwriting `document.modelContext`),
 * plus discovers declarative `<form toolname="...">` tools if present.
 */
(function () {
  if (window.__WEBMCP_BRIDGE_V2_INITIALIZED__) return;
  window.__WEBMCP_BRIDGE_V2_INITIALIZED__ = true;

  // Clean up any stale proxy left on document.modelContext or navigator.modelContext by earlier builds
  try {
    if (
      document.modelContext &&
      document.modelContext.getTools &&
      document.modelContext.getTools.__webmcpProxy
    ) {
      delete document.modelContext;
    }
  } catch (e) {}

  function sanitizeSchema(schema) {
    if (!schema) {
      return { type: 'object', properties: {} };
    }
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

  /**
   * Scan DOM for Declarative WebMCP tools: <form toolname="..." tooldescription="...">
   */
  function getDeclarativeFormTools() {
    const forms = document.querySelectorAll('form[toolname], [data-webmcp-tool]');
    const tools = [];

    forms.forEach((form) => {
      const name = form.getAttribute('toolname') || form.getAttribute('data-webmcp-tool');
      if (!name) return;
      const description =
        form.getAttribute('tooldescription') ||
        form.getAttribute('data-webmcp-description') ||
        form.getAttribute('aria-label') ||
        `Submit the ${name} form`;
      const autoSubmit = form.hasAttribute('toolautosubmit') || form.getAttribute('data-webmcp-autosubmit') === 'true';

      const properties = {};
      const required = [];

      const fields = form.querySelectorAll('input, select, textarea');
      fields.forEach((field) => {
        const typeAttr = (field.getAttribute('type') || '').toLowerCase();
        if (['submit', 'button', 'reset', 'hidden', 'image'].includes(typeAttr)) return;

        const fieldName = field.getAttribute('name') || field.getAttribute('id');
        if (!fieldName) return;

        let paramDesc =
          field.getAttribute('toolparamdescription') ||
          field.getAttribute('aria-description') ||
          field.getAttribute('aria-label') ||
          '';

        if (!paramDesc && field.id) {
          const label = form.querySelector(`label[for="${CSS.escape(field.id)}"]`);
          if (label) paramDesc = label.textContent.trim();
        }
        if (!paramDesc) {
          const parentLabel = field.closest('label');
          if (parentLabel) paramDesc = parentLabel.textContent.trim();
        }
        if (!paramDesc && field.getAttribute('placeholder')) {
          paramDesc = field.getAttribute('placeholder');
        }

        let propSchema = { type: 'string', description: paramDesc || `Value for ${fieldName}` };

        if (field.tagName.toLowerCase() === 'select') {
          const options = Array.from(field.querySelectorAll('option'))
            .map((opt) => opt.value)
            .filter((v) => v !== '');
          if (options.length > 0) {
            propSchema.enum = options;
          }
        } else if (typeAttr === 'number' || typeAttr === 'range') {
          propSchema.type = 'number';
        } else if (typeAttr === 'checkbox') {
          propSchema.type = 'boolean';
        } else if (typeAttr === 'radio') {
          if (properties[fieldName]) {
            if (!properties[fieldName].enum) properties[fieldName].enum = [];
            if (field.value && !properties[fieldName].enum.includes(field.value)) {
              properties[fieldName].enum.push(field.value);
            }
            return;
          }
          propSchema.enum = field.value ? [field.value] : undefined;
        }

        properties[fieldName] = propSchema;
        if (field.hasAttribute('required') && !required.includes(fieldName)) {
          required.push(fieldName);
        }
      });

      tools.push({
        name,
        description,
        autoSubmit,
        source: 'declarative',
        inputSchema: {
          type: 'object',
          properties,
          ...(required.length > 0 ? { required } : {})
        }
      });
    });

    return tools;
  }

  /**
   * Execute a Declarative WebMCP form tool by populating fields and optionally submitting
   */
  async function executeDeclarativeFormTool(name, args = {}) {
    const form =
      document.querySelector(`form[toolname="${CSS.escape(name)}"]`) ||
      document.querySelector(`[data-webmcp-tool="${CSS.escape(name)}"]`);

    if (!form) {
      throw new Error(`WebMCP tool "${name}" not found.`);
    }

    const updatedFields = [];

    for (const [key, value] of Object.entries(args || {})) {
      const elements = form.querySelectorAll(`[name="${CSS.escape(key)}"], #${CSS.escape(key)}`);
      if (!elements.length) continue;

      const first = elements[0];
      const typeAttr = (first.getAttribute('type') || '').toLowerCase();

      if (typeAttr === 'radio') {
        elements.forEach((el) => {
          if (String(el.value) === String(value)) {
            el.checked = true;
            el.dispatchEvent(new Event('input', { bubbles: true }));
            el.dispatchEvent(new Event('change', { bubbles: true }));
          }
        });
      } else if (typeAttr === 'checkbox') {
        first.checked = Boolean(value);
        first.dispatchEvent(new Event('input', { bubbles: true }));
        first.dispatchEvent(new Event('change', { bubbles: true }));
      } else {
        const proto =
          first.tagName === 'TEXTAREA'
            ? window.HTMLTextAreaElement.prototype
            : first.tagName === 'SELECT'
              ? window.HTMLSelectElement.prototype
              : window.HTMLInputElement.prototype;
        const nativeSetter = Object.getOwnPropertyDescriptor(proto, 'value')?.set;
        if (nativeSetter) {
          nativeSetter.call(first, String(value));
        } else {
          first.value = String(value);
        }
        first.dispatchEvent(new Event('input', { bubbles: true }));
        first.dispatchEvent(new Event('change', { bubbles: true }));
      }
      updatedFields.push(key);
    }

    const autoSubmit = form.hasAttribute('toolautosubmit') || form.getAttribute('data-webmcp-autosubmit') === 'true';
    if (autoSubmit) {
      const submitEvent = new SubmitEvent('submit', { bubbles: true, cancelable: true });
      const notPrevented = form.dispatchEvent(submitEvent);
      if (notPrevented && form.getAttribute('action')) {
        form.submit();
      }
      return {
        status: 'submitted',
        tool: name,
        updatedFields,
        message: `Form "${name}" filled (${updatedFields.join(', ')}) and submitted.`
      };
    }

    return {
      status: 'filled_awaiting_user_submit',
      tool: name,
      updatedFields,
      message: `Form "${name}" filled (${updatedFields.join(', ')}). Awaiting user confirmation to submit.`
    };
  }

  /**
   * Normalize any tool list format returned by `await document.modelContext.getTools()`
   */
  function extractToolsArray(rawResult) {
    if (!rawResult) return [];
    if (Array.isArray(rawResult)) return rawResult;
    if (typeof rawResult === 'string') {
      try {
        const parsed = JSON.parse(rawResult);
        return extractToolsArray(parsed);
      } catch {
        return [];
      }
    }
    if (Array.isArray(rawResult.tools)) return rawResult.tools;
    if (rawResult instanceof Map) return Array.from(rawResult.values());
    if (typeof rawResult[Symbol.iterator] === 'function') return Array.from(rawResult);
    if (typeof rawResult === 'object') {
      return Object.entries(rawResult).map(([key, val]) =>
        val && typeof val === 'object' ? { name: val.name || key, ...val } : { name: key }
      );
    }
    return [];
  }

  /**
   * Collect tools using `await document.modelContext.getTools()`
   */
  async function collectAllTools() {
    const merged = new Map();

    // 1. Primary: await document.modelContext.getTools()
    if (document.modelContext && typeof document.modelContext.getTools === 'function') {
      try {
        const rawTools = await document.modelContext.getTools();
        const toolsList = extractToolsArray(rawTools);
        for (const t of toolsList) {
          if (t && t.name) {
            merged.set(String(t.name), {
              name: String(t.name),
              description: String(t.description || ''),
              inputSchema: sanitizeSchema(t.inputSchema || t.parameters || t.schema),
              source: 'modelContext'
            });
          }
        }
      } catch (err) {
        console.warn('[webmcp] document.modelContext.getTools() error:', err);
      }
    }

    // 2. Also include any declarative <form toolname="..."> tools not already returned
    const declarativeTools = getDeclarativeFormTools();
    for (const dt of declarativeTools) {
      if (!merged.has(dt.name)) {
        merged.set(dt.name, dt);
      }
    }

    return Array.from(merged.values());
  }

  /**
   * Execute tool using `await document.modelContext.executeTool()`
   */
  async function executeWebMCPTool(name, args = {}) {
    if (document.modelContext && typeof document.modelContext.executeTool === 'function') {
      try {
        return await document.modelContext.executeTool(name, args || {});
      } catch (err) {
        // Fallback if executeTool expects a JSON string argument or single object { name, arguments }
        try {
          const jsonArg = typeof args === 'string' ? args : JSON.stringify(args || {});
          return await document.modelContext.executeTool(name, jsonArg);
        } catch (err2) {
          throw err;
        }
      }
    }

    return await executeDeclarativeFormTool(name, args);
  }

  let notifyTimer = null;
  function notifyToolsChanged() {
    clearTimeout(notifyTimer);
    notifyTimer = setTimeout(async () => {
      const tools = await collectAllTools();
      window.postMessage(
        {
          source: 'webmcp-page-bridge',
          type: 'WEBMCP_TOOLS_CHANGED',
          tools
        },
        '*'
      );
    }, 100);
  }

  // Listen for requests from the isolated content script
  window.addEventListener('message', async (event) => {
    if (event.source !== window || !event.data || event.data.source !== 'webmcp-content-script') {
      return;
    }

    const { requestId, action, payload } = event.data;

    if (action === 'LIST_TOOLS') {
      try {
        const tools = await collectAllTools();
        window.postMessage(
          {
            source: 'webmcp-page-bridge',
            requestId,
            success: true,
            data: {
              tools,
              pageInfo: {
                title: document.title,
                url: location.href
              }
            }
          },
          '*'
        );
      } catch (err) {
        window.postMessage(
          {
            source: 'webmcp-page-bridge',
            requestId,
            success: false,
            error: err?.message || String(err)
          },
          '*'
        );
      }
    } else if (action === 'EXECUTE_TOOL') {
      try {
        const rawResult = await executeWebMCPTool(payload.name, payload.args);
        let safeResult = rawResult;
        try {
          safeResult =
            rawResult !== undefined ? JSON.parse(JSON.stringify(rawResult)) : { status: 'ok' };
        } catch {
          safeResult = String(rawResult);
        }
        window.postMessage(
          {
            source: 'webmcp-page-bridge',
            requestId,
            success: true,
            data: safeResult
          },
          '*'
        );
      } catch (err) {
        window.postMessage(
          {
            source: 'webmcp-page-bridge',
            requestId,
            success: false,
            error: err?.message || String(err)
          },
          '*'
        );
      }
    }
  });

  // Observe DOM changes and also re-check tools on load
  window.addEventListener('DOMContentLoaded', () => {
    notifyToolsChanged();
    const observer = new MutationObserver((mutations) => {
      for (const m of mutations) {
        if (m.type === 'childList' && m.addedNodes.length > 0) {
          notifyToolsChanged();
          break;
        }
      }
    });
    if (document.documentElement) {
      observer.observe(document.documentElement, { childList: true, subtree: true });
    }
  });

  window.addEventListener('load', () => {
    notifyToolsChanged();
    setTimeout(notifyToolsChanged, 500);
    setTimeout(notifyToolsChanged, 1500);
  });
})();
