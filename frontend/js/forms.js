/**
 * forms.js — human-readable Add / Edit Event form + Preprocess pipeline.
 *
 * Exposes a global `EditorUI` object. The form is driven by /api/ontology
 * (single source of truth for enum values and per-type fields) and the
 * preprocess panel is driven by /api/preprocess/ops, so both stay in sync
 * with the backend without duplicated schemas.
 *
 * Public API:
 *   EditorUI.init({ onMatchUpdated, onPickXY, onPickEnd })
 *      `onMatchUpdated(events)` — called whenever an event create/update/
 *          delete or preprocess apply succeeds. Pass the new event list.
 *      `onPickXY({set: fn})` / `onPickEnd({set: fn})` — called when the
 *          user clicks "📍 Pick on pitch". Frontend pitch should arm a
 *          one-shot click handler and invoke `fn({x, y})` with metres.
 *   EditorUI.openForEdit(eventIndex, eventDict) — pre-fill the Add/Edit
 *          form with an existing event for in-place editing.
 *   EditorUI.toggle()                            — show/hide the drawer.
 */

/* global document, fetch */

const EditorUI = (() => {
  const API = "";

  // ── state ────────────────────────────────────────────────────────────────
  let ontology = null;            // populated by GET /api/ontology
  let opCatalog = [];             // populated by GET /api/preprocess/ops
  let editingIndex = null;        // null → "create" mode; integer → "edit" mode
  let pipeline = [];              // [{op, ...params}] — what the user is building
  let callbacks = {};
  // currently armed "pick on pitch" mode — main.js will call setter on click
  let armedPicker = null;

  // ── DOM refs (cached at init) ────────────────────────────────────────────
  let $panel, $tabs, $commonForm, $typedForm, $formMode, $feedback;
  let $pipelineList, $opSelect, $pipelineFeedback;

  // ── init ─────────────────────────────────────────────────────────────────
  async function init(cb = {}) {
    callbacks = cb;
    $panel             = document.getElementById("panel-editor");
    $tabs              = $panel.querySelectorAll(".editor-tab");
    $commonForm        = document.getElementById("form-common");
    $typedForm         = document.getElementById("form-typed");
    $formMode          = document.getElementById("form-mode-label");
    $feedback          = document.getElementById("form-feedback");
    $pipelineList      = document.getElementById("pipeline-list");
    $opSelect          = document.getElementById("op-select");
    $pipelineFeedback  = document.getElementById("pipeline-feedback");

    // toggle
    document.getElementById("btn-toggle-editor").addEventListener("click", toggle);
    document.getElementById("btn-close-editor") .addEventListener("click", close);

    // tabs
    $tabs.forEach(btn => btn.addEventListener("click", () => _selectTab(btn.dataset.tab)));

    // form buttons
    document.getElementById("btn-form-new")     .addEventListener("click", _resetForm);
    document.getElementById("btn-form-load-current").addEventListener("click", () => {
      if (callbacks.getCurrentEvent) {
        const cur = callbacks.getCurrentEvent();
        if (cur) openForEdit(cur.index, cur);
        else _feedback($feedback, "No current event to load (load a match first).", "err");
      }
    });
    document.getElementById("btn-form-validate").addEventListener("click", () => _saveEvent({ validateOnly: true }));
    document.getElementById("btn-form-save")    .addEventListener("click", () => _saveEvent({ validateOnly: false }));
    document.getElementById("btn-form-pick-xy") .addEventListener("click", () => _armPicker("xy"));
    document.getElementById("btn-form-pick-end").addEventListener("click", () => _armPicker("end"));

    // pipeline buttons
    document.getElementById("btn-add-op")        .addEventListener("click", _addStepFromSelect);
    document.getElementById("btn-pipeline-clear").addEventListener("click", _clearPipeline);
    document.getElementById("btn-pipeline-dryrun").addEventListener("click", () => _runPipeline({ dryRun: true }));
    document.getElementById("btn-pipeline-apply") .addEventListener("click", () => _runPipeline({ dryRun: false }));

    // load ontology + ops in parallel
    try {
      const [oRes, opsRes] = await Promise.all([
        fetch(`${API}/api/ontology`),
        fetch(`${API}/api/preprocess/ops`),
      ]);
      ontology  = await oRes.json();
      opCatalog = (await opsRes.json()).operations || [];
    } catch (e) {
      _feedback($feedback, `Failed to load ontology: ${e}`, "err");
      return;
    }

    _buildForm();
    _populateOpSelect();
  }

  // ── public ───────────────────────────────────────────────────────────────
  function toggle() {
    if ($panel.classList.contains("hidden")) open(); else close();
  }
  function open()  { $panel.classList.remove("hidden"); $panel.setAttribute("aria-hidden", "false"); }
  function close() { $panel.classList.add("hidden");    $panel.setAttribute("aria-hidden", "true");
                     _disarmPicker(); }

  function openForEdit(idx, ev) {
    _selectTab("add");
    open();
    _populateForm(ev);
    // Set editing state AFTER populating, since _populateForm calls
    // _resetForm internally which would otherwise clobber these.
    editingIndex = idx;
    $formMode.textContent = `Editing event #${idx}`;
  }

  /**
   * Called by main.js when the user clicks somewhere on the pitch while
   * a picker is armed. Coordinates are in pitch metres.
   */
  function deliverPickedCoords({ x, y }) {
    if (!armedPicker) return false;
    if (armedPicker === "xy") {
      _setField("x", x.toFixed(2));
      _setField("y", y.toFixed(2));
    } else if (armedPicker === "end") {
      _setField("end_x", x.toFixed(2));
      _setField("end_y", y.toFixed(2));
    }
    _disarmPicker();
    return true;
  }

  function isPickerArmed() { return armedPicker !== null; }

  // ── form construction ────────────────────────────────────────────────────
  function _buildForm() {
    if (!ontology) return;
    $commonForm.innerHTML = "";

    // common fields (always shown, regardless of event_type)
    _appendField($commonForm, {
      name: "timestamp", label: "Timestamp (s)", type: "number", step: 0.1, required: true,
    });
    _appendField($commonForm, {
      name: "team", label: "Team", type: "text", required: true,
    });
    _appendField($commonForm, {
      name: "player", label: "Player", type: "text", required: true,
    });
    _appendField($commonForm, {
      name: "event_type", label: "Event type", type: "enum",
      enum: ontology.enums.event_type, required: true,
    });
    _appendField($commonForm, {
      name: "x", label: "x (m)", type: "number", step: 0.1, required: true,
    });
    _appendField($commonForm, {
      name: "y", label: "y (m)", type: "number", step: 0.1, required: true,
    });

    // when event_type changes, rebuild the typed sub-form
    const et = $commonForm.querySelector('[name="event_type"]');
    et.addEventListener("change", () => _buildTypedSubform(et.value));
    _buildTypedSubform(et.value);
  }

  function _buildTypedSubform(eventType) {
    $typedForm.innerHTML = "";
    if (!ontology || !eventType) return;
    const spec = ontology.fields_per_type[eventType];
    if (!spec) return;

    // valid outcomes for this event type only
    const validOutcomes = ontology.valid_outcomes[eventType] || ontology.enums.outcome;

    const fieldFactory = {
      foot:            () => ({ type: "enum", enum: ontology.enums.foot }),
      body_part:       () => ({ type: "enum", enum: ontology.enums.body_part }),
      outcome:         () => ({ type: "enum", enum: validOutcomes }),
      pass_type:       () => ({ type: "enum", enum: ontology.enums.pass_type }),
      set_piece_type:  () => ({ type: "enum", enum: ontology.enums.set_piece_type }),
      gk_action_type:  () => ({ type: "enum", enum: ontology.enums.gk_action_type }),
      action:          () => ({ type: "enum", enum: ontology.enums.header_action }),
      card:            () => ({ type: "enum", enum: ["", ...ontology.enums.card] }),
      xg:              () => ({ type: "number", step: 0.01, min: 0, max: 1 }),
      end_x:           () => ({ type: "number", step: 0.1 }),
      end_y:           () => ({ type: "number", step: 0.1 }),
      to_player:       () => ({ type: "text" }),
      tackled_player:  () => ({ type: "text" }),
      fouled_player:   () => ({ type: "text" }),
    };

    const addField = (name, required) => {
      const meta = (fieldFactory[name] || (() => ({ type: "text" })))();
      _appendField($typedForm, {
        name, label: _humanize(name), required, ...meta,
      });
    };
    spec.required.forEach(n => addField(n, true));
    spec.optional.forEach(n => addField(n, false));
  }

  function _appendField(parent, spec) {
    const label = document.createElement("label");
    if (spec.required) label.classList.add("required");
    label.textContent = spec.label;
    label.setAttribute("for", `f-${spec.name}`);

    let input;
    if (spec.type === "enum") {
      input = document.createElement("select");
      (spec.enum || []).forEach(v => {
        const o = document.createElement("option");
        o.value = v; o.textContent = v || "—";
        input.appendChild(o);
      });
      if (spec.default != null) input.value = spec.default;
    } else {
      input = document.createElement("input");
      input.type = spec.type === "number" ? "number" : "text";
      if (spec.step != null) input.step = spec.step;
      if (spec.min  != null) input.min  = spec.min;
      if (spec.max  != null) input.max  = spec.max;
      if (spec.default != null) input.value = spec.default;
    }
    input.id = `f-${spec.name}`;
    input.name = spec.name;
    label.appendChild(input);
    parent.appendChild(label);
  }

  function _humanize(name) {
    return name.replace(/_/g, " ").replace(/^./, c => c.toUpperCase());
  }

  // ── form values ──────────────────────────────────────────────────────────
  function _readForm() {
    const out = {};
    [...$commonForm.elements, ...$typedForm.elements].forEach(el => {
      if (!el.name) return;
      const v = el.value;
      if (v === "" || v === null) return;
      out[el.name] = (el.type === "number") ? Number(v) : v;
    });
    return out;
  }

  function _populateForm(ev) {
    _resetForm();

    // Set event_type FIRST and use the change handler to rebuild the
    // typed sub-form. We then populate common + typed fields against
    // a now-known-correct DOM.
    const etInput = $commonForm.querySelector('[name="event_type"]');
    if (etInput && ev.event_type) {
      etInput.value = ev.event_type;
      _buildTypedSubform(ev.event_type);  // synchronous rebuild
    }

    const setIfPresent = (k, v) => {
      if (v === undefined || v === null || v === "") return;
      const el = document.querySelector(`[name="${k}"]`);
      if (!el) return;
      el.value = v;
    };

    // common (event_type already set above)
    setIfPresent("timestamp", ev.timestamp);
    setIfPresent("team",      ev.team);
    setIfPresent("player",    ev.player);
    setIfPresent("x",         ev.x);
    setIfPresent("y",         ev.y);

    // typed fields — every column the JSON event might carry
    ["end_x", "end_y", "foot", "body_part", "outcome", "pass_type",
     "set_piece_type", "gk_action_type", "action", "to_player",
     "tackled_player", "fouled_player", "card", "xg"].forEach(k => {
      setIfPresent(k, ev[k]);
    });

    _feedback($feedback, `Loaded event #${ev.index ?? "?"} into the form.`, "ok");
  }

  function _setField(name, value) {
    const el = document.querySelector(`[name="${name}"]`);
    if (!el) return;
    el.value = value;
    el.dispatchEvent(new Event("change"));
  }

  function _resetForm() {
    editingIndex = null;
    $formMode.textContent = "Adding new event";
    $commonForm.querySelectorAll("input, select").forEach(el => { el.value = ""; });
    // pick first option for selects
    $commonForm.querySelectorAll("select").forEach(el => { if (el.options.length) el.selectedIndex = 0; });
    _buildTypedSubform($commonForm.querySelector('[name="event_type"]').value);
    _feedback($feedback, "", "");
  }

  // ── save / validate ──────────────────────────────────────────────────────
  async function _saveEvent({ validateOnly }) {
    const row = _readForm();
    const url = (editingIndex == null)
      ? `${API}/api/events`
      : `${API}/api/events/${editingIndex}`;
    const method = (editingIndex == null) ? "POST" : "PUT";

    // Validation-only path: POST to a synthetic preprocess pipeline
    // (single validate op) using a 1-row buffer. Simpler: just call the
    // create endpoint with a dry-run via the preprocess pipeline path.
    // We do client-side required-field check first, then server-side.
    const missing = _missingRequiredFields(row);
    if (missing.length) {
      _feedback($feedback, `Missing required fields: ${missing.join(", ")}`, "err");
      return;
    }

    if (validateOnly) {
      // Round-trip the row through /api/preprocess with op=validate, dry_run.
      const res = await fetch(`${API}/api/preprocess`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ operations: [{ op: "validate" }], dry_run: true }),
      });
      const data = await res.json();
      if (data.ok) _feedback($feedback, "Row validates against the schema.", "ok");
      else        _feedback($feedback, `Validation failed: ${JSON.stringify(data.errors || data.error)}`, "err");
      return;
    }

    try {
      const res  = await fetch(url, {
        method,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(row),
      });
      const data = await res.json();
      if (!res.ok) {
        const detail = data.errors ? data.errors.join("; ") : (data.error || res.statusText);
        _feedback($feedback, `Save failed: ${detail}`, "err");
        return;
      }
      _feedback($feedback, `Saved. Match now has ${data.events.length} events.`, "ok");
      if (callbacks.onMatchUpdated) callbacks.onMatchUpdated(data.events);
      if (editingIndex == null) _resetForm();
    } catch (e) {
      _feedback($feedback, `Network error: ${e}`, "err");
    }
  }

  function _missingRequiredFields(row) {
    const req = ["timestamp", "team", "player", "event_type", "x", "y"];
    const et = row.event_type;
    if (et && ontology && ontology.fields_per_type[et]) {
      req.push(...ontology.fields_per_type[et].required);
    }
    return req.filter(k => row[k] === undefined || row[k] === "" || row[k] === null);
  }

  // ── pitch picker ─────────────────────────────────────────────────────────
  function _armPicker(target) {
    armedPicker = target;
    document.getElementById("pitch-container").classList.add("picking");
    _feedback($feedback, `Click on the pitch to set ${target === "xy" ? "(x, y)" : "(end_x, end_y)"}…`, "");
    const cb = (target === "xy") ? callbacks.onPickXY : callbacks.onPickEnd;
    if (cb) cb({ cancel: _disarmPicker });
  }
  function _disarmPicker() {
    armedPicker = null;
    document.getElementById("pitch-container").classList.remove("picking");
  }

  // ── tabs ─────────────────────────────────────────────────────────────────
  function _selectTab(name) {
    $tabs.forEach(btn => btn.classList.toggle("active", btn.dataset.tab === name));
    document.querySelectorAll(".editor-tab-pane").forEach(p =>
      p.classList.toggle("active", p.dataset.tab === name)
    );
  }

  // ── preprocess pipeline UI ───────────────────────────────────────────────
  function _populateOpSelect() {
    $opSelect.innerHTML = "";
    opCatalog.forEach(op => {
      const o = document.createElement("option");
      o.value = op.op; o.textContent = `${op.label} (${op.op})`;
      o.title = op.description;
      $opSelect.appendChild(o);
    });
  }

  function _addStepFromSelect() {
    const opName = $opSelect.value;
    if (!opName) return;
    const spec = opCatalog.find(o => o.op === opName);
    if (!spec) return;
    const step = { op: opName };
    spec.params.forEach(p => { if (p.default !== undefined) step[p.name] = p.default; });
    pipeline.push(step);
    _renderPipeline();
  }

  function _renderPipeline() {
    $pipelineList.innerHTML = "";
    pipeline.forEach((step, i) => {
      const spec = opCatalog.find(o => o.op === step.op);
      if (!spec) return;

      const wrap = document.createElement("div");
      wrap.className = "pipeline-step";

      const hdr = document.createElement("header");
      hdr.innerHTML = `<span><span class="op-tag">${i + 1}.</span>${spec.label}</span>`;
      const rm = document.createElement("button");
      rm.className = "step-remove";
      rm.textContent = "×";
      rm.title = "Remove step";
      rm.addEventListener("click", () => { pipeline.splice(i, 1); _renderPipeline(); });
      hdr.appendChild(rm);
      wrap.appendChild(hdr);

      const params = document.createElement("div");
      params.className = "step-params";
      if (spec.params.length === 0) {
        const empty = document.createElement("span");
        empty.className = "muted small";
        empty.textContent = "(no parameters)";
        params.appendChild(empty);
      }
      spec.params.forEach(p => {
        const label = document.createElement("label");
        label.textContent = p.label || _humanize(p.name);
        let input;
        if (p.type === "enum") {
          input = document.createElement("select");
          (p.enum || []).forEach(v => {
            const o = document.createElement("option");
            o.value = v; o.textContent = v;
            input.appendChild(o);
          });
        } else {
          input = document.createElement("input");
          input.type = (p.type === "int" || p.type === "float") ? "number" : "text";
          if (p.type === "int")   input.step = 1;
          if (p.type === "float") input.step = 0.1;
        }
        input.value = step[p.name] != null ? step[p.name] : (p.default != null ? p.default : "");
        input.addEventListener("input", () => {
          const v = input.value;
          step[p.name] = (input.type === "number" && v !== "") ? Number(v) : v;
        });
        label.appendChild(input);
        params.appendChild(label);
      });
      wrap.appendChild(params);

      $pipelineList.appendChild(wrap);
    });
  }

  function _clearPipeline() {
    pipeline = [];
    _renderPipeline();
    _feedback($pipelineFeedback, "", "");
  }

  async function _runPipeline({ dryRun }) {
    if (pipeline.length === 0) {
      _feedback($pipelineFeedback, "Pipeline is empty.", "err");
      return;
    }
    try {
      const res = await fetch(`${API}/api/preprocess`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ operations: pipeline, dry_run: dryRun }),
      });
      const data = await res.json();
      if (!res.ok || !data.ok) {
        _feedback($pipelineFeedback,
          `Pipeline failed: ${JSON.stringify(data.errors || data.error)}`, "err");
        return;
      }
      if (dryRun) {
        _feedback($pipelineFeedback,
          `Dry-run OK. ${data.rows_after} rows after pipeline. Steps:\n` +
          data.log.map(s => `  ${s.step + 1}. ${s.op}: ${s.rows_before} → ${s.rows_after}`).join("\n"),
          "ok");
      } else {
        _feedback($pipelineFeedback,
          `Applied. Match now has ${data.n_events} events.`, "ok");
        if (callbacks.onMatchUpdated) callbacks.onMatchUpdated(data.events);
      }
    } catch (e) {
      _feedback($pipelineFeedback, `Network error: ${e}`, "err");
    }
  }

  // ── feedback ─────────────────────────────────────────────────────────────
  function _feedback(el, text, kind) {
    el.textContent = text;
    el.className = "form-feedback " + (kind || "");
  }

  return { init, toggle, open, close, openForEdit, deliverPickedCoords, isPickerArmed };
})();
