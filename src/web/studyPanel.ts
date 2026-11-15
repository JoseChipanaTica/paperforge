import type { LearnKind } from "../shared/learn";
import { TABS, initialPanel, panelReducer, type PanelAction, type PanelState, type TabId } from "./panelState";

/** The study panel: five ARIA tabs whose state lives in panelReducer. */

const LABEL: Record<TabId, string> = {
  overview: "Overview",
  paper: "Paper",
  walkthrough: "Walkthrough",
  example: "Example",
  pytorch: "PyTorch",
};

const isKind = (t: TabId): t is LearnKind => t === "walkthrough" || t === "example" || t === "pytorch";

export class StudyPanel {
  state: PanelState = initialPanel();
  onSelect: (id: TabId) => void = () => {};
  private tabs = new Map<TabId, HTMLButtonElement>();
  private panels = new Map<TabId, HTMLElement>();

  constructor(readonly root: HTMLElement) {
    const list = document.createElement("div");
    list.className = "tablist";
    list.setAttribute("role", "tablist");
    list.setAttribute("aria-label", "Study this paper");
    const body = document.createElement("div");
    body.className = "tabpanels";

    for (const id of TABS) {
      const tab = document.createElement("button");
      tab.type = "button";
      tab.id = `tab-${id}`;
      tab.className = "tab";
      tab.setAttribute("role", "tab");
      tab.setAttribute("aria-controls", `tabpanel-${id}`);
      tab.innerHTML = `<span>${LABEL[id]}</span><i class="tab-dot" aria-hidden="true"></i>`;
      tab.addEventListener("click", () => this.select(id));
      list.append(tab);
      this.tabs.set(id, tab);

      const panel = document.createElement("section");
      panel.id = `tabpanel-${id}`;
      panel.className = `tabpanel tabpanel-${id}`;
      panel.setAttribute("role", "tabpanel");
      panel.setAttribute("aria-labelledby", tab.id);
      panel.tabIndex = 0;
      body.append(panel);
      this.panels.set(id, panel);
    }

    list.addEventListener("keydown", (e) => {
      const i = TABS.indexOf(this.state.active);
      const to =
        e.key === "ArrowRight" ? TABS[(i + 1) % TABS.length]
        : e.key === "ArrowLeft" ? TABS[(i - 1 + TABS.length) % TABS.length]
        : e.key === "Home" ? TABS[0]
        : e.key === "End" ? TABS[TABS.length - 1]
        : null;
      if (!to) return;
      e.preventDefault();
      this.select(to);
      this.tabs.get(to)!.focus();
    });

    root.append(list, body);
    this.render();
  }

  /** The tabpanel element a module renders into. */
  tab(id: TabId): HTMLElement {
    return this.panels.get(id)!;
  }

  select(id: TabId) {
    const changed = id !== this.state.active;
    this.dispatch({ type: "select", tab: id });
    if (changed || isKind(id)) this.onSelect(id);
  }

  dispatch(a: PanelAction) {
    this.state = panelReducer(this.state, a);
    this.render();
  }

  private render() {
    for (const id of TABS) {
      const on = id === this.state.active;
      const tab = this.tabs.get(id)!;
      tab.setAttribute("aria-selected", String(on));
      tab.tabIndex = on ? 0 : -1;
      this.panels.get(id)!.hidden = !on;
      tab.dataset.status = isKind(id) ? this.state.status[id] : "";
    }
  }
}
