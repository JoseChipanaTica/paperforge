/** The stage toolbar: Skip during assembly; Back (inside a block), Replay, Explode, Labels, Pulse and Publish after. */

export interface ToolbarActions {
  skip(): void;
  outside(): void;
  replay(): void;
  explode(): void;
  labels(): void;
  pulse(): void;
  publish(): void;
  /** Current toggle states, read after every action. */
  state(): { explode: boolean; labels: boolean; pulse: boolean };
}

export function initToolbar(root: HTMLElement, actions: ToolbarActions) {
  const btn = (name: string) => root.querySelector<HTMLButtonElement>(`[data-action="${name}"]`)!;
  const sync = () => {
    const s = actions.state();
    for (const name of ["explode", "labels", "pulse"] as const) {
      btn(name).setAttribute("aria-pressed", String(s[name]));
    }
  };
  root.addEventListener("click", (e) => {
    const name = (e.target as HTMLElement).closest<HTMLElement>("button")?.dataset.action as keyof ToolbarActions | undefined;
    if (!name || name === "state") return;
    actions[name]();
    sync();
  });
  sync();
  return { sync, publishButton: btn("publish"), outsideButton: btn("outside") };
}
