import { DISCLAIMERS } from "@/lib/nav";

/** Always visible, independent of the backend (PRD section 17). */
export function Disclaimer() {
  return (
    <footer className="border-t border-line px-4 py-2.5 text-[11px] leading-4 text-faint md:px-6">
      {DISCLAIMERS.join(" ")}
    </footer>
  );
}
