// Dashboard pages (PRD section 14). `phase` is the roadmap phase that delivers the
// page's engine; pages are only marked live once that engine exists and is verified.

export type NavItem = {
  href: string;
  label: string;
  code: string;
  phase: number;
  live: boolean;
};

export const NAV_ITEMS: NavItem[] = [
  { href: "/", label: "Overview", code: "OVW", phase: 1, live: true },
  { href: "/market", label: "Market Intelligence", code: "MKT", phase: 1, live: true },
  { href: "/news", label: "AI News Intelligence", code: "NWS", phase: 3, live: false },
  { href: "/scenarios", label: "GAN Scenario Explorer", code: "GAN", phase: 4, live: false },
  { href: "/optimizer", label: "RL Portfolio Optimizer", code: "RLP", phase: 5, live: false },
  { href: "/backtest", label: "Backtesting Lab", code: "BKT", phase: 2, live: true },
  { href: "/experiments", label: "Experiment Tracking", code: "EXP", phase: 5, live: false },
  { href: "/paper", label: "Paper Trading", code: "PPR", phase: 7, live: false },
];

export function navItemFor(pathname: string): NavItem | undefined {
  return NAV_ITEMS.find((item) =>
    item.href === "/" ? pathname === "/" : pathname.startsWith(item.href),
  );
}

export const DISCLAIMERS = [
  "Research and decision-support platform. Not a SEBI-registered investment adviser or research analyst; nothing here is investment advice.",
  "Historical and simulated performance does not guarantee future results.",
];
