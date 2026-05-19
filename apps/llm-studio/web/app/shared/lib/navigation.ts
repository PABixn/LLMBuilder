export type AppRouteId =
  | "home"
  | "studio"
  | "tokenizer"
  | "training"
  | "inference"
  | "simple";

export type SimpleStepId = "architecture" | "tokenizer" | "training" | "inference";

export interface AppNavLink {
  id: AppRouteId;
  label: string;
  href: string;
}

export const EXPERT_NAV_LINKS: AppNavLink[] = [
  { id: "home", label: "Home", href: "/" },
  { id: "studio", label: "Model Studio", href: "/studio" },
  { id: "tokenizer", label: "Tokenizer", href: "/tokenizer" },
  { id: "training", label: "Training", href: "/training" },
  { id: "inference", label: "Inference", href: "/inference" },
];

export const SIMPLE_NAV_LINKS: AppNavLink[] = [
  { id: "simple", label: "Guide", href: "/simple" },
  { id: "home", label: "Workspace", href: "/" },
];

export function expertRouteForSimpleStep(step: SimpleStepId | null | undefined): string {
  if (step === "architecture") {
    return "/studio";
  }
  if (step === "tokenizer") {
    return "/tokenizer";
  }
  if (step === "training") {
    return "/training";
  }
  if (step === "inference") {
    return "/inference";
  }
  return "/";
}

export function isRouteActive(pathname: string, href: string): boolean {
  if (href === "/") {
    return pathname === "/";
  }
  return pathname === href || pathname.startsWith(`${href}/`);
}
